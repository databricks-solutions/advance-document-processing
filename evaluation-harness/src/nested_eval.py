"""Type-aware nested `ai_extract` scorer — the tested source of truth.

Pure Python, no Spark / MLflow, so it is unit-testable off-cluster. Recipe 04
(`notebooks/04_extract_eval_nested.py`) carries an **inline copy** of these
functions so the notebook stays self-contained and copy-pasteable; this module is
the canonical, tested implementation. **Keep the two in sync** — if you change the
scoring logic in one, mirror it here (or there) and update the tests below.

Scoring follows the type-aware definition documented in `../README.md`:

- string        → normalized match (LLM-judge fallback lives in the notebook only)
- number/int    → exact match
- boolean/enum  → exact match (+ per-value P/R/F1 in the notebook summary)
- object        → mean of child scores (`object_score_mean`)
- array         → optimal item matching → soft F1
- array/object  → optimal object matching (sim = mean of field scores) → soft F1

Absence is uniform: both sides absent = 1.0, one side absent = 0.0.
"""

import math
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# --- ai_extract VARIANT unwrapping ------------------------------------------

WRAPPER_KEYS = {"value", "confidence_score", "citation_ids"}


def is_wrapper(node):
    return isinstance(node, dict) and "value" in node and set(node).issubset(WRAPPER_KEYS)


def strip_wrappers(node):
    """Collapse ai_extract leaf wrappers to plain values; recurse objects/arrays."""
    if is_wrapper(node):
        return strip_wrappers(node["value"])
    if isinstance(node, dict):
        return {k: strip_wrappers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [strip_wrappers(x) for x in node]
    return node


def walk_confidence(node, prefix, out):
    """Record confidence_score for scalar-leaf wrappers, keyed by dotted path."""
    if is_wrapper(node):
        if node.get("confidence_score") is not None and not isinstance(node["value"], (dict, list)):
            out[prefix] = float(node["confidence_score"])
        return
    if isinstance(node, dict):
        for k, v in node.items():
            walk_confidence(v, f"{prefix}.{k}" if prefix else k, out)
    # arrays intentionally skipped for calibration


def norm_key(s):
    """Normalize a document id to a join key (basename, no ext, alnum, lower)."""
    base = re.sub(r"^.*/", "", str(s))       # basename
    base = re.sub(r"\.[^.]+$", "", base)     # strip extension
    return re.sub(r"[^A-Za-z0-9]", "", base).lower()


# --- scoring primitives -----------------------------------------------------

def node_type(node):
    t = node.get("type")
    if "enum" in node:
        return "enum"
    if t == "array":
        return "array_object" if node.get("items", {}).get("type") == "object" else "array"
    return t


def is_absent(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def norm_str(v):
    return "" if is_absent(v) else re.sub(r"\s+", " ", str(v).strip().lower())


def to_num(v):
    if is_absent(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^0-9.\-]", "", str(v))
    try:
        return float(s)
    except ValueError:
        return None


def leaf_sim(node, a, b):
    """Pure similarity for a scalar leaf (normalized only, no LLM judge)."""
    aa, ba = is_absent(a), is_absent(b)
    if aa and ba:
        return 1.0
    if aa or ba:
        return 0.0
    t = node_type(node)
    if t in ("number", "integer"):
        na, nb = to_num(a), to_num(b)
        return 1.0 if (na is not None and nb is not None and abs(na - nb) < 1e-9) else 0.0
    return 1.0 if norm_str(a) == norm_str(b) else 0.0


def sim_node(node, a, b):
    """Pure recursive similarity in [0,1] — used to build array-matching matrices."""
    t = node_type(node)
    if t == "object":
        props = node["properties"]
        a, b = a or {}, b or {}
        scores = [sim_node(c, a.get(n), b.get(n)) for n, c in props.items()]
        return sum(scores) / len(scores) if scores else 1.0
    if t in ("array", "array_object"):
        return soft_f1(node["items"], a or [], b or [])
    return leaf_sim(node, a, b)


def optimal_pairs(item_node, A, B):
    """Optimal 1:1 matching maximizing total similarity. Returns (pairs, total)."""
    M = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            M[i, j] = sim_node(item_node, x, y)
    ri, ci = linear_sum_assignment(-M)
    pairs = [(int(i), int(j), float(M[i, j])) for i, j in zip(ri, ci)]
    return pairs, sum(p[2] for p in pairs)


def soft_f1(item_node, A, B):
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    _, total = optimal_pairs(item_node, A, B)
    p, r = total / len(A), total / len(B)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# --- recursive scorer (records metrics into an accumulator) -----------------

def new_acc():
    return {
        "accuracy": defaultdict(list),      # path -> [0/1]        (string/number/int/enum/bool)
        "confusion": defaultdict(list),     # path -> [(pred,exp)] (enum/bool per-value P/R/F1)
        "object_score": defaultdict(list),  # path -> [float]      (object_score_mean)
        "soft_f1": defaultdict(list),       # path -> [float]      (array & array-of-object)
        "leaf": [],                         # (doc, path, score, conf) for calibration
    }


def score_node(node, pred, exp, path, acc, ctx):
    t = node_type(node)

    if t == "object":
        props = node["properties"]
        pred, exp = pred or {}, exp or {}
        child_scores = [
            score_node(c, pred.get(n), exp.get(n), f"{path}.{n}" if path else n, acc, ctx)
            for n, c in props.items()
        ]
        s = sum(child_scores) / len(child_scores) if child_scores else 1.0
        acc["object_score"][path].append(s)
        return s

    if t == "array":
        s = soft_f1(node["items"], pred or [], exp or [])
        acc["soft_f1"][path].append(s)
        return s

    if t == "array_object":
        item_node, A, B = node["items"], pred or [], exp or []
        if not A and not B:
            s, pairs = 1.0, []
        elif not A or not B:
            s, pairs = 0.0, []
        else:
            pairs, total = optimal_pairs(item_node, A, B)
            p, r = total / len(A), total / len(B)
            s = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        acc["soft_f1"][path].append(s)
        # per-child soft F1 across matched objects
        for cname, cnode in item_node["properties"].items():
            if not A and not B:
                acc["soft_f1"][f"{path}.{cname}"].append(1.0)
                continue
            if not A or not B:
                acc["soft_f1"][f"{path}.{cname}"].append(0.0)
                continue
            ctot = sum(
                sim_node(cnode,
                         A[i].get(cname) if isinstance(A[i], dict) else None,
                         B[j].get(cname) if isinstance(B[j], dict) else None)
                for i, j, _ in pairs
            )
            cp, cr = ctot / len(A), ctot / len(B)
            acc["soft_f1"][f"{path}.{cname}"].append(2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0)
        return s

    # --- scalar leaves ---
    aa, ea = is_absent(pred), is_absent(exp)
    if aa and ea:
        s = 1.0
    elif aa or ea:
        s = 0.0
    elif t in ("number", "integer"):
        s = leaf_sim(node, pred, exp)
    elif t in ("boolean", "enum"):
        s = 1.0 if norm_str(pred) == norm_str(exp) else 0.0
    else:  # string
        if norm_str(pred) == norm_str(exp):
            s = 1.0
        elif ctx.get("judge") is not None:
            s = 1.0 if ctx["judge"](path, pred, exp) else 0.0
        else:
            s = 0.0

    acc["accuracy"][path].append(s)
    if t in ("boolean", "enum"):
        acc["confusion"][path].append((norm_str(pred), norm_str(exp)))
    acc["leaf"].append((ctx["doc_id"], path, s, ctx["conf"].get(path)))
    return s


def summarize(acc):
    """Roll an accumulator up into the flat metric dict Recipe 04 logs."""
    summary = {}
    for path, vals in acc["accuracy"].items():
        summary[f"{path}_accuracy"] = sum(vals) / len(vals)
    for path, vals in acc["object_score"].items():
        summary[f"{path}_object_score_mean"] = sum(vals) / len(vals)
    for path, vals in acc["soft_f1"].items():
        summary[f"{path}_soft_f1_mean"] = sum(vals) / len(vals)
    for path, pairs in acc["confusion"].items():
        values = sorted({v for pair in pairs for v in pair if v != ""})
        for v in values:
            tp = sum(1 for p, e in pairs if p == v and e == v)
            fp = sum(1 for p, e in pairs if p == v and e != v)
            fn = sum(1 for p, e in pairs if p != v and e == v)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            summary[f"{path}_{v}_precision"] = precision
            summary[f"{path}_{v}_recall"] = recall
            summary[f"{path}_{v}_f1"] = f1
    return summary
