"""Pure, unit-testable builders for the insurance-KB vector-search pipeline.

These run on the driver — they assemble the classification label set, the
doc_type→domain routing map, and SQL-string / name fragments that the stage
notebooks feed to spark.sql / expr. Keeping them here lets the construction be
tested without Spark (see tests/test_sql_builders.py)."""

import json

# Five document types, each with a description to sharpen ai_classify accuracy.
DOC_TYPE_LABELS = {
    "policy_document": (
        "An insurance policy: the declarations page and/or policy wording for a "
        "homeowners or auto policy. Named insured, policy number, policy period, "
        "coverages and coverage limits, deductibles, premiums, and exclusions."
    ),
    "endorsement": (
        "An endorsement or rider that amends an existing base policy — e.g. a "
        "water-backup endorsement or scheduled-personal-property (jewelry) rider. "
        "References a base policy number and states the specific coverage added, "
        "changed, or removed; not a standalone full policy."
    ),
    "underwriting_guideline": (
        "Internal underwriting guidelines: rules for risk acceptance, referral, or "
        "pricing (e.g. maximum roof age, coastal/wildfire exposure, prior-claims "
        "thresholds). A reference rulebook, not tied to one named insured."
    ),
    "fnol_claim_form": (
        "A First Notice of Loss (FNOL) / claim report form. Structured fields: "
        "claim number, claimant, date of loss, policy number, peril/cause of loss, "
        "loss location, and a short loss description."
    ),
    "adjuster_report": (
        "A claims adjuster's inspection or investigation report: narrative findings, "
        "damage assessment, cause-of-loss analysis, photos/estimates references, and "
        "a recommended reserve or settlement."
    ),
}

# The single routing decision the pipeline makes.
DOC_TYPE_TO_DOMAIN = {
    "policy_document": "reference",
    "endorsement": "reference",
    "underwriting_guideline": "reference",
    "fnol_claim_form": "claims",
    "adjuster_report": "claims",
}

CLASSIFY_INSTRUCTIONS = (
    "These are insurance documents for an adjuster/underwriter knowledge base. "
    "Each document is a single type (not a mixed packet). Classify by the document's "
    "dominant purpose. Use 'policy_document' for a full policy/declarations, "
    "'endorsement' only when it amends an existing policy, 'underwriting_guideline' "
    "for internal risk/pricing rulebooks, 'fnol_claim_form' for a first-notice-of-loss "
    "report, and 'adjuster_report' for an adjuster's inspection/investigation findings."
)


def labels_json():
    """JSON object string (label -> description) for ai_classify."""
    return json.dumps(DOC_TYPE_LABELS)


def domain_for(doc_type):
    """Retrieval domain for a doc_type; 'unknown' if unrecognized."""
    return DOC_TYPE_TO_DOMAIN.get(doc_type, "unknown")


def classify_expr(input_col, labels_sql_json, instructions):
    """Build the ai_classify(...) SQL. Doubles single quotes in string literals."""
    labels_sql = labels_sql_json.replace("'", "''")
    instr_sql = instructions.replace("'", "''")
    return f"""
        ai_classify(
            {input_col},
            '{labels_sql}',
            map('instructions', '{instr_sql}')
        )
    """


def domain_case_expr(doc_type_sql):
    """SQL CASE mapping a doc_type SQL expression to its retrieval domain."""
    ref = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "reference"]
    clm = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "claims"]
    ref_in = ", ".join(f"'{k}'" for k in ref)
    clm_in = ", ".join(f"'{k}'" for k in clm)
    return f"""
        CASE
            WHEN {doc_type_sql} IN ({ref_in}) THEN 'reference'
            WHEN {doc_type_sql} IN ({clm_in}) THEN 'claims'
            ELSE 'unknown'
        END
    """


def gold_table_name(prefix, domain):
    """Unqualified gold chunk-table name for a domain."""
    return f"{prefix}_{domain}_chunks"


def index_name(catalog, schema, prefix, domain):
    """Fully-qualified Vector Search index name for a domain."""
    return f"{catalog}.{schema}.{prefix}_{domain}_index"
