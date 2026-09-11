"""Generate synthetic, fully fictional insurance PDFs for the
ai-search-knowledge-base-creation pipeline.

3 documents per type x 5 types = 15 PDFs, plus ground_truth_doc_types.csv.
Deterministic (seeded) so filenames are stable across regenerations.

Run (from repo root):
    uv run --with reportlab python scripts/generate_sample_insurance_docs.py
Output: ai-search-knowledge-base-creation/sample_data/<doc_type>_<n>_<slug>.pdf
        ai-search-knowledge-base-creation/sample_data/ground_truth_doc_types.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "ai-search-knowledge-base-creation"
    / "sample_data"
)

# Local literal map (scripts stay standalone; must match the spec's mapping).
DOC_TYPE_TO_DOMAIN = {
    "policy_document": "reference",
    "endorsement": "reference",
    "underwriting_guideline": "reference",
    "fnol_claim_form": "claims",
    "adjuster_report": "claims",
}

# --- color constants (mirror generate_sample_loan_files.py) ----------------
NAVY = colors.HexColor("#1b3a5b")
LIGHT = colors.HexColor("#f3f6fa")
RULE = colors.HexColor("#cfd8e3")
USABLE_W = LETTER[0] - 1.8 * inch  # 0.9" margins

# --- synthetic data pools -------------------------------------------------

CARRIERS = [
    "Harbor Mutual Insurance",
    "Summit Ridge Casualty",
    "Blue Meridian Insurance Group",
]

INSUREDS = [
    dict(
        name="Patricia J. Carmichael",
        address="48 Ridgewood Lane, Fairview, TX 75069",
        city="Fairview", state="TX", zip="75069",
        policy_no="HO-2026-40012",
        agent="Westwood Agency, Inc.",
        agent_phone="(972) 555-0184",
    ),
    dict(
        name="Marcus T. Okonkwo",
        address="2217 Sycamore Court, Lakeside, OH 44311",
        city="Lakeside", state="OH", zip="44311",
        policy_no="HO-2026-40237",
        agent="Lakeside Insurance Services",
        agent_phone="(330) 555-0219",
    ),
    dict(
        name="Denise R. Varga",
        address="904 Coral Drive, Marigold Beach, FL 34231",
        city="Marigold Beach", state="FL", zip="34231",
        policy_no="HO-2026-40891",
        agent="Suncoast Insurance Partners",
        agent_phone="(941) 555-0143",
    ),
]

ADJUSTERS = [
    dict(name="Brian K. Larson",   lic="TX-CLM-48921", phone="(972) 555-0266"),
    dict(name="Monica J. Sato",     lic="OH-CLM-31104", phone="(330) 555-0378"),
    dict(name="Terrence W. Buell",  lic="FL-CLM-72014", phone="(941) 555-0191"),
]

PERILS = [
    "Water damage (plumbing)",
    "Wind/hail",
    "Fire",
    "Theft",
    "Vehicle collision",
]

ENDORSEMENT_TYPES = [
    (
        "Water Backup and Sump Overflow",
        [
            "Coverage is extended to include direct physical loss caused by water "
            "that backs up through sewers or drains, or overflows from a sump pump "
            "or related equipment. The sublimit of liability for this endorsement is "
            "$25,000 per occurrence, subject to a $500 deductible.",
            "Coverage under this endorsement does not apply to loss caused by surface "
            "water, flood, waves, tidal water, or overflow of any body of water, "
            "including but not limited to storm surge.",
        ],
        False,   # no scheduled-items table
    ),
    (
        "Scheduled Personal Property — Jewelry",
        [
            "The following items of personal jewelry are specifically scheduled and "
            "valued for purposes of this endorsement. Coverage is provided on an "
            "agreed-value, all-risk basis. No deductible applies to scheduled items.",
        ],
        True,    # show scheduled-items table
    ),
    (
        "Home Business Liability Broadening",
        [
            "Coverage is extended to include incidental business liability arising "
            "from an insured's home-based professional practice not otherwise excluded, "
            "subject to a sublimit of $50,000 per occurrence and $100,000 annual "
            "aggregate under this endorsement.",
            "This endorsement does not apply to medical or legal malpractice, products "
            "liability, professional errors and omissions, or any business conducted "
            "with employees other than the named insured.",
        ],
        False,
    ),
]

UW_RULE_SECTIONS = [
    (
        "1.  Eligibility Requirements",
        [
            "1.1  Insured property must be an owner-occupied, single-family dwelling "
                 "or condominium unit. Non-owner-occupied or tenant-occupied dwellings "
                 "require referral to the Commercial Lines unit.",
            "1.2  Minimum dwelling replacement cost of $150,000. Properties with "
                 "replacement cost below this threshold are ineligible for this program.",
            "1.3  Insured must reside at the property for a minimum of nine (9) months "
                 "per calendar year. Seasonal or vacation homes require endorsement "
                 "HO-500 (Seasonal Dwelling).",
            "1.4  Properties with active foreclosure proceedings, tax liens exceeding "
                 "$10,000, or open code-enforcement violations are not eligible.",
        ],
    ),
    (
        "2.  Roof Age and Condition",
        [
            "2.1  Roofs 20 years or older at policy inception require a completed Roof "
                 "Certification Form (RCF-10) or a third-party inspection report dated "
                 "within 90 days of the effective date. Coverage for roof surfaces may "
                 "be limited to Actual Cash Value absent current certification.",
            "2.2  Roofs exhibiting active leaks, missing shingles covering more than "
                 "10% of total surface area, or structural sagging must be repaired "
                 "prior to binding coverage.",
            "2.3  Metal and tile roofs qualify for the Preferred Rate tier if installed "
                 "within the past 25 years. Cedar shake roofs are subject to a 10% "
                 "surcharge unless fire-treated and independently certified.",
        ],
    ),
    (
        "3.  Coastal and Wildfire Exposure",
        [
            "3.1  Properties within 2,500 feet of tidal water require a Windstorm "
                 "Mitigation Inspection (Form WMI-2). A hurricane deductible of 2% of "
                 "Coverage A applies unless Named Storm Deductible Endorsement HO-410 "
                 "is separately attached.",
            "3.2  Properties located within a FEMA Special Flood Hazard Area (Zone A "
                 "or V) require separate NFIP or private flood coverage as a condition "
                 "of binding homeowners coverage.",
            "3.3  Properties in Wildfire Risk Score Band 7-10 (per the approved vendor "
                 "scoring model) require defensible-space documentation and are subject "
                 "to a $2,500 minimum fire-peril deductible.",
            "3.4  Properties within a Very High Fire Hazard Severity Zone (VHFHSZ) "
                 "require Firewise USA certification and are eligible only under the "
                 "High-Value Homeowners program.",
        ],
    ),
    (
        "4.  Prior Claims History",
        [
            "4.1  Two or more water-damage claims totaling $15,000 or more in the "
                 "prior three policy years trigger mandatory referral to a senior "
                 "underwriter.",
            "4.2  Any single liability claim exceeding $50,000 in the prior five years "
                 "requires a signed Loss Prevention Agreement prior to binding.",
            "4.3  Properties with three or more claims of any type in the prior five "
                 "years are ineligible for new business. Mid-term non-renewal authority "
                 "applies at two claims exceeding $10,000 each within the same policy year.",
            "4.4  Dog-bite claims are excluded from favorable tier qualification for "
                 "36 months from the date of loss, regardless of severity.",
        ],
    ),
    (
        "5.  Referral Triggers — Submit to Underwriting",
        [
            "5.1  Any risk with a trampoline, diving board, or above-ground pool with "
                 "a deck exceeding 18 inches above grade.",
            "5.2  Exotic or dangerous animals, including but not limited to wolf "
                 "hybrids, large constrictors over 6 feet, or venomous reptiles.",
            "5.3  Home-based daycare, group home, or assisted-living operation on "
                 "the premises.",
            "5.4  Active renovation exceeding 25% of Coverage A replacement cost value.",
            "5.5  Business equipment on premises with replacement value above $15,000.",
        ],
    ),
]

# --- styles (mirror generate_sample_loan_files.py) -------------------------

styles = getSampleStyleSheet()
H_TITLE = ParagraphStyle("ins_title",  parent=styles["Title"],    fontSize=15,
                          spaceAfter=4, textColor=NAVY)
H_SUB   = ParagraphStyle("ins_sub",    parent=styles["Normal"],   fontSize=8.5,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=12)
H_SEC   = ParagraphStyle("ins_sec",    parent=styles["Heading3"], fontSize=10,
                          spaceBefore=10, spaceAfter=2, textColor=colors.white,
                          backColor=NAVY, leftIndent=4, leading=16)
BODY    = ParagraphStyle("ins_body",   parent=styles["Normal"],   fontSize=8.5, leading=12)
BODY_R  = ParagraphStyle("ins_body_r", parent=BODY, alignment=TA_RIGHT)
CELL    = ParagraphStyle("ins_cell",   parent=styles["Normal"],   fontSize=7.5, leading=9.5)
CELL_R  = ParagraphStyle("ins_cell_r", parent=CELL, alignment=TA_RIGHT)
SMALL   = ParagraphStyle("ins_small",  parent=styles["Normal"],   fontSize=7,
                          textColor=colors.grey, leading=9.5)
BULLET  = ParagraphStyle("ins_bullet", parent=styles["Normal"],   fontSize=8.5,
                          leading=12, leftIndent=18, firstLineIndent=0)


# --- shared helpers (mirror generate_sample_loan_files.py) ----------------

def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, dp: int = 3) -> str:
    return f"{x:.{dp}f}%"


def section(title: str) -> Paragraph:
    return Paragraph(title, H_SEC)


def title_block(title: str, subtitle: str) -> list:
    return [Paragraph(title, H_TITLE), Paragraph(subtitle, H_SUB)]


def kv_table(rows, col0=2.3 * inch, col1=None) -> Table:
    col1 = col1 or (USABLE_W - col0)
    data = [[Paragraph(f"<b>{k}</b>", BODY), Paragraph(str(v), BODY)] for k, v in rows]
    t = Table(data, colWidths=[col0, col1])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.25, RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def two_col_fields(rows) -> Table:
    """Render field/value pairs in two side-by-side columns (form-like)."""
    cells = [(Paragraph(f"<b>{k}</b>", CELL), Paragraph(str(v), CELL)) for k, v in rows]
    data = []
    for i in range(0, len(cells), 2):
        left  = cells[i]
        right = cells[i + 1] if i + 1 < len(cells) else (Paragraph("", CELL), Paragraph("", CELL))
        data.append([left[0], left[1], right[0], right[1]])
    w = USABLE_W / 4
    t = Table(data, colWidths=[w * 0.95, w * 1.05, w * 0.95, w * 1.05])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.25, RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def grid_table(header, rows, col_widths=None, align_right_from=1, font=7.5) -> Table:
    head = [Paragraph(f"<b>{h}</b>",
                      ParagraphStyle("ins_gh", parent=CELL, textColor=colors.white))
            for h in header]
    body = []
    for r in rows:
        row = []
        for j, c in enumerate(r):
            style = CELL_R if j >= align_right_from else CELL
            row.append(Paragraph(str(c), style))
        body.append(row)
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), NAVY),
        ("FONTSIZE",       (0, 0), (-1, -1), font),
        ("GRID",           (0, 0), (-1, -1), 0.25, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",     (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 2),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def make_decorator(doc_label: str, ref_no: str):
    """Return an onPage callback that draws a header rule and page footer."""
    def _decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(0.9 * inch, LETTER[1] - 0.6 * inch,
                    LETTER[0] - 0.9 * inch, LETTER[1] - 0.6 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.9 * inch, LETTER[1] - 0.52 * inch, doc_label)
        canvas.drawRightString(LETTER[0] - 0.9 * inch, LETTER[1] - 0.52 * inch,
                               ref_no)
        canvas.line(0.9 * inch, 0.62 * inch, LETTER[0] - 0.9 * inch, 0.62 * inch)
        canvas.drawString(0.9 * inch, 0.5 * inch,
                          "CONFIDENTIAL — FOR AUTHORIZED USE ONLY")
        canvas.drawRightString(LETTER[0] - 0.9 * inch, 0.5 * inch,
                               f"Page {doc.page}")
        canvas.restoreState()
    return _decorate


# --- per-document builders ------------------------------------------------

def build_policy_document(rng: random.Random) -> list:
    carrier     = rng.choice(CARRIERS)
    insured     = rng.choice(INSUREDS)
    dwell_limit = rng.choice([280_000, 320_000, 395_000, 450_000])
    deductible  = rng.choice([1_000, 2_500, 5_000])
    premium     = round(dwell_limit * rng.uniform(0.0045, 0.0065), 2)

    story = title_block(
        carrier,
        "HOMEOWNERS POLICY DECLARATIONS — HO-3 Special Form",
    )
    story += [
        section("Policy Information"),
        kv_table([
            ("Policy Number",    insured["policy_no"]),
            ("Policy Period",    "04/15/2026 to 04/15/2027 — 12:01 AM Standard Time "
                                 "at the described property"),
            ("Named Insured",    insured["name"]),
            ("Mailing Address",  insured["address"]),
            ("Property Address", insured["address"]),
            ("Agent / Broker",   f"{insured['agent']}  ·  {insured['agent_phone']}"),
            ("Form",             "HO-3 (Special Form — Open Perils on Dwelling; "
                                 "Named Perils on Personal Property)"),
            ("Territory",        insured["state"]),
        ], col0=2.5 * inch),
        section("Coverages and Limits of Liability"),
        grid_table(
            ["Coverage", "Description", "Limit", "Deductible"],
            [
                ["A — Dwelling",
                 "Dwelling structure and attached structures",
                 money(dwell_limit), money(deductible)],
                ["B — Other Structures",
                 "Detached garages, fences, sheds",
                 money(round(dwell_limit * 0.10, 2)), money(deductible)],
                ["C — Personal Property",
                 "Contents and personal belongings",
                 money(round(dwell_limit * 0.50, 2)), money(deductible)],
                ["D — Loss of Use",
                 "Additional living expenses; fair rental value",
                 money(round(dwell_limit * 0.20, 2)), "N/A"],
                ["E — Personal Liability",
                 "Bodily injury and property damage liability",
                 money(300_000), "N/A"],
                ["F — Medical Payments",
                 "Medical payments to others",
                 money(5_000), "N/A"],
            ],
            col_widths=[USABLE_W * 0.12, USABLE_W * 0.40, USABLE_W * 0.24, USABLE_W * 0.24],
            align_right_from=4,
        ),
        section("Premium Summary"),
        grid_table(
            ["Premium Component", "Amount"],
            [
                ["Base Premium — Dwelling",   money(round(premium * 0.72, 2))],
                ["Contents Premium",          money(round(premium * 0.18, 2))],
                ["Liability / Medical",       money(round(premium * 0.06, 2))],
                ["Surcharges and Credits",    money(round(premium * -0.04, 2))],
                ["TOTAL ANNUAL PREMIUM",      money(premium)],
            ],
            col_widths=[USABLE_W * 0.70, USABLE_W * 0.30],
        ),
        Spacer(1, 0.10 * inch),
        section("Principal Exclusions (Summary)"),
        Paragraph(
            "This policy does NOT cover loss or damage caused by or resulting from: "
            "(1) flood, surface water, or overflow of any body of water; (2) earth "
            "movement including earthquake, landslide, or subsidence; (3) ordinance or "
            "law enforcement costs; (4) neglect, wear and tear, or intentional loss; "
            "(5) war or nuclear hazard; or (6) mold, fungi, or wet rot except as a "
            "direct result of a covered water loss discovered and reported within "
            "72 hours. These exclusions are subject to the complete terms set forth in "
            "the policy contract. A copy of the full policy form is available on "
            "request from your agent.",
            BODY,
        ),
        Spacer(1, 0.10 * inch),
        Paragraph(
            "This declarations page is part of your policy contract. Please read your "
            "policy carefully and contact your agent if you have any questions about "
            "your coverages or limits.",
            SMALL,
        ),
    ]
    return story


def build_endorsement(rng: random.Random) -> list:
    carrier                          = rng.choice(CARRIERS)
    insured                          = rng.choice(INSUREDS)
    end_idx                          = rng.randrange(len(ENDORSEMENT_TYPES))
    end_title, end_paragraphs, sched = ENDORSEMENT_TYPES[end_idx]
    end_no                           = f"END-{rng.randint(1000, 9999)}"
    eff_date                         = rng.choice(["05/01/2026", "06/01/2026", "07/15/2026"])
    add_prem                         = round(rng.uniform(35, 185), 2)

    story = title_block(
        carrier,
        "POLICY ENDORSEMENT",
    )
    story += [
        section("Endorsement Identification"),
        kv_table([
            ("Endorsement Number",  end_no),
            ("Base Policy Number",  insured["policy_no"]),
            ("Named Insured",       insured["name"]),
            ("Property Address",    insured["address"]),
            ("Effective Date",      eff_date),
            ("Date Issued",         "04/02/2026"),
            ("Additional Premium",  money(add_prem)),
            ("Endorsement Title",   end_title),
        ], col0=2.5 * inch),
        section("Coverage Amendment"),
    ]

    for para in end_paragraphs:
        story += [Paragraph(para, BODY), Spacer(1, 0.06 * inch)]

    if sched:
        story += [
            Spacer(1, 0.06 * inch),
            grid_table(
                ["Item No.", "Description", "Agreed Value"],
                [
                    ["1", "Ladies' platinum engagement ring, round brilliant cut, 1.20 ct",
                     money(8_500)],
                    ["2", "Strand of cultured freshwater pearls, 18 in., gold clasp",
                     money(2_200)],
                    ["3", "Gentleman's Swiss automatic dress watch, stainless case",
                     money(4_800)],
                ],
                col_widths=[USABLE_W * 0.10, USABLE_W * 0.65, USABLE_W * 0.25],
            ),
        ]

    story += [
        section("All Other Terms Unchanged"),
        Paragraph(
            "Except as amended by this endorsement, all terms, conditions, exclusions, "
            "and limitations of the policy to which this endorsement is attached remain "
            "in full force and effect without change. In the event of any conflict "
            "between the terms of this endorsement and those of the base policy, the "
            "terms of this endorsement shall control.",
            BODY,
        ),
        Spacer(1, 0.20 * inch),
        Paragraph(
            f"Countersigned by an authorized representative of {carrier} on behalf of "
            "the company.",
            SMALL,
        ),
        Spacer(1, 0.60 * inch),
        Paragraph(
            "Authorized Representative: ______________________________   "
            "Date: ___________",
            BODY,
        ),
    ]
    return story


def build_underwriting_guideline(rng: random.Random) -> list:
    carrier       = rng.choice(CARRIERS)
    order         = list(range(len(UW_RULE_SECTIONS)))
    rng.shuffle(order)

    story = title_block(
        f"{carrier} — Personal Lines Division",
        "UNDERWRITING GUIDELINES — Personal Lines Homeowners Program",
    )
    story += [
        Paragraph(
            "These guidelines govern the eligibility, rating, and referral "
            "requirements for the Personal Lines Homeowners Program. Underwriters and "
            "agents must comply with these rules when quoting, binding, or endorsing "
            "coverage. Guidelines are effective for all new business and renewals with "
            "effective dates on or after January 1, 2026. Prior versions are "
            "superseded.",
            BODY,
        ),
        Spacer(1, 0.10 * inch),
    ]

    for idx in order:
        sec_title, rules = UW_RULE_SECTIONS[idx]
        story += [section(sec_title)]
        for rule in rules:
            story += [
                Paragraph(f"• {rule}", BULLET),
                Spacer(1, 0.04 * inch),
            ]
        story += [Spacer(1, 0.06 * inch)]

    story += [
        Spacer(1, 0.10 * inch),
        Paragraph(
            "Questions regarding guideline interpretation should be directed to the "
            "Regional Underwriting Supervisor. Exceptions may be considered on a "
            "case-by-case basis with written approval from a senior underwriter. "
            "These guidelines are subject to change; refer to the current version "
            "in the underwriting portal before binding.",
            SMALL,
        ),
    ]
    return story


def build_fnol_claim_form(rng: random.Random) -> list:
    carrier      = rng.choice(CARRIERS)
    insured      = rng.choice(INSUREDS)
    peril        = rng.choice(PERILS)
    claim_no     = f"CLM-{rng.randint(100_000, 999_999)}"
    date_of_loss = rng.choice(["03/02/2026", "04/10/2026", "05/18/2026"])
    reported     = rng.choice(["03/03/2026", "04/11/2026", "05/19/2026"])

    loss_desc = {
        "Water damage (plumbing)":
            "The insured reports that a supply line beneath the kitchen sink failed "
            "overnight, causing significant water intrusion into the kitchen and "
            "adjacent dining room. Laminate flooring, lower cabinetry, and drywall "
            "are visibly damaged. Estimated affected area is approximately 300 sq ft. "
            "A plumber has been engaged to complete emergency repairs.",
        "Wind/hail":
            "Severe thunderstorm with hail between 1.5 and 2.0 inches in diameter "
            "impacted the property. The insured reports multiple broken roof shingles, "
            "damaged gutters, dented aluminum siding on the north and west elevations, "
            "and two broken exterior window panes. No interior water intrusion was "
            "observed at time of reporting.",
        "Fire":
            "The insured reports a kitchen fire originating from an unattended "
            "stovetop. The fire was suppressed by the local fire department; the "
            "responding unit confirmed fire contained to the kitchen. Smoke and heat "
            "damage to kitchen and adjacent family room. Local fire marshal "
            "investigation is pending. Property is temporarily uninhabitable.",
        "Theft":
            "The insured reports a residential burglary occurring while the property "
            "was unoccupied. Point of entry was a rear sliding door with the lock "
            "forced. Items reported stolen include a laptop computer, jewelry, and "
            "consumer electronics. A police report has been filed (Case No. "
            "2026-04-8821). Insured has provided a preliminary inventory of missing "
            "items.",
        "Vehicle collision":
            "A vehicle traveling on the adjacent street left the roadway and collided "
            "with the insured's attached garage. Structural damage includes the garage "
            "door frame and front wall of the garage. One vehicle inside the garage "
            "sustained cosmetic damage. Police report is on file and third-party "
            "driver information has been obtained.",
    }

    story = title_block(
        carrier,
        "FIRST NOTICE OF LOSS — Homeowners",
    )
    story += [
        section("Claim Identification"),
        two_col_fields([
            ("Claim Number",   claim_no),
            ("Policy Number",  insured["policy_no"]),
            ("Named Insured",  insured["name"]),
            ("Loss Location",  insured["address"]),
            ("Date of Loss",   date_of_loss),
            ("Reported Date",  reported),
            ("Peril",          peril),
            ("Policy Period",  "04/15/2025 to 04/15/2026"),
        ]),
        section("Claimant Contact Information"),
        two_col_fields([
            ("Claimant Name",      insured["name"]),
            ("Relationship",       "Named Insured"),
            ("Daytime Phone",      insured["agent_phone"]),
            ("Email",              insured["name"].split()[0].lower() + "@example.com"),
            ("Preferred Contact",  "Phone"),
            ("Alternate Phone",    "(555) 555-0100"),
        ]),
        section("Description of Loss"),
        Paragraph(loss_desc.get(peril, "See attached statement."), BODY),
        Spacer(1, 0.15 * inch),
        section("Authorization"),
        Paragraph(
            "I hereby authorize the insurance company and its authorized "
            "representatives to investigate the facts and circumstances of the "
            "above-described loss, to inspect the damaged property, and to obtain "
            "any records necessary to process this claim. I certify that the "
            "information provided above is true and accurate to the best of my "
            "knowledge.",
            BODY,
        ),
        Spacer(1, 0.50 * inch),
        Paragraph(
            "Claimant Signature: _______________________________   "
            "Date: ___________",
            BODY,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph(
            f"This form was received by {carrier} Claims Department on {reported}. "
            "A claims representative will contact the insured within 2 business days.",
            SMALL,
        ),
    ]
    return story


def build_adjuster_report(rng: random.Random) -> list:
    carrier    = rng.choice(CARRIERS)
    insured    = rng.choice(INSUREDS)
    adjuster   = rng.choice(ADJUSTERS)
    peril      = rng.choice(PERILS)
    claim_no   = f"CLM-{rng.randint(100_000, 999_999)}"
    date_loss  = rng.choice(["03/02/2026", "04/10/2026", "05/18/2026"])
    insp_date  = rng.choice(["03/15/2026", "04/22/2026", "05/28/2026"])

    damage_items = {
        "Water damage (plumbing)": [
            ["Laminate flooring removal and replacement (320 sq ft)",  money(3_840)],
            ["Lower cabinet replacement — kitchen (6 linear ft)",      money(1_920)],
            ["Drywall repair and paint — kitchen and dining room",     money(1_250)],
            ["Plumbing supply line replacement",                        money(380)],
            ["Moisture mitigation and drying services",                 money(1_100)],
            ["General contractor overhead and profit (15%)",            money(1_274)],
        ],
        "Wind/hail": [
            ["Asphalt shingle roof replacement (22 squares)",          money(9_900)],
            ["Gutter and downspout replacement (160 linear ft)",       money(2_240)],
            ["Aluminum siding repair — north and west elevations",     money(1_760)],
            ["Window pane replacement (2 units)",                      money(640)],
            ["General contractor overhead and profit (15%)",           money(2_162)],
        ],
        "Fire": [
            ["Kitchen cabinet removal and replacement",                 money(6_400)],
            ["Appliance replacement — range, microwave, hood",         money(2_800)],
            ["Structural drywall and ceiling repair",                  money(2_100)],
            ["Smoke odor remediation — full interior",                 money(2_500)],
            ["Paint — kitchen, family room, and hallway",              money(1_800)],
            ["General contractor overhead and profit (15%)",           money(2_340)],
        ],
        "Theft": [
            ["Laptop computer (replacement value)",                    money(1_499)],
            ["Jewelry — per scheduled items endorsement",              money(4_200)],
            ["Consumer electronics",                                   money(1_850)],
            ["Sliding door lock and hardware repair",                  money(320)],
            ["Contents — miscellaneous items per inventory",           money(875)],
        ],
        "Vehicle collision": [
            ["Garage door frame structural repair",                    money(3_200)],
            ["Garage door replacement",                                money(1_800)],
            ["Exterior wall framing and sheathing repair",             money(2_400)],
            ["Drywall, insulation, and finish — interior wall",        money(950)],
            ["General contractor overhead and profit (15%)",           money(1_252)],
        ],
    }

    narrative = {
        "Water damage (plumbing)":
            "Upon arrival, the kitchen and dining room showed evidence of extensive "
            "water intrusion from a failed 3/8-inch compression supply line beneath "
            "the kitchen sink. Water spread beneath the laminate flooring and wicked "
            "into the lower cabinetry and adjacent drywall. Moisture readings of "
            "18-24% were recorded in affected areas. Emergency drying equipment has "
            "been placed by a restoration contractor. The damage is consistent with "
            "the insured's report and is covered under the policy.",
        "Wind/hail":
            "Inspection confirmed hail damage consistent with a storm event reported "
            "in the area on the date of loss. Impact marks are present on the roof "
            "surface (22 squares), gutters, north and west siding elevations, and two "
            "window panes. Hail size is estimated at 1.5-2.0 inches based on impressed "
            "diameter measurements on the aluminum components. Loss is covered under "
            "the policy subject to the windstorm deductible.",
        "Fire":
            "A kitchen fire originating from an unattended gas range caused heat, "
            "smoke, and soot damage to the kitchen, adjacent family room, and hallway. "
            "The fire department suppressed the fire within 45 minutes of dispatch. "
            "Structural damage is limited to the kitchen; smoke odor permeates the "
            "full main floor. Cause of loss is consistent with the insured's statement "
            "and is covered under the policy.",
        "Theft":
            "Exterior inspection revealed forced entry at the rear sliding door. "
            "Interior inspection confirmed missing items as documented in the insured's "
            "property inventory. The police report corroborates the claimed event. "
            "Scheduled jewelry is confirmed as covered under the scheduled personal "
            "property endorsement. Loss is covered under the policy.",
        "Vehicle collision":
            "A third-party vehicle left the adjacent roadway and impacted the front "
            "wall of the attached garage. Structural damage includes the garage door "
            "frame, framing members, and exterior sheathing. Interior wall framing is "
            "intact. Police report confirms the event. Coverage applies under Coverage "
            "A (dwelling) and Coverage B (other structures) of the policy.",
    }

    items = damage_items.get(peril, [["General repairs", money(5_000)]])
    total = sum(
        float(row[1].replace("$", "").replace(",", ""))
        for row in items
    )
    reserve = round(total * 1.05, 2)

    story = title_block(
        carrier,
        "CLAIM INSPECTION REPORT",
    )
    story += [
        section("Claim and Adjuster Information"),
        kv_table([
            ("Claim Number",       claim_no),
            ("Policy Number",      insured["policy_no"]),
            ("Named Insured",      insured["name"]),
            ("Loss Location",      insured["address"]),
            ("Date of Loss",       date_loss),
            ("Date of Inspection", insp_date),
            ("Peril",              peril),
            ("Assigned Adjuster",  f"{adjuster['name']}  ·  License {adjuster['lic']}"),
            ("Adjuster Phone",     adjuster["phone"]),
        ], col0=2.5 * inch),
        section("Summary"),
        Paragraph(narrative.get(peril, "Loss inspected; see findings below."), BODY),
        Spacer(1, 0.08 * inch),
        section("Site Inspection Findings"),
        Paragraph(
            "A physical inspection of the loss site was conducted on the date noted "
            "above. Photographs and measurements were taken and are retained in the "
            "claim file. All findings are consistent with the reported cause of loss.",
            BODY,
        ),
        Spacer(1, 0.08 * inch),
        section("Cause of Loss Determination"),
        Paragraph(
            f"Cause of loss determined to be: <b>{peril}</b>. The loss is covered "
            "under the terms of the policy subject to applicable deductibles and any "
            "sublimits noted in the declarations or attached endorsements.",
            BODY,
        ),
        Spacer(1, 0.08 * inch),
        section("Damage Assessment and Estimate"),
        grid_table(
            ["Line Item", "Estimated Cost"],
            items,
            col_widths=[USABLE_W * 0.72, USABLE_W * 0.28],
        ),
        Spacer(1, 0.08 * inch),
        grid_table(
            ["", "Amount"],
            [
                ["TOTAL ESTIMATED REPLACEMENT COST VALUE",  money(total)],
                ["Less: Policy Deductible",                  "(to be applied at payment)"],
                ["Applicable Depreciation (if ACV basis)",  "(to be calculated)"],
                ["RECOMMENDED INITIAL RESERVE",             money(reserve)],
            ],
            col_widths=[USABLE_W * 0.72, USABLE_W * 0.28],
        ),
        Spacer(1, 0.10 * inch),
        section("Recommended Reserve"),
        Paragraph(
            f"Based on the field inspection and line-item estimate, an initial reserve "
            f"of {money(reserve)} is recommended (replacement cost value plus 5% "
            "contingency). The reserve may be revised pending contractor bids, "
            "supplemental documentation, or identification of additional covered "
            "damages.",
            BODY,
        ),
        Spacer(1, 0.10 * inch),
        Paragraph(
            f"Report prepared by: {adjuster['name']}, Licensed Property Adjuster "
            f"({adjuster['lic']}). This report is for internal claims use only and "
            "does not constitute an admission of liability or a final coverage "
            "determination.",
            SMALL,
        ),
    ]
    return story


# --- manifest and dispatch ------------------------------------------------

DOC_BUILDERS = {
    "policy_document":        build_policy_document,
    "endorsement":            build_endorsement,
    "underwriting_guideline": build_underwriting_guideline,
    "fnol_claim_form":        build_fnol_claim_form,
    "adjuster_report":        build_adjuster_report,
}


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def build_manifest(seed: int = 7):
    """Return an ordered list of (filename, doc_type, domain) — 3 per type."""
    rng = random.Random(seed)
    rows = []
    for doc_type in DOC_TYPE_TO_DOMAIN:
        for n in range(1, 4):
            tag = slug(
                rng.choice(CARRIERS if "policy" in doc_type or "endorse" in doc_type
                           else PERILS)
            )[:18]
            fname = f"{doc_type}_{n}_{tag}.pdf"
            rows.append((fname, doc_type, DOC_TYPE_TO_DOMAIN[doc_type]))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    rng = random.Random(7)
    for fname, doc_type, _domain in manifest:
        flowables = DOC_BUILDERS[doc_type](rng)
        label    = doc_type.replace("_", " ").upper()
        ref      = fname.replace(".pdf", "").upper()[:36]
        decorate = make_decorator(label, ref)
        doc = SimpleDocTemplate(
            str(OUT_DIR / fname),
            pagesize=LETTER,
            topMargin=0.8 * inch,
            bottomMargin=0.85 * inch,
            leftMargin=0.9 * inch,
            rightMargin=0.9 * inch,
            title=fname.replace(".pdf", "").replace("_", " ").title(),
            author=CARRIERS[0],
        )
        doc.build(flowables, onFirstPage=decorate, onLaterPages=decorate)
        print(f"  wrote {(OUT_DIR / fname).relative_to(OUT_DIR.parent.parent)}")

    with open(OUT_DIR / "ground_truth_doc_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "doc_type", "domain"])
        w.writerows(manifest)

    print(f"\nWrote {len(manifest)} PDFs + ground_truth_doc_types.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
