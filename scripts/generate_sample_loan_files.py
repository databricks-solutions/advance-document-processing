"""Generate synthetic residential mortgage loan-file PDFs for the
page-classify-extraction pipeline.

Each PDF is a multi-document loan-origination *packet*. Every sub-document is
form-accurate and spans as many pages as its content needs, so page-level
classification + per-class extraction get realistic, heterogeneous input:

    Fax cover sheet ................. noise
    URLA / Form 1003 (multi-page) ... loan_application
    Pay stub (full earnings/deds) ... income_verification
    W-2 wage statement (boxes) ...... income_verification
    Bank statement (txn register) ... bank_statement
    IRS Form 1040 (line items) ...... tax_return
    URAR appraisal (comp grid) ...... property_appraisal
    Closing Disclosure (itemized) ... closing_disclosure
    HIPAA/notice + signature page ... noise

A page footer ("CONFIDENTIAL — <loan no> — Page N") and a running header are
drawn on every page; NB02 strips that chrome before classification.

Run (from repo root):
    uv run --with reportlab python scripts/generate_sample_loan_files.py
Output: document-page-classify-extraction/sample_data/loan_file_<n>_<lastname>.pdf
"""

from __future__ import annotations

import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "document-page-classify-extraction"
    / "sample_data"
)

NAVY = colors.HexColor("#1b3a5b")
LIGHT = colors.HexColor("#f3f6fa")
RULE = colors.HexColor("#cfd8e3")
USABLE_W = LETTER[0] - 1.8 * inch  # 0.9" margins

# --- synthetic data pools -------------------------------------------------

BORROWERS = [
    dict(name="Maria L. Hernandez", co="Carlos J. Hernandez", ssn="XXX-XX-4821",
         employer="Cedar Valley Health System", title="Registered Nurse",
         emp_addr="2200 Medical Center Blvd, Trenton, NJ 08611",
         emp_years=6, street="148 Maplewood Drive", city="Trenton", state="NJ",
         zip="08611", bank="Garden State Federal Credit Union",
         routing="031300012", monthly_income=8450, annual=101400,
         co_annual=58800, self_employed=False),
    dict(name="David R. Okafor", co="", ssn="XXX-XX-1093",
         employer="Meridian Logistics LLC", title="Operations Manager",
         emp_addr="55 Commerce Way, Edison, NJ 08837",
         emp_years=9, street="3320 Birchwood Court", city="Edison", state="NJ",
         zip="08820", bank="Liberty National Bank", routing="021200339",
         monthly_income=9920, annual=119040, co_annual=0, self_employed=False),
    dict(name="Susan K. Whitfield", co="Thomas A. Whitfield", ssn="XXX-XX-7756",
         employer="Atlantic Coast University", title="Associate Professor",
         emp_addr="1 University Plaza, Asbury Park, NJ 07712",
         emp_years=12, street="61 Harborview Lane", city="Asbury Park",
         state="NJ", zip="07712", bank="Shoreline Savings Bank",
         routing="031201467", monthly_income=7310, annual=87720,
         co_annual=72000, self_employed=False),
    dict(name="James P. Donnelly", co="", ssn="XXX-XX-3328",
         employer="Donnelly Design Studio (Self-Employed)",
         title="Principal / Owner",
         emp_addr="905 Riverbend Parkway, New Brunswick, NJ 08901",
         emp_years=7, street="905 Riverbend Parkway", city="New Brunswick",
         state="NJ", zip="08901", bank="Liberty National Bank",
         routing="021200339", monthly_income=11200, annual=134400,
         co_annual=0, self_employed=True),
]

LOAN_PURPOSES = ["Purchase", "Refinance", "Cash-out Refinance"]
LOAN_TYPES = ["Conventional", "FHA", "VA"]
PROPERTY_TYPES = ["Single Family", "Condo", "PUD"]

TXN_DESCRIPTIONS = [
    "POS PURCHASE - GROCERY", "ACH DEPOSIT - PAYROLL", "ONLINE TRANSFER TO SAVINGS",
    "UTILITY PAYMENT - PSE&G", "POS PURCHASE - FUEL", "ATM WITHDRAWAL",
    "CHECK #1042", "AUTOPAY - AUTO LOAN", "POS PURCHASE - PHARMACY",
    "MOBILE DEPOSIT", "INTEREST PAYMENT", "CARD PURCHASE - RESTAURANT",
    "INSURANCE PREMIUM - AUTO", "POS PURCHASE - HOME IMPROVEMENT",
    "ACH DEBIT - CREDIT CARD", "DIRECT DEPOSIT - REIMBURSEMENT",
    "POS PURCHASE - WAREHOUSE CLUB", "TRANSFER FROM CHECKING",
]


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, dp: int = 3) -> str:
    return f"{x:.{dp}f}%"


# --- styles ---------------------------------------------------------------

styles = getSampleStyleSheet()
H_TITLE = ParagraphStyle("doc_title", parent=styles["Title"], fontSize=15,
                         spaceAfter=4, textColor=NAVY)
H_SUB = ParagraphStyle("doc_sub", parent=styles["Normal"], fontSize=8.5,
                       textColor=colors.grey, alignment=TA_CENTER, spaceAfter=12)
H_SEC = ParagraphStyle("section", parent=styles["Heading3"], fontSize=10,
                       spaceBefore=10, spaceAfter=2, textColor=colors.white,
                       backColor=NAVY, leftIndent=4, leading=16)
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5, leading=12)
BODY_R = ParagraphStyle("body_r", parent=BODY, alignment=TA_RIGHT)
CELL = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7.5, leading=9.5)
CELL_R = ParagraphStyle("cell_r", parent=CELL, alignment=TA_RIGHT)
SMALL = ParagraphStyle("small", parent=styles["Normal"], fontSize=7,
                       textColor=colors.grey, leading=9.5)


def section(title: str) -> Paragraph:
    return Paragraph(title, H_SEC)


def title_block(title: str, subtitle: str) -> list:
    return [Paragraph(title, H_TITLE), Paragraph(subtitle, H_SUB)]


def kv_table(rows, col0=2.3 * inch, col1=None) -> Table:
    col1 = col1 or (USABLE_W - col0)
    data = [[Paragraph(f"<b>{k}</b>", BODY), Paragraph(str(v), BODY)] for k, v in rows]
    t = Table(data, colWidths=[col0, col1])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def two_col_fields(rows) -> Table:
    """Render field/value pairs in two side-by-side columns (form-like)."""
    cells = [(Paragraph(f"<b>{k}</b>", CELL), Paragraph(str(v), CELL)) for k, v in rows]
    data = []
    for i in range(0, len(cells), 2):
        left = cells[i]
        right = cells[i + 1] if i + 1 < len(cells) else (Paragraph("", CELL), Paragraph("", CELL))
        data.append([left[0], left[1], right[0], right[1]])
    w = USABLE_W / 4
    t = Table(data, colWidths=[w * 0.95, w * 1.05, w * 0.95, w * 1.05])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def grid_table(header, rows, col_widths=None, align_right_from=1, font=7.5) -> Table:
    head = [Paragraph(f"<b>{h}</b>", ParagraphStyle("h", parent=CELL, textColor=colors.white)) for h in header]
    body = []
    for r in rows:
        row = []
        for j, c in enumerate(r):
            style = CELL_R if j >= align_right_from else CELL
            row.append(Paragraph(str(c), style))
        body.append(row)
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# --- per-document builders ------------------------------------------------

def doc_fax_cover(b, loan) -> list:
    return title_block("FACSIMILE TRANSMITTAL COVER SHEET", "CONFIDENTIAL") + [
        kv_table([
            ("TO:", "Wholesale Underwriting Department"),
            ("FROM:", f"{b['employer']} — Lending Office"),
            ("FAX:", "(609) 555-0142"),
            ("PHONE:", "(609) 555-0188"),
            ("DATE:", "03/14/2026"),
            ("RE:", f"Loan File {loan['loan_no']} — {b['name']}"),
            ("TOTAL PAGES (incl. cover):", "—"),
            ("URGENCY:", "Please review for clear-to-close"),
        ]),
        Spacer(1, 0.25 * inch),
        Paragraph("MESSAGE:", H_SEC),
        Spacer(1, 0.05 * inch),
        Paragraph("Attached please find the complete loan origination package for "
                  "the above-referenced file, including the signed application, "
                  "income and asset documentation, the appraisal report, and the "
                  "final Closing Disclosure. Please confirm receipt and advise of "
                  "any outstanding conditions.", BODY),
        Spacer(1, 0.4 * inch),
        Paragraph("CONFIDENTIALITY NOTICE: This facsimile transmission contains "
                  "privileged and confidential information intended solely for the "
                  "use of the individual or entity named above. If you are not the "
                  "intended recipient, you are hereby notified that any disclosure, "
                  "copying, distribution, or the taking of any action in reliance "
                  "on the contents of this transmission is strictly prohibited. If "
                  "you have received this transmission in error, please notify the "
                  "sender immediately and destroy all copies.", SMALL),
    ]


def doc_1003(b, loan) -> list:
    joint = bool(b["co"])
    story = title_block("Uniform Residential Loan Application",
                        "Fannie Mae Form 1003 · Freddie Mac Form 65 · Page 1 of 2")
    story += [
        section("Section 1a. Personal Information — Borrower"),
        two_col_fields([
            ("Name", b["name"]),
            ("Social Security #", b["ssn"]),
            ("Date of Birth", "04/17/1985"),
            ("Citizenship", "U.S. Citizen"),
            ("Marital Status", "Married" if joint else "Unmarried"),
            ("Dependents", "2" if joint else "0"),
            ("Home Phone", "(609) 555-0173"),
            ("Email", b["name"].split()[0].lower() + "@example.com"),
            ("Current Address", f"{b['street']}, {b['city']}, {b['state']} {b['zip']}"),
            ("Housing", "Own" if loan["purpose"] != "Purchase" else "Rent"),
        ]),
        section("Section 1b. Current Employment / Self-Employment and Income"),
        two_col_fields([
            ("Employer", b["employer"]),
            ("Position", b["title"]),
            ("Employer Address", b["emp_addr"]),
            ("Years in this line of work", str(b["emp_years"])),
            ("Self-Employed", "Yes" if b["self_employed"] else "No"),
            ("Ownership Share", "100%" if b["self_employed"] else "N/A"),
        ]),
        grid_table(
            ["Gross Monthly Income", "Amount"],
            [
                ["Base", money(round(b["annual"] / 12, 2))],
                ["Overtime", money(320 if not b["self_employed"] else 0)],
                ["Bonus", money(540 if not b["self_employed"] else 0)],
                ["Self-Employment (net)", money(round(b["annual"] / 12, 2) if b["self_employed"] else 0)],
                ["TOTAL", money(b["monthly_income"])],
            ],
            col_widths=[USABLE_W * 0.6, USABLE_W * 0.4],
        ),
    ]
    if joint:
        story += [
            section("Section 2a. Personal Information — Co-Borrower"),
            two_col_fields([
                ("Name", b["co"]),
                ("Social Security #", "XXX-XX-9920"),
                ("Citizenship", "U.S. Citizen"),
                ("Annual Income", money(b["co_annual"])),
            ]),
        ]
    story += [PageBreak()]
    story += [Paragraph("Uniform Residential Loan Application", H_TITLE),
              Paragraph("Page 2 of 2", H_SUB)]
    story += [
        section("Section 2. Financial Information — Assets and Liabilities"),
        grid_table(
            ["Asset — Account Type", "Institution", "Account #", "Value"],
            [
                ["Checking", b["bank"], "****" + b["ssn"][-4:], money(round(b["monthly_income"] * 3.2, 2))],
                ["Savings", b["bank"], "****8841", money(round(b["monthly_income"] * 6.5, 2))],
                ["Retirement (401k)", "Fidelity", "****2207", money(round(b["annual"] * 1.4, 2))],
                ["Brokerage", "Vanguard", "****5530", money(round(b["annual"] * 0.6, 2))],
            ],
            col_widths=[USABLE_W * 0.28, USABLE_W * 0.3, USABLE_W * 0.2, USABLE_W * 0.22],
        ),
        Spacer(1, 0.08 * inch),
        grid_table(
            ["Liability", "Monthly Payment", "Unpaid Balance"],
            [
                ["Auto Loan", money(465), money(18240)],
                ["Credit Card — Visa", money(120), money(3870)],
                ["Student Loan", money(285), money(21500)],
            ],
            col_widths=[USABLE_W * 0.5, USABLE_W * 0.25, USABLE_W * 0.25],
        ),
        section("Section 4. Loan and Property Information"),
        two_col_fields([
            ("Loan Amount", money(loan["loan_amount"])),
            ("Loan Purpose", loan["purpose"]),
            ("Loan Type", loan["loan_type"]),
            ("Note Rate", pct(loan["rate"])),
            ("Loan Term", f"{loan['term_months']} months"),
            ("Occupancy", "Primary Residence"),
            ("Property Address", f"{b['street']}, {b['city']}, {b['state']} {b['zip']}"),
            ("Property Type", loan["property_type"]),
            ("Estimated Value", money(loan["property_value"])),
            ("Number of Units", "1"),
        ]),
        section("Section 5. Declarations"),
        grid_table(
            ["Declaration", "Response"],
            [
                ["Will you occupy the property as your primary residence?", "YES"],
                ["Have you had an ownership interest in a property in the last 3 years?",
                 "YES" if loan["purpose"] != "Purchase" else "NO"],
                ["Are there any outstanding judgments against you?", "NO"],
                ["Have you declared bankruptcy within the past 7 years?", "NO"],
                ["Are you a party to a lawsuit?", "NO"],
            ],
            col_widths=[USABLE_W * 0.82, USABLE_W * 0.18],
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("By signing below, each Borrower certifies that the information "
                  "provided in this application is true and correct as of the date "
                  "set forth opposite the signature.  Borrower signature on file — "
                  "03/12/2026.", SMALL),
    ]
    return story


def doc_paystub(b) -> list:
    base = round(b["annual"] / 24, 2)
    ot = 0.0 if b["self_employed"] else round(base * 0.06, 2)
    bonus = 0.0 if b["self_employed"] else 250.00
    gross = round(base + ot + bonus, 2)
    fed = round(gross * 0.14, 2)
    ss = round(gross * 0.062, 2)
    med = round(gross * 0.0145, 2)
    state = round(gross * 0.05, 2)
    k401 = round(gross * 0.06, 2)
    med_ins = 142.50
    ded_total = round(fed + ss + med + state + k401 + med_ins, 2)
    net = round(gross - ded_total, 2)
    f = 5  # pay periods YTD
    return title_block("EARNINGS STATEMENT", b["employer"]) + [
        two_col_fields([
            ("Employee", b["name"]),
            ("Employee ID", "E-" + b["ssn"][-4:]),
            ("Pay Period", "02/16/2026 – 02/29/2026"),
            ("Pay Date", "03/05/2026"),
            ("Department", b["title"]),
            ("Pay Frequency", "Semi-Monthly"),
            ("Employer Address", b["emp_addr"]),
            ("Annual Salary", money(b["annual"])),
        ]),
        section("Earnings"),
        grid_table(
            ["Description", "Rate", "Hours", "Current", "Year-to-Date"],
            [
                ["Regular", money(round(base / 86.67, 2)), "86.67", money(base), money(round(base * f, 2))],
                ["Overtime", money(round(base / 86.67 * 1.5, 2)), "4.00", money(ot), money(round(ot * f, 2))],
                ["Bonus", "—", "—", money(bonus), money(round(bonus * f, 2))],
                ["Gross Pay", "", "", money(gross), money(round(gross * f, 2))],
            ],
            col_widths=[USABLE_W * 0.28, USABLE_W * 0.16, USABLE_W * 0.14, USABLE_W * 0.21, USABLE_W * 0.21],
        ),
        section("Taxes and Deductions"),
        grid_table(
            ["Description", "Current", "Year-to-Date"],
            [
                ["Federal Income Tax", money(fed), money(round(fed * f, 2))],
                ["Social Security", money(ss), money(round(ss * f, 2))],
                ["Medicare", money(med), money(round(med * f, 2))],
                ["NJ State Income Tax", money(state), money(round(state * f, 2))],
                ["401(k) Pre-Tax", money(k401), money(round(k401 * f, 2))],
                ["Medical Insurance", money(med_ins), money(round(med_ins * f, 2))],
                ["Total Deductions", money(ded_total), money(round(ded_total * f, 2))],
            ],
            col_widths=[USABLE_W * 0.5, USABLE_W * 0.25, USABLE_W * 0.25],
        ),
        Spacer(1, 0.1 * inch),
        grid_table(
            ["", "Current", "Year-to-Date"],
            [["NET PAY", money(net), money(round(net * f, 2))]],
            col_widths=[USABLE_W * 0.5, USABLE_W * 0.25, USABLE_W * 0.25],
        ),
        Spacer(1, 0.1 * inch),
        Paragraph("Leave balances:  Vacation 64.5 hrs · Sick 38.0 hrs.  This is "
                  "not a check. Direct deposit to account ending " + b["ssn"][-4:] + ".",
                  SMALL),
    ]


def doc_w2(b) -> list:
    wages = b["annual"]
    ss_wages = round(wages, 2)
    fed = round(wages * 0.14, 2)
    return title_block("Form W-2  Wage and Tax Statement", "Tax Year 2025 · Copy B — Filed with employee's federal tax return") + [
        section("Employee and Employer"),
        two_col_fields([
            ("Employee", b["name"]),
            ("Employee SSN", b["ssn"]),
            ("Employer", b["employer"]),
            ("Employer EIN", "22-376" + b["ssn"][-4:]),
            ("Employer Address", b["emp_addr"]),
            ("Control Number", "0042-" + b["ssn"][-4:]),
        ]),
        section("Federal Wage and Tax Data"),
        grid_table(
            ["Box", "Description", "Amount"],
            [
                ["1", "Wages, tips, other compensation", money(wages)],
                ["2", "Federal income tax withheld", money(fed)],
                ["3", "Social Security wages", money(ss_wages)],
                ["4", "Social Security tax withheld", money(round(ss_wages * 0.062, 2))],
                ["5", "Medicare wages and tips", money(wages)],
                ["6", "Medicare tax withheld", money(round(wages * 0.0145, 2))],
                ["12a", "Code D — 401(k) elective deferral", money(round(wages * 0.06, 2))],
            ],
            col_widths=[USABLE_W * 0.1, USABLE_W * 0.6, USABLE_W * 0.3],
            align_right_from=2,
        ),
        section("State Data"),
        grid_table(
            ["Box", "Description", "Amount"],
            [
                ["15", "State / Employer state ID", "NJ / 22-376" + b["ssn"][-4:]],
                ["16", "State wages, tips, etc.", money(wages)],
                ["17", "State income tax", money(round(wages * 0.05, 2))],
            ],
            col_widths=[USABLE_W * 0.1, USABLE_W * 0.6, USABLE_W * 0.3],
            align_right_from=2,
        ),
    ]


def doc_bank_statement(b, rng) -> list:
    beginning = round(b["monthly_income"] * 3.2, 2)
    balance = beginning
    rows = []
    total_dep = 0.0
    total_wd = 0.0
    day = 1
    for _ in range(18):
        day += rng.randint(1, 2)
        if day > 28:
            break
        desc = rng.choice(TXN_DESCRIPTIONS)
        is_deposit = "DEPOSIT" in desc or "PAYROLL" in desc or "INTEREST" in desc or "TRANSFER FROM" in desc
        if is_deposit:
            amt = round(rng.uniform(150, b["monthly_income"]), 2)
            balance = round(balance + amt, 2)
            total_dep += amt
            rows.append([f"02/{day:02d}", desc, "", money(amt), money(balance)])
        else:
            amt = round(rng.uniform(20, 900), 2)
            balance = round(balance - amt, 2)
            total_wd += amt
            rows.append([f"02/{day:02d}", desc, money(amt), "", money(balance)])
    ending = balance
    story = title_block(b["bank"], "Personal Checking — Monthly Account Statement")
    story += [
        two_col_fields([
            ("Account Holder", b["name"]),
            ("Account Number", "****-****-" + b["ssn"][-4:]),
            ("Routing Number", b["routing"]),
            ("Account Type", "Interest Checking"),
            ("Statement Period", "02/01/2026 – 02/28/2026"),
            ("Statement Date", "02/28/2026"),
        ]),
        section("Account Summary"),
        grid_table(
            ["", "Amount"],
            [
                ["Beginning Balance", money(beginning)],
                ["Total Deposits & Credits", money(round(total_dep, 2))],
                ["Total Withdrawals & Debits", money(round(total_wd, 2))],
                ["Ending Balance", money(ending)],
                ["Average Daily Balance", money(round((beginning + ending) / 2, 2))],
            ],
            col_widths=[USABLE_W * 0.7, USABLE_W * 0.3],
        ),
        section("Transaction Register"),
        grid_table(
            ["Date", "Description", "Withdrawals", "Deposits", "Balance"],
            rows,
            col_widths=[USABLE_W * 0.1, USABLE_W * 0.42, USABLE_W * 0.16, USABLE_W * 0.16, USABLE_W * 0.16],
            align_right_from=2,
        ),
        Spacer(1, 0.1 * inch),
        Paragraph("Member FDIC. For questions about this statement call "
                  "1-800-555-0100. Overdraft protection is linked to savings "
                  "account ****8841.", SMALL),
    ]
    return story


def doc_1040(b) -> list:
    wages = b["annual"]
    sch_c = round(wages * 1.0, 2) if b["self_employed"] else 0.0
    interest = 412.00
    total_income = round(wages + sch_c + interest, 2)
    adjustments = round(sch_c * 0.0765, 2) if b["self_employed"] else 0.0
    agi = round(total_income - adjustments, 2)
    std_ded = 29200.00 if b["co"] else 14600.00
    taxable = round(max(agi - std_ded, 0), 2)
    tax = round(taxable * 0.16, 2)
    withheld = round(wages * 0.14, 2)
    status = "Married Filing Jointly" if b["co"] else "Single"
    story = title_block("Form 1040  U.S. Individual Income Tax Return",
                        "Department of the Treasury — Internal Revenue Service — Tax Year 2025")
    story += [
        section("Filing Information"),
        two_col_fields([
            ("Name", b["name"] + ((" & " + b["co"]) if b["co"] else "")),
            ("SSN", b["ssn"]),
            ("Filing Status", status),
            ("Home Address", f"{b['street']}, {b['city']}, {b['state']} {b['zip']}"),
        ]),
        section("Income"),
        grid_table(
            ["Line", "Description", "Amount"],
            [
                ["1a", "Total wages (Form W-2, box 1)", money(wages)],
                ["2b", "Taxable interest", money(interest)],
                ["8", "Additional income from Schedule 1 (Sch C net)", money(sch_c)],
                ["9", "Total income", money(total_income)],
                ["10", "Adjustments to income", money(adjustments)],
                ["11", "Adjusted gross income (AGI)", money(agi)],
                ["12", "Standard deduction", money(std_ded)],
                ["15", "Taxable income", money(taxable)],
            ],
            col_widths=[USABLE_W * 0.1, USABLE_W * 0.62, USABLE_W * 0.28],
            align_right_from=2,
        ),
        section("Tax, Payments, and Refund"),
        grid_table(
            ["Line", "Description", "Amount"],
            [
                ["16", "Tax", money(tax)],
                ["22", "Total tax", money(tax)],
                ["25", "Federal income tax withheld", money(withheld)],
                ["33", "Total payments", money(withheld)],
                ["34", "Overpayment (refund)" if withheld > tax else "Amount you owe",
                 money(abs(round(withheld - tax, 2)))],
            ],
            col_widths=[USABLE_W * 0.1, USABLE_W * 0.62, USABLE_W * 0.28],
            align_right_from=2,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("Under penalties of perjury, I declare that I have examined this "
                  "return and accompanying schedules and statements, and to the best "
                  "of my knowledge they are true, correct, and complete.", SMALL),
    ]
    return story


def doc_appraisal(b, loan, rng) -> list:
    gla = loan["gla"]
    year = loan["year_built"]
    beds = loan["beds"]
    baths = loan["baths"]
    av = loan["appraised_value"]
    comps = []
    for i in range(3):
        delta = rng.randrange(-18000, 18000, 1000)
        comps.append(dict(price=av + delta, gla=gla + rng.randrange(-150, 180, 10),
                          adj=-delta + rng.randrange(-3000, 3000, 500)))
    story = title_block("Uniform Residential Appraisal Report", "Fannie Mae Form 1004 · Page 1 of 2")
    story += [
        section("Subject"),
        two_col_fields([
            ("Property Address", f"{b['street']}, {b['city']}, {b['state']} {b['zip']}"),
            ("Borrower", b["name"]),
            ("Legal Description", f"Lot 14, Block 7, {b['city']} Twp."),
            ("Assessor's Parcel #", "07-00148-0014"),
            ("Property Type", loan["property_type"]),
            ("Occupant", "Owner"),
            ("R.E. Taxes (yr)", money(round(av * 0.021, 2))),
            ("Appraised Value", money(av)),
        ]),
        section("Neighborhood and Site"),
        two_col_fields([
            ("Location", "Suburban"),
            ("Built-Up", "Over 75%"),
            ("Growth", "Stable"),
            ("Property Values", "Stable"),
            ("Site Area", "0.28 acres (12,196 sq ft)"),
            ("Zoning", "R-1 Residential"),
            ("Utilities", "Public water/sewer/electric/gas"),
            ("FEMA Flood Zone", "Zone X (minimal)"),
        ]),
        section("Improvements"),
        two_col_fields([
            ("Design (Style)", "Colonial"),
            ("Year Built", str(year)),
            ("Effective Age", "12 yrs"),
            ("Condition", "C3 — well maintained"),
            ("Above-Grade Rooms", str(beds + 3)),
            ("Bedrooms", str(beds)),
            ("Bathrooms", f"{baths:.1f}"),
            ("Gross Living Area", f"{gla:,} sq ft"),
            ("Basement", "Full, 60% finished"),
            ("Garage", "2-car attached"),
        ]),
        PageBreak(),
        Paragraph("Uniform Residential Appraisal Report", H_TITLE),
        Paragraph("Sales Comparison Approach · Page 2 of 2", H_SUB),
        section("Sales Comparison Approach"),
        grid_table(
            ["Feature", "Subject", "Comparable 1", "Comparable 2", "Comparable 3"],
            [
                ["Address", f"{b['city']}", "22 Oak St", "418 Pine Ave", "9 Cedar Ct"],
                ["Sale Price", "—", money(comps[0]["price"]), money(comps[1]["price"]), money(comps[2]["price"])],
                ["GLA (sq ft)", f"{gla:,}", f"{comps[0]['gla']:,}", f"{comps[1]['gla']:,}", f"{comps[2]['gla']:,}"],
                ["Site", "0.28 ac", "0.26 ac", "0.31 ac", "0.27 ac"],
                ["Design", "Colonial", "Colonial", "Ranch", "Colonial"],
                ["Age", f"{2026-year} yrs", "14 yrs", "20 yrs", "11 yrs"],
                ["Condition", "C3", "C3", "C4", "C3"],
                ["Rooms / Bd / Ba", f"{beds+3}/{beds}/{baths:.1f}", "8/4/2.5", "7/3/2.0", "8/4/3.0"],
                ["Garage", "2-car", "2-car", "1-car", "2-car"],
                ["Net Adjustment", "", money(comps[0]["adj"]), money(comps[1]["adj"]), money(comps[2]["adj"])],
                ["Adjusted Price", "", money(comps[0]["price"] + comps[0]["adj"]),
                 money(comps[1]["price"] + comps[1]["adj"]), money(comps[2]["price"] + comps[2]["adj"])],
            ],
            col_widths=[USABLE_W * 0.2, USABLE_W * 0.16, USABLE_W * 0.21, USABLE_W * 0.21, USABLE_W * 0.22],
            font=7,
        ),
        section("Reconciliation"),
        kv_table([
            ("Indicated Value by Sales Comparison Approach", money(av)),
            ("Indicated Value by Cost Approach", money(av + 6000)),
            ("Final Reconciled Market Value", money(av)),
            ("Effective Date of Appraisal", "03/10/2026"),
            ("Appraiser", "R. Coleman"),
            ("License #", "NJ-42-008917 (expires 12/2027)"),
        ]),
    ]
    return story


def doc_closing_disclosure(b, loan) -> list:
    r = loan["rate"] / 1200
    n = loan["term_months"]
    pi = round(loan["loan_amount"] * r / (1 - (1 + r) ** (-n)), 2)
    taxes_ins = round(loan["appraised_value"] * 0.021 / 12 + 118, 2)
    total_pmt = round(pi + taxes_ins, 2)
    origination = round(loan["loan_amount"] * 0.01, 2)
    services = round(loan["loan_amount"] * 0.008, 2)
    total_loan_costs = round(origination + services, 2)
    prepaids = round(loan["loan_amount"] * 0.006, 2)
    escrow = round(taxes_ins * 3, 2)
    other_costs = round(prepaids + escrow + 1850, 2)
    total_closing = round(total_loan_costs + other_costs, 2)
    down = round(loan["property_value"] - loan["loan_amount"], 2)
    cash_to_close = round(down + total_closing - 1500, 2)
    story = title_block("Closing Disclosure",
                        "This form is a statement of final loan terms and closing costs. Page 1 of 2")
    story += [
        section("Loan Information"),
        two_col_fields([
            ("Date Issued", "03/13/2026"),
            ("Closing Date", "03/28/2026"),
            ("Loan ID #", loan["loan_no"]),
            ("Loan Type", loan["loan_type"]),
            ("Loan Purpose", loan["purpose"]),
            ("Product", "Fixed Rate"),
            ("Borrower", b["name"]),
            ("Property", f"{b['street']}, {b['city']}, {b['state']} {b['zip']}"),
        ]),
        section("Loan Terms"),
        grid_table(
            ["", "Amount", "Can this amount increase after closing?"],
            [
                ["Loan Amount", money(loan["loan_amount"]), "NO"],
                ["Interest Rate", pct(loan["rate"]), "NO"],
                ["Monthly Principal & Interest", money(pi), "NO"],
                ["Prepayment Penalty", "None", "—"],
                ["Balloon Payment", "None", "—"],
            ],
            col_widths=[USABLE_W * 0.32, USABLE_W * 0.23, USABLE_W * 0.45],
        ),
        section("Projected Payments"),
        grid_table(
            ["", "Amount"],
            [
                ["Principal & Interest", money(pi)],
                ["Estimated Escrow (taxes + insurance)", money(taxes_ins)],
                ["Estimated Total Monthly Payment", money(total_pmt)],
                ["Annual Percentage Rate (APR)", pct(loan["rate"] + 0.142)],
            ],
            col_widths=[USABLE_W * 0.6, USABLE_W * 0.4],
        ),
        PageBreak(),
        Paragraph("Closing Disclosure", H_TITLE),
        Paragraph("Closing Cost Details · Page 2 of 2", H_SUB),
        section("Loan Costs"),
        grid_table(
            ["", "Borrower-Paid"],
            [
                ["A. Origination Charges (1.00% of loan amount)", money(origination)],
                ["B. Services Borrower Did Not Shop For", money(round(services * 0.4, 2))],
                ["C. Services Borrower Did Shop For", money(round(services * 0.6, 2))],
                ["D. TOTAL LOAN COSTS", money(total_loan_costs)],
            ],
            col_widths=[USABLE_W * 0.7, USABLE_W * 0.3],
        ),
        section("Other Costs"),
        grid_table(
            ["", "Borrower-Paid"],
            [
                ["E. Taxes and Other Government Fees (recording)", money(1850)],
                ["F. Prepaids (homeowner's ins., interest, taxes)", money(prepaids)],
                ["G. Initial Escrow Payment at Closing", money(escrow)],
                ["H. Other (owner's title policy)", money(0)],
                ["I. TOTAL OTHER COSTS", money(other_costs)],
                ["J. TOTAL CLOSING COSTS", money(total_closing)],
            ],
            col_widths=[USABLE_W * 0.7, USABLE_W * 0.3],
        ),
        section("Calculating Cash to Close"),
        grid_table(
            ["", "Amount"],
            [
                ["Down Payment / Funds from Borrower", money(down)],
                ["Total Closing Costs (J)", money(total_closing)],
                ["Deposit / Earnest Money", money(-1500)],
                ["Cash to Close", money(cash_to_close)],
            ],
            col_widths=[USABLE_W * 0.6, USABLE_W * 0.4],
        ),
    ]
    return story


def doc_notice_signature() -> list:
    return title_block("BORROWER ACKNOWLEDGMENTS & PRIVACY NOTICE", "") + [
        Paragraph("PRIVACY NOTICE (Gramm-Leach-Bliley Act): We collect nonpublic "
                  "personal information about you from the following sources: "
                  "information we receive from you on applications or other forms; "
                  "information about your transactions with us, our affiliates, or "
                  "others; and information we receive from a consumer reporting "
                  "agency. We do not disclose any nonpublic personal information "
                  "about our customers or former customers except as permitted by "
                  "law.", SMALL),
        Spacer(1, 0.2 * inch),
        Paragraph("ESIGN CONSENT: By signing below you consent to receive "
                  "disclosures electronically. This page contains no loan-specific "
                  "financial data and is retained for compliance purposes only.", SMALL),
        Spacer(1, 1.6 * inch),
        Paragraph("X ____________________________________      Date: ____________", BODY),
        Paragraph("Borrower Signature", SMALL),
        Spacer(1, 0.5 * inch),
        Paragraph("X ____________________________________      Date: ____________", BODY),
        Paragraph("Co-Borrower Signature", SMALL),
        Spacer(1, 0.6 * inch),
        Paragraph("[ This page intentionally left blank below this line — document separator ]",
                  SMALL),
    ]


# --- packet assembly ------------------------------------------------------

def build_loan(b, idx) -> dict:
    rng = random.Random(idx * 7 + 13)
    property_value = rng.randrange(320_000, 720_000, 5_000)
    loan_amount = round(property_value * rng.choice([0.80, 0.85, 0.90]), 2)
    return dict(
        loan_no=f"LN-2026-{1000 + idx}",
        purpose=rng.choice(LOAN_PURPOSES),
        loan_type=rng.choice(LOAN_TYPES),
        property_type=rng.choice(PROPERTY_TYPES),
        rate=round(rng.uniform(5.75, 7.25), 3),
        term_months=rng.choice([180, 360]),
        property_value=property_value,
        appraised_value=property_value + rng.randrange(-15_000, 20_000, 1_000),
        loan_amount=loan_amount,
        gla=rng.choice([1680, 1920, 2150, 2480, 2760]),
        year_built=rng.choice([1978, 1992, 2001, 2014]),
        beds=rng.choice([3, 4, 4, 5]),
        baths=rng.choice([2.0, 2.5, 3.0]),
    )


def make_decorator(loan, bank_name):
    def _decorate(canvas, doc):
        canvas.saveState()
        # header rule + running header
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(0.9 * inch, LETTER[1] - 0.6 * inch,
                    LETTER[0] - 0.9 * inch, LETTER[1] - 0.6 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.9 * inch, LETTER[1] - 0.52 * inch,
                          f"MORTGAGE LOAN FILE · {loan['loan_no']}")
        canvas.drawRightString(LETTER[0] - 0.9 * inch, LETTER[1] - 0.52 * inch,
                               bank_name)
        # footer
        canvas.line(0.9 * inch, 0.62 * inch, LETTER[0] - 0.9 * inch, 0.62 * inch)
        canvas.drawString(0.9 * inch, 0.5 * inch,
                          f"CONFIDENTIAL — {loan['loan_no']}")
        canvas.drawRightString(LETTER[0] - 0.9 * inch, 0.5 * inch,
                               f"Page {doc.page}")
        canvas.restoreState()
    return _decorate


def build_pdf(b, idx) -> Path:
    rng = random.Random(idx * 101 + 7)
    loan = build_loan(b, idx)
    last = b["name"].split()[-1].lower()
    out = OUT_DIR / f"loan_file_{idx + 1}_{last}.pdf"

    docs = [
        doc_fax_cover(b, loan),
        doc_1003(b, loan),
        doc_paystub(b),
        doc_w2(b),
        doc_bank_statement(b, rng),
        doc_1040(b),
        doc_appraisal(b, loan, rng),
        doc_closing_disclosure(b, loan),
        doc_notice_signature(),
    ]
    story: list = []
    for i, d in enumerate(docs):
        story.extend(d)
        if i < len(docs) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(out), pagesize=LETTER,
        topMargin=0.8 * inch, bottomMargin=0.85 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        title=f"Mortgage Loan File {loan['loan_no']}",
        author=b["bank"],
    )
    decorate = make_decorator(loan, b["bank"])
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, b in enumerate(BORROWERS):
        out = build_pdf(b, idx)
        print(f"  wrote {out.relative_to(OUT_DIR.parent.parent)}")
    print(f"\n{len(BORROWERS)} loan-file PDFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
