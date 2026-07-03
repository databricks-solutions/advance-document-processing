"""Generate synthetic Oracle-Payslip-style paystub PDFs for the
ai-extract-word-level-citation pipeline, plus the matching ground-truth CSV for
the evaluation-harness recipes.

The layout mirrors the real paystubs in
``/Users/q.yu/workspace/developments/ai-extract-word-level-citation/pdf_files``
(a two-column identity block, Pay Period & Salary, Summary Current/YTD, Hours &
Earnings, Pre-Tax Deductions / Taxes, After-Tax Deductions / Accruals, Tax
Withholding, Net Pay Distribution) but every value is **fake** — invented names,
employers, masked identifiers, and amounts. No real PII.

This script is the single source of truth: the same ``RECORDS`` drive both the
rendered PDFs and the ground-truth CSV, so GT values match the documents exactly.
One record (Priya Nair) intentionally omits the national identifier so the
evaluation-harness recipe 03 has an "absent field" (null ground truth) to score.

Run (from repo root):
    uv run --with reportlab python scripts/generate_sample_paystubs.py

Outputs:
    ai-extract-word-level-citation/sample_data/paystub_synth_<n>.pdf
    evaluation-harness/sample_data/paystub_ground_truth.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# --- palette / styles -------------------------------------------------------

LABEL_BG = colors.HexColor("#d6e4f0")   # shaded label cells (blue-grey)
SECTION_BG = colors.HexColor("#e8e8e8")  # section header band
GRID = colors.HexColor("#b0b0b0")

_LABEL = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=6.5, textColor=colors.HexColor("#33475b"))
_VALUE = ParagraphStyle("value", fontName="Helvetica", fontSize=6.5, leading=8)
_SECTION = ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=7.5)
_BRAND = ParagraphStyle("brand", fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#c8102e"))
_TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=12, alignment=1)


# --- the four synthetic employees (all data fake) ---------------------------

RECORDS = [
    {
        "file": "paystub_synth_1.pdf",
        "employee_name": "Jordan Rivera",
        "job_title": "Transit Operator",
        "national_id": "XXX-XX-4821",
        "employee_number": "100488",
        "latest_hire_date": "14-Mar-2021",
        "original_hire_date": "14-Mar-2021",
        "adjusted_service_date": "14-Mar-2021",
        "assignment_number": "104488-2",
        "location": "Garage 7",
        "position": "5210 BUS OPERATIONS, DEPOT 7",
        "payroll": "Ops Weekly",
        "emp_addr": ["482 Maple Ridge Rd", "Franklin, Ohio 45005", "US"],
        "employer_name": "Metro City Transit Authority",
        "employer_phone": "2220 Metro City Transit Authority",
        "organization": "2220 Metro City Transit Authority",
        "pay_basis": "Hourly",
        "frequency": "Week",
        "shift": "Day",
        "bargaining_unit": "MCT Operators Union",
        "grade": "72.J",
        "employer_site": "3120 Depot Avenue",
        "employer_addr": ["3120 Depot Avenue", "Franklin, Ohio 45005", "US"],
        "pay_period_type": "Bi-Week",
        "payment_date": "07-Nov-2025",
        "pay_begin_date": "20-Oct-2025",
        "pay_end_date": "02-Nov-2025",
        "pay_rate": "31.40",
        "annual_salary": "65,312.00",
        "marital_status": "Married",
        "exemptions": "2",
        "gross_current": "1,884.00", "pretax_current": "142.10", "taxes_current": "318.55",
        "deductions_current": "36.00", "net_current": "1,387.35",
        "gross_ytd": "43,332.00", "pretax_ytd": "3,268.30", "taxes_ytd": "7,326.65",
        "deductions_ytd": "828.00", "net_ytd": "31,909.05",
        "earnings": [
            ("Operator Labor", "31.40", "80.00", "1,884.00", "1840.00", "43,332.00"),
            ("Overtime Operator", "47.10", "0.00", "0.00", "60.00", "2,826.00"),
            ("Holiday Pay", "31.40", "0.00", "0.00", "40.00", "1,256.00"),
        ],
        "pretax_ded": [("MCT Retirement PPG", "142.10", "3,268.30"), ("Delta Dental", "0.00", "31.62")],
        "taxes": [("Federal Tax", "196.40", "4,517.20"), ("Medicare", "27.32", "628.31"),
                  ("OH State Tax", "62.18", "1,430.14"), ("Franklin City", "32.65", "751.00")],
        "aftertax_ded": [("Union Dues MCT", "36.00", "828.00")],
        "accruals": [("Vacation", "6.15", "88.40"), ("Sick", "3.08", "22.16")],
        "bank_name": "Franklin Community CU", "account_type": "Checking",
        "account_number": "XXXXXX2043", "net_deposit": "1,387.35",
    },
    {
        "file": "paystub_synth_2.pdf",
        "employee_name": "Alicia Chen",
        "job_title": "Warehouse Associate",
        "national_id": "XXX-XX-7156",
        "employee_number": "205519",
        "latest_hire_date": "02-Aug-2022",
        "original_hire_date": "02-Aug-2022",
        "adjusted_service_date": "02-Aug-2022",
        "assignment_number": "205519-1",
        "location": "DC-3 Receiving",
        "position": "3300 WAREHOUSE, DC-3",
        "payroll": "Hourly Bi-Weekly",
        "emp_addr": ["77 Birchwood Ln Apt 4B", "Aurora, Illinois 60504", "US"],
        "employer_name": "Northgate Logistics LLC",
        "employer_phone": "4410 Northgate Logistics LLC",
        "organization": "4410 Northgate Logistics LLC",
        "pay_basis": "Hourly",
        "frequency": "Bi-Week",
        "shift": "Evening",
        "bargaining_unit": "Non-Union",
        "grade": "WH-3",
        "employer_site": "900 Commerce Pkwy",
        "employer_addr": ["900 Commerce Pkwy", "Aurora, Illinois 60504", "US"],
        "pay_period_type": "Bi-Week",
        "payment_date": "31-Oct-2025",
        "pay_begin_date": "13-Oct-2025",
        "pay_end_date": "26-Oct-2025",
        "pay_rate": "22.75",
        "annual_salary": "47,320.00",
        "marital_status": "Single",
        "exemptions": "1",
        "gross_current": "1,910.00", "pretax_current": "95.50", "taxes_current": "301.20",
        "deductions_current": "18.00", "net_current": "1,495.30",
        "gross_ytd": "40,110.00", "pretax_ytd": "2,005.50", "taxes_ytd": "6,325.20",
        "deductions_ytd": "378.00", "net_ytd": "31,401.30",
        "earnings": [
            ("Regular Hours", "22.75", "80.00", "1,820.00", "1680.00", "38,220.00"),
            ("Overtime", "34.13", "2.64", "90.00", "42.00", "1,890.00"),
        ],
        "pretax_ded": [("HSA Contribution", "95.50", "2,005.50")],
        "taxes": [("Federal Tax", "188.40", "3,956.40"), ("Medicare", "27.70", "581.60"),
                  ("IL State Tax", "85.10", "1,787.20")],
        "aftertax_ded": [("Parking", "18.00", "378.00")],
        "accruals": [("Vacation", "4.62", "54.80"), ("Sick", "1.85", "18.40")],
        "bank_name": "Prairie State Bank", "account_type": "Checking",
        "account_number": "XXXXXX8890", "net_deposit": "1,495.30",
    },
    {
        "file": "paystub_synth_3.pdf",
        "employee_name": "Marcus Bell",
        "job_title": "Line Cook",
        "national_id": "XXX-XX-3390",
        "employee_number": "318702",
        "latest_hire_date": "19-Jun-2023",
        "original_hire_date": "19-Jun-2023",
        "adjusted_service_date": "19-Jun-2023",
        "assignment_number": "318702-1",
        "location": "Unit 12 Kitchen",
        "position": "6100 CULINARY, UNIT 12",
        "payroll": "Weekly Hourly",
        "emp_addr": ["1330 Harbor View St", "Tacoma, Washington 98402", "US"],
        "employer_name": "Harbor Foods Group Inc",
        "employer_phone": "5150 Harbor Foods Group Inc",
        "organization": "5150 Harbor Foods Group Inc",
        "pay_basis": "Hourly",
        "frequency": "Week",
        "shift": "Split",
        "bargaining_unit": "UNITE HERE Local",
        "grade": "K-2",
        "employer_site": "50 Dockside Blvd",
        "employer_addr": ["50 Dockside Blvd", "Tacoma, Washington 98402", "US"],
        "pay_period_type": "Week",
        "payment_date": "24-Oct-2025",
        "pay_begin_date": "13-Oct-2025",
        "pay_end_date": "19-Oct-2025",
        "pay_rate": "19.50",
        "annual_salary": "40,560.00",
        "marital_status": "Head of Household",
        "exemptions": "2",
        "gross_current": "877.50", "pretax_current": "43.88", "taxes_current": "121.40",
        "deductions_current": "12.50", "net_current": "699.72",
        "gross_ytd": "35,977.50", "pretax_ytd": "1,798.88", "taxes_ytd": "4,977.40",
        "deductions_ytd": "512.50", "net_ytd": "28,688.72",
        "earnings": [
            ("Regular Hours", "19.50", "40.00", "780.00", "1640.00", "31,980.00"),
            ("Overtime", "29.25", "3.33", "97.50", "58.00", "2,842.50"),
            ("Holiday Pay", "19.50", "0.00", "0.00", "56.00", "1,092.00"),
        ],
        "pretax_ded": [("Vision Plan", "43.88", "1,798.88")],
        "taxes": [("Federal Tax", "76.20", "3,123.40"), ("Medicare", "12.72", "521.67"),
                  ("WA L&I", "32.48", "1,332.33")],
        "aftertax_ded": [("Meal Program", "12.50", "512.50")],
        "accruals": [("Sick", "1.35", "12.10")],
        "bank_name": "Sound Credit Union", "account_type": "Savings",
        "account_number": "XXXXXX1177", "net_deposit": "699.72",
    },
    {
        "file": "paystub_synth_4.pdf",
        "employee_name": "Priya Nair",
        "job_title": "Administrative Assistant",
        "national_id": "",   # intentionally absent -> ground truth is null
        "employee_number": "409145",
        "latest_hire_date": "05-Feb-2020",
        "original_hire_date": "05-Feb-2020",
        "adjusted_service_date": "05-Feb-2020",
        "assignment_number": "409145-3",
        "location": "HQ Suite 200",
        "position": "1200 ADMIN SERVICES, HQ",
        "payroll": "Salary Bi-Weekly",
        "emp_addr": ["908 Cedar Valley Dr", "Durham, North Carolina 27701", "US"],
        "employer_name": "Cedar Valley Health System",
        "employer_phone": "6600 Cedar Valley Health System",
        "organization": "6600 Cedar Valley Health System",
        "pay_basis": "Salaried",
        "frequency": "Bi-Week",
        "shift": "Day",
        "bargaining_unit": "Non-Union",
        "grade": "A-4",
        "employer_site": "1 Health Plaza",
        "employer_addr": ["1 Health Plaza", "Durham, North Carolina 27701", "US"],
        "pay_period_type": "Bi-Week",
        "payment_date": "07-Nov-2025",
        "pay_begin_date": "20-Oct-2025",
        "pay_end_date": "02-Nov-2025",
        "pay_rate": "26.44",
        "annual_salary": "55,000.00",
        "marital_status": "Married",
        "exemptions": "3",
        "gross_current": "2,115.38", "pretax_current": "158.65", "taxes_current": "352.10",
        "deductions_current": "24.00", "net_current": "1,580.63",
        "gross_ytd": "48,653.74", "pretax_ytd": "3,648.95", "taxes_ytd": "8,098.30",
        "deductions_ytd": "552.00", "net_ytd": "36,354.49",
        "earnings": [
            ("Salary", "26.44", "80.00", "2,115.38", "1840.00", "48,653.74"),
        ],
        "pretax_ded": [("401(k) Pretax", "126.92", "2,919.16"), ("Medical Plan", "31.73", "729.79")],
        "taxes": [("Federal Tax", "232.69", "5,351.87"), ("Medicare", "30.67", "705.48"),
                  ("NC State Tax", "88.74", "2,040.95")],
        "aftertax_ded": [("Life Insurance", "24.00", "552.00")],
        "accruals": [("Vacation", "6.15", "102.30"), ("Sick", "3.08", "44.20")],
        "bank_name": "Triangle Federal CU", "account_type": "Checking",
        "account_number": "XXXXXX5521", "net_deposit": "1,580.63",
    },
]

# Which record fields become ground-truth columns (map to the pipeline's
# EXTRACT_FIELDS names). doc_id is the join key used by evaluation-harness recipe 03.
GT_FIELD_MAP = [
    ("employee_name", "employee_name"),
    ("employer_name", "employer_name"),
    ("social_security_number", "national_id"),
    ("taxable_marital_status", "marital_status"),
    ("hire_date", "latest_hire_date"),
    ("pay_date", "payment_date"),
    ("period_start_date", "pay_begin_date"),
    ("period_end_date", "pay_end_date"),
    ("payment_frequency", "frequency"),
    ("gross__current_amount", "gross_current"),
    ("gross__year_to_date_amount", "gross_ytd"),
    ("net_pay__current_amount", "net_current"),
    ("net_pay__year_to_date_amount", "net_ytd"),
]


# --- rendering helpers ------------------------------------------------------

def _p(text, style):
    return Paragraph(str(text) if text is not None else "", style)


def _section(title, width):
    t = Table([[_p(title, _SECTION)]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SECTION_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _identity_block(rec, width):
    """Two side-by-side label:value grids (employee | employer)."""
    left = [
        ("Employee Full Name", rec["employee_name"]),
        ("Job Title", rec["job_title"]),
        ("National Identifier", rec["national_id"]),
        ("Employee Number", rec["employee_number"]),
        ("Latest Hire Date", rec["latest_hire_date"]),
        ("Original Hire Date", rec["original_hire_date"]),
        ("Adjusted Service Date", rec["adjusted_service_date"]),
        ("Assignment Number", rec["assignment_number"]),
        ("Location", rec["location"]),
        ("Position", rec["position"]),
        ("Payroll", rec["payroll"]),
        ("Employee Address", "\n".join(rec["emp_addr"])),
    ]
    right = [
        ("Employer Name", rec["employer_name"]),
        ("Employer Phone Number", rec["employer_phone"]),
        ("Organization", rec["organization"]),
        ("Pay Basis", rec["pay_basis"]),
        ("Frequency", rec["frequency"]),
        ("Shift", rec["shift"]),
        ("Bargaining Unit", rec["bargaining_unit"]),
        ("Grade", rec["grade"]),
        ("Employer Site", rec["employer_site"]),
        ("Employer Address", "\n".join(rec["employer_addr"])),
        ("", ""),
        ("", ""),
    ]
    rows = []
    for (ll, lv), (rl, rv) in zip(left, right):
        rows.append([_p(ll, _LABEL), _p(lv.replace("\n", "<br/>"), _VALUE),
                     _p(rl, _LABEL), _p(rv.replace("\n", "<br/>"), _VALUE)])
    lw = width / 2.0
    t = Table(rows, colWidths=[lw * 0.42, lw * 0.58, lw * 0.42, lw * 0.58])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LABEL_BG),
        ("BACKGROUND", (2, 0), (2, -1), LABEL_BG),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _grid_table(header, rows, col_widths, aligns=None):
    data = [[_p(h, _LABEL) for h in header]] + [[_p(c, _VALUE) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), LABEL_BG),
        ("GRID", (0, 0), (-1, -1), 0.25, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ]
    for col in (aligns or []):
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _two_up(left_tbl, right_tbl, width):
    gap = 0.15 * inch
    half = (width - gap) / 2.0
    outer = Table([[left_tbl, "", right_tbl]], colWidths=[half, gap, half])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return outer, half


def build_pdf(rec, out_path, width):
    story = []
    header = Table(
        [[_p("ACME Payroll", _BRAND), _p("Payslip", _TITLE), _p("Page 1", _VALUE)]],
        colWidths=[width * 0.3, width * 0.4, width * 0.3],
    )
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ALIGN", (2, 0), (2, 0), "RIGHT")]))
    story += [header, Spacer(1, 6)]

    story += [_identity_block(rec, width), Spacer(1, 6)]

    story += [_section("Pay Period and Salary", width)]
    story += [_grid_table(
        ["Assignment Number", "Assignment Type", "Pay Period", "Payment Date",
         "Pay Begin Date", "Pay End Date", "Pay Rate", "Annual Salary"],
        [[rec["assignment_number"], "Primary", rec["pay_period_type"], rec["payment_date"],
          rec["pay_begin_date"], rec["pay_end_date"], rec["pay_rate"], rec["annual_salary"]]],
        [width * f for f in (0.15, 0.12, 0.10, 0.12, 0.12, 0.12, 0.10, 0.17)],
    ), Spacer(1, 6)]

    story += [_section("Summary", width)]
    story += [_grid_table(
        ["", "Gross", "Pre-Tax", "Taxes", "Deductions", "Net Pay"],
        [["Current", rec["gross_current"], rec["pretax_current"], rec["taxes_current"],
          rec["deductions_current"], rec["net_current"]],
         ["YTD", rec["gross_ytd"], rec["pretax_ytd"], rec["taxes_ytd"],
          rec["deductions_ytd"], rec["net_ytd"]]],
        [width * f for f in (0.14, 0.172, 0.172, 0.172, 0.172, 0.172)],
        aligns=[1, 2, 3, 4, 5],
    ), Spacer(1, 6)]

    story += [_section("Hours and Earnings", width)]
    story += [_grid_table(
        ["Description", "Rate", "Current Hours", "Current Amount", "YTD Hours", "YTD Amount"],
        rec["earnings"],
        [width * f for f in (0.30, 0.12, 0.16, 0.16, 0.13, 0.13)],
        aligns=[1, 2, 3, 4, 5],
    ), Spacer(1, 6)]

    half = (width - 0.15 * inch) / 2.0
    hdr, _ = _two_up(_section("Pre Tax Deductions", half), _section("Taxes", half), width)
    story += [hdr]
    pre = _grid_table(["Description", "Current", "YTD"], rec["pretax_ded"],
                      [half * 0.5, half * 0.25, half * 0.25], aligns=[1, 2])
    tax = _grid_table(["Description", "Current", "YTD"], rec["taxes"],
                      [half * 0.5, half * 0.25, half * 0.25], aligns=[1, 2])
    body, _ = _two_up(pre, tax, width)
    story += [body, Spacer(1, 6)]

    hdr2, _ = _two_up(_section("After Tax Deductions", half), _section("Accruals", half), width)
    story += [hdr2]
    aft = _grid_table(["Description", "Current", "YTD"], rec["aftertax_ded"],
                      [half * 0.5, half * 0.25, half * 0.25], aligns=[1, 2])
    acc = _grid_table(["Description", "Current", "Balance"], rec["accruals"],
                      [half * 0.5, half * 0.25, half * 0.25], aligns=[1, 2])
    body2, _ = _two_up(aft, acc, width)
    story += [body2, Spacer(1, 6)]

    story += [_section("Tax Withholding Information", width)]
    story += [_grid_table(
        ["Type", "Marital Status", "Exemptions", "Additional Amount", "Override Amount"],
        [["Federal", rec["marital_status"], rec["exemptions"], "0.00", "0.00"]],
        [width * f for f in (0.14, 0.34, 0.16, 0.18, 0.18)],
    ), Spacer(1, 6)]

    story += [_section("Net Pay Distribution", width)]
    story += [_grid_table(
        ["Bank Name", "Account Type", "Account Number", "Amount"],
        [[rec["bank_name"], rec["account_type"], rec["account_number"], rec["net_deposit"]]],
        [width * f for f in (0.34, 0.20, 0.26, 0.20)],
        aligns=[3],
    )]

    SimpleDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    ).build(story)


def main():
    repo = Path(__file__).resolve().parents[1]
    pdf_dir = repo / "ai-extract-word-level-citation" / "sample_data"
    gt_dir = repo / "evaluation-harness" / "sample_data"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    width = LETTER[0] - 1.2 * inch
    for rec in RECORDS:
        build_pdf(rec, pdf_dir / rec["file"], width)
        print(f"wrote {pdf_dir / rec['file']}")

    gt_path = gt_dir / "paystub_ground_truth.csv"
    columns = ["doc_id"] + [gt_col for gt_col, _ in GT_FIELD_MAP]
    with gt_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for rec in RECORDS:
            row = [rec["file"]] + [rec[src_key] for _, src_key in GT_FIELD_MAP]
            writer.writerow(row)
    print(f"wrote {gt_path}")


if __name__ == "__main__":
    main()
