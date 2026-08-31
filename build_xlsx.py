#!/usr/bin/env python3
"""Render out/parameters.csv as a formatted, filterable workbook."""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

df = pd.read_csv("out/parameters.csv")

HDR = "FF1F3A4D"
TIER_FILL = {"Measured": "FFE3F0EB", "Weak inference": "FFF5ECE0",
             "Do not ship": "FFF6E6E3"}
TIER_FONT = {"Measured": "FF146B51", "Weak inference": "FF8C5A06",
             "Do not ship": "FF9D3628"}
STATUS_FONT = {"Implemented": "FF146B51", "Phase 1": "FF8C5A06",
               "Phase 2": "FF8C5A06", "Excluded": "FF9D3628"}

wb = Workbook()

# ------------------------------------------------------------- Read me
ws = wb.active
ws.title = "Read me"
notes = [
    ("Interview Signal Layer — Parameter Dictionary", 16, True),
    ("Generated 31 Aug 2026 from the pipeline source. Rev 1.0", 10, False),
    ("", 10, False),
    ("HOW TO READ THIS", 12, True),
    ("Every row is one number the pipeline emits. Two columns decide whether "
     "you may act on it:", 10, False),
    ("", 10, False),
    ("validity_tier — what the evidence supports", 11, True),
    ("  Measured        A real physical quantity. Trust it within its stated "
     "error and confounds.", 10, False),
    ("  Weak inference  Computable, but the mapping to any psychological state "
     "is contested. Display with error bars; never feed a decision.", 10, False),
    ("  Do not ship     Listed so nobody re-adds it later. The measurement "
     "cannot be made validly in this setting.", 10, False),
    ("", 10, False),
    ("status — where it is in the build", 11, True),
    ("  Implemented  Emitted today by the prototype in interview-signals.zip", 10, False),
    ("  Phase 1/2    Specified, not yet built. See the build sequence in the spec.", 10, False),
    ("  Excluded     Deliberately not built.", 10, False),
    ("", 10, False),
    ("principal_confound is the column to read before anyone quotes a number "
     "in a meeting. Most of these features are as sensitive to camera "
     "placement, chair design and network quality as they are to the person.", 10, False),
    ("", 10, False),
    ("NOT IN THIS LIST, ON PURPOSE", 12, True),
    ("Thought content, honesty/deception, personality traits, and any "
     "hireability score. None can be validly derived from these signals.", 10, False),
]
for i, (text, size, bold) in enumerate(notes, start=2):
    c = ws.cell(row=i, column=2, value=text)
    c.font = Font(name="Arial", size=size, bold=bold,
                  color=HDR if bold and size > 11 else "FF1F1F1F")
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 2
ws.column_dimensions["B"].width = 105
for r in (6, 19, 23):
    ws.row_dimensions[r].height = 32

# ---------------------------------------------------------- Parameters
ws = wb.create_sheet("Parameters")
cols = ["parameter_id", "group", "subgroup", "parameter", "type", "unit_range",
        "rate_hz", "validity_tier", "status", "what_it_measures",
        "principal_confound"]
titles = ["Parameter ID", "Group", "Sub-group", "Parameter", "Type",
          "Unit / range", "Rate (Hz)", "Validity tier", "Status",
          "What it measures", "Principal confound"]
widths = [24, 24, 26, 34, 10, 26, 9, 15, 13, 62, 62]

for j, (t, w) in enumerate(zip(titles, widths), start=1):
    c = ws.cell(row=1, column=j, value=t)
    c.font = Font(name="Arial", size=10, bold=True, color="FFFFFFFF")
    c.fill = PatternFill("solid", fgColor=HDR)
    c.alignment = Alignment(vertical="center", wrap_text=True)
    ws.column_dimensions[get_column_letter(j)].width = w
ws.row_dimensions[1].height = 30

thin = Side(style="thin", color="FFD9DEE3")
for i, rec in enumerate(df[cols].itertuples(index=False), start=2):
    for j, val in enumerate(rec, start=1):
        c = ws.cell(row=i, column=j, value=val)
        c.font = Font(name="Arial", size=9.5)
        c.alignment = Alignment(vertical="top", wrap_text=(j >= 10))
        c.border = Border(bottom=thin)
        if j == 1:
            c.font = Font(name="Consolas", size=9.5)
        if j == 7:
            c.alignment = Alignment(horizontal="center", vertical="top")
        if j == 8:
            c.fill = PatternFill("solid", fgColor=TIER_FILL[val])
            c.font = Font(name="Arial", size=9.5, bold=True,
                          color=TIER_FONT[val])
        if j == 9:
            c.font = Font(name="Arial", size=9.5,
                          color=STATUS_FONT.get(val, "FF1F1F1F"))

ws.freeze_panes = "D2"
ws.auto_filter.ref = f"A1:K{len(df) + 1}"

# ------------------------------------------------------------- Summary
ws = wb.create_sheet("Summary", 1)
n = len(df) + 1
P = "Parameters"


def block(row, title, header, values, col_letter):
    c = ws.cell(row=row, column=2, value=title)
    c.font = Font(name="Arial", size=12, bold=True, color=HDR)
    for j, h in enumerate(["", header, "Count", "Share"][1:], start=2):
        hc = ws.cell(row=row + 1, column=j, value=h)
        hc.font = Font(name="Arial", size=10, bold=True, color="FFFFFFFF")
        hc.fill = PatternFill("solid", fgColor=HDR)
    for k, v in enumerate(values):
        r = row + 2 + k
        ws.cell(row=r, column=2, value=v).font = Font(name="Arial", size=10)
        cc = ws.cell(row=r, column=3,
                     value=f'=COUNTIF({P}!${col_letter}$2:${col_letter}${n},$B{r})')
        cc.font = Font(name="Arial", size=10)
        cc.alignment = Alignment(horizontal="center")
        pc = ws.cell(row=r, column=4, value=f'=IFERROR(C{r}/$C${row + 2 + len(values)},0)')
        pc.font = Font(name="Arial", size=10)
        pc.number_format = "0.0%"
        pc.alignment = Alignment(horizontal="center")
    tr = row + 2 + len(values)
    ws.cell(row=tr, column=2, value="Total").font = Font(name="Arial", size=10, bold=True)
    tc = ws.cell(row=tr, column=3, value=f"=SUM(C{row + 2}:C{tr - 1})")
    tc.font = Font(name="Arial", size=10, bold=True)
    tc.alignment = Alignment(horizontal="center")
    return tr + 2


ws.cell(row=2, column=2, value="Parameter counts").font = \
    Font(name="Arial", size=15, bold=True, color=HDR)
ws.cell(row=3, column=2,
        value="Live formulas over the Parameters sheet — they update if you "
              "add or edit rows there.").font = Font(name="Arial", size=9.5,
                                                     italic=True, color="FF5A6672")

r = 5
r = block(r, "By signal group", "Group", sorted(df["group"].unique()), "B")
r = block(r, "By validity tier", "Validity tier",
          ["Measured", "Weak inference", "Do not ship"], "H")
r = block(r, "By build status", "Status",
          ["Implemented", "Phase 1", "Phase 2", "Excluded"], "I")

ws.column_dimensions["A"].width = 2
ws.column_dimensions["B"].width = 30
ws.column_dimensions["C"].width = 11
ws.column_dimensions["D"].width = 11

wb.save("out/interview-parameters.xlsx")
print(f"wrote out/interview-parameters.xlsx — {len(df)} parameters")
