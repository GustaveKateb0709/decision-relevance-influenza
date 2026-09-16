"""
Build the Word deliverables for Paper 16 (PLOS Computational Biology build)
from the markdown sources.

Writes into 01_Manuscript/:
  Manuscript_full.docx     full version (figures embedded after References;
                           the supplementary figure S1 Fig is NOT embedded and
                           is uploaded separately as Supporting Information)
  Manuscript_blinded.docx  blinded version (no author block / contributions)
  Title_Page.docx
  Cover_Letter.docx        PLOS Computational Biology version
  Figure_captions.docx

Formatting:
  * Normal style line spacing == 2.0 (kept so the manuscript-submission-qa
    spacing check continues to pass)
  * continuous line numbers (w:lnNumType) in sectPr
  * PAGE field in the footer part (not in document.xml)
  * main figures embedded AFTER the References section with full captions

Run:  python src/make_docx.py
"""
from __future__ import annotations

import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "manuscript")
FIGDIR = os.path.join(ROOT, "figures")
OUT = os.path.join(MD, "docx")

JOURNAL = "PLOS Computational Biology"


def paper_title() -> str:
    """Single source of truth for the title: the first '# ' line of manuscript.md."""
    with open(os.path.join(MD, "manuscript.md"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# "):
                return line[2:].strip()
    raise RuntimeError("no '# ' title line found in manuscript.md")

# main figures embedded in the manuscript (PLOS numbering). The supplementary
# figure S1 Fig (files s1_fig_drs_vs_ppf.*) lives in 03_Supplementary and is
# uploaded as separate Supporting Information — never embedded here.
FIGURES = [
    (1, "fig1_drs_positive_control.png", None),
    (2, "fig2_parameter_identifiability_vs_decision_instability.png", None),
    (3, "fig3_conditioning_vs_relevance.png", None),
    (4, "fig4_snr_peakcount.png", None),
]


# --------------------------------------------------------------------------
# docx plumbing
# --------------------------------------------------------------------------
def set_two_line_spacing(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.paragraph_format.line_spacing = 2.0
    st.paragraph_format.space_after = Pt(0)
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)


def add_line_numbers(doc: Document) -> None:
    """Continuous line numbers, inserted in schema order (before w:cols)."""
    sectPr = doc.sections[0]._sectPr
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), "360")
    cols = sectPr.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(ln)
    else:
        sectPr.append(ln)


def add_page_footer(doc: Document) -> None:
    for section in doc.sections:
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = " PAGE "
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        run._r.append(f1); run._r.append(it); run._r.append(f2)


def new_doc() -> Document:
    doc = Document()
    set_two_line_spacing(doc)
    add_line_numbers(doc)
    add_page_footer(doc)
    return doc


# --------------------------------------------------------------------------
# inline markdown
# --------------------------------------------------------------------------
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`[^`]+`)", re.S)


def add_inline(par, text: str) -> None:
    for piece in TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            par.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            par.add_run(piece[1:-1]).italic = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1]); r.font.name = "Courier New"
        else:
            par.add_run(piece)


# --------------------------------------------------------------------------
# markdown -> docx
# --------------------------------------------------------------------------
def render_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci in range(ncol):
            txt = row[ci] if ci < len(row) else ""
            p = cells[ci].paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, txt)
            if ri == 0:
                for r in p.runs:
                    r.bold = True


def render_markdown(doc: Document, md: str) -> None:
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # table block
        if stripped.startswith("|") and i + 1 < len(lines) and \
                set(lines[i + 1].strip()) <= set("|-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            render_table(doc, rows)
            doc.add_paragraph()
            continue

        if not stripped:
            i += 1
            continue

        if stripped.startswith("#"):
            lvl = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[lvl:].strip()
            h = doc.add_heading(level=min(lvl, 4))
            h.paragraph_format.line_spacing = 2.0
            add_inline(h, text)
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.line_spacing = 2.0
            add_inline(p, re.sub(r"^[-*]\s+", "", stripped))
            i += 1
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.line_spacing = 2.0
            add_inline(p, re.sub(r"^\d+[.)]\s+", "", stripped))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            # bare separators must never reach the docx (QA G3)
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        add_inline(p, stripped)
        i += 1


def parse_captions() -> dict[str, tuple[str, list[str]]]:
    """Parse figure_captions.md -> {"Fig 1": (heading, [body lines]), ...}."""
    caps = open(os.path.join(MD, "figure_captions.md"), encoding="utf-8").read()
    out: dict[str, tuple[str, list[str]]] = {}
    cur_key = None
    heading = ""
    body: list[str] = []
    for line in caps.split("\n"):
        if line.startswith("## "):
            if cur_key:
                out[cur_key] = (heading, body)
            heading = line[3:].strip()
            cur_key = heading.split(".")[0].strip()  # "Fig 1" ... "S1 Fig"
            body = []
        elif cur_key and not line.startswith("#"):
            body.append(line)
    if cur_key:
        out[cur_key] = (heading, body)
    return out


def embed_figures(doc: Document) -> None:
    """Embed the four main figures after the References section, with full
    captions from figure_captions.md. The supplementary figure S1 Fig is
    uploaded separately as Supporting Information and is not embedded."""
    caps = parse_captions()
    h = doc.add_heading(level=1)
    h.add_run("Figures")
    for num, fname, note in FIGURES:
        key = f"Fig {num}"
        path = os.path.join(FIGDIR, fname)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        if os.path.exists(path):
            p.add_run().add_picture(path, width=Inches(6.0))
        else:
            print(f"  [warn] missing figure {path}")
        # caption: bold "Fig N." + formal heading, then body paragraphs
        heading, body = caps.get(key, (key, []))
        cap_head = doc.add_paragraph()
        cap_head.paragraph_format.line_spacing = 2.0
        add_inline(cap_head, f"**{key}.** {heading.split('. ', 1)[-1]}")
        for b in body:
            b = b.strip()
            if not b or set(b) <= {"-"}:
                continue
            para = doc.add_paragraph()
            para.paragraph_format.line_spacing = 2.0
            add_inline(para, b)
        if note:
            n = doc.add_paragraph()
            n.paragraph_format.line_spacing = 2.0
            add_inline(n, f"*File note: {fname} — {note}.*")


# --------------------------------------------------------------------------
def build_manuscript(blinded: bool) -> str:
    src = "manuscript_blinded.md" if blinded else "manuscript.md"
    md = open(os.path.join(MD, src), encoding="utf-8").read()
    doc = new_doc()
    render_markdown(doc, md)
    # PLOS: main figures embedded at the end of the manuscript with full
    # captions; S1 Fig stays in 03_Supplementary as Supporting Information.
    embed_figures(doc)
    out_name = "Manuscript_blinded.docx" if blinded else "Manuscript_full.docx"
    out = os.path.join(OUT, out_name)
    doc.save(out)
    return out


def word_counts() -> tuple[int, int, int]:
    """Reproducible word counts, computed from the markdown at build time.

    PLOS convention (single rule, stated on the title page):
      * a "word" is a whitespace-delimited token containing at least one
        alphanumeric character (so '|', '---', '**' do not count);
      * tables (lines beginning with '|') are excluded from the main-text count;
      * the abstract is unstructured (limit 300 words) and sits between
        '## Abstract' and '## Author summary';
      * the Author Summary (150-200 words) sits between the abstract and the
        introduction;
      * main text = Introduction through Conclusions (PLOS order), tables
        excluded.

    Returns (main_text, abstract, author_summary) word counts.
    """
    md = open(os.path.join(MD, "manuscript.md"), encoding="utf-8").read()

    def count(s: str) -> int:
        return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])

    def strip_tables(s: str) -> str:
        return "\n".join(l for l in s.splitlines() if not l.strip().startswith("|"))

    abstract = md[md.index("## Abstract") + len("## Abstract"):md.index("## Author summary")]
    author_summary = md[md.index("## Author summary") + len("## Author summary"):md.index("## Introduction")]
    main_text = md[md.index("## Introduction"):md.index("## Declarations")]
    return (count(strip_tables(main_text)), count(strip_tables(abstract)),
            count(strip_tables(author_summary)))


def build_title_page() -> str:
    doc = new_doc()
    h = doc.add_heading(level=1); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_inline(h, paper_title())

    # Author block: final single-author block (user-confirmed), no reference
    # to paper 15 or any co-author.
    for label, value in [
        ("Author", "Tian-Yu Wang"),
        ("Affiliations",
         "Medical Office, General Affairs Office, Guangdong Peizheng College, "
         "Guangzhou 510830, Guangdong, China"),
        ("Corresponding author",
         "Tian-Yu Wang, Medical Office, General Affairs Office, Guangdong Peizheng "
         "College, Guangzhou 510830, Guangdong, China. "
         "Email: wangty0709@hotmail.com. ORCID: 0000-0002-5124-7965"),
        ("Funding", "The author received no specific funding for this work."),
    ]:
        p = doc.add_paragraph(); p.paragraph_format.line_spacing = 2.0
        add_inline(p, f"**{label}:** {value}")

    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 2.0
    add_inline(p, f"**Target journal:** {JOURNAL}")

    mt, ab, aus = word_counts()
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 2.0
    add_inline(p, f"**Word count:** abstract {ab} words (limit 300); main text "
                  f"(Introduction through Conclusions, tables excluded) {mt:,} "
                  f"words; Author Summary {aus} words (150-200). Counts are "
                  f"whitespace-delimited tokens with at least one alphanumeric "
                  f"character, computed from the manuscript source at build time.")

    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 2.0
    add_inline(p, "**Figures:** 4 main (Fig 1-4) plus 1 supporting-information "
                  "figure (S1 Fig). **Tables:** 7. **References:** 17.")

    out = os.path.join(OUT, "Title_Page.docx")
    doc.save(out)
    return out


COVER_LETTER = """Dear Editors of PLOS Computational Biology,

We submit for your consideration the manuscript "{title}".

Public-health agencies routinely turn surveillance data into model-based
recommendations - which antiviral tier to stock, when to commit, for which
season. The usual quality checks ask whether the model fits the data and
whether the posterior is well calibrated. This paper contributes a practical
diagnostic that asks a different, decision-level question: once the posterior
is well calibrated, is the recommended intervention actually identified by
the data at all? The diagnostic runs alongside standard calibration, uses
only the posterior a modelling team already holds, requires no additional
data, and flags the specific seasons in which the recommendation would be
untrustworthy.

The mechanism is concrete for influenza. Surveillance systems report cases,
not infections, so the reporting rate and the initial epidemic size enter the
likelihood only through their product - an exact structural ridge that no
amount of training or data volume removes. On a 192-cell synthetic grid the
recommended intervention tier flips with one-in-four probability under a
well-calibrated posterior, and a variance-weighted diagnostic separates
decision-sensitive from decision-insensitive settings where the natural
geometric alternative fails its own negative control. On 204 real CDC
FluView region-seasons the audit flags 7 of 204 (3.4%) as decision-uncertain;
six of the seven fall in the 2022-2023 season, the highest-volume season in
the sample. The conclusions are stable when the intervention cost ratio is
varied three-fold in either direction, and we report a null result -
parameter-level non-identifiability does not predict decision instability -
as such.

Why this manuscript suits PLOS Computational Biology: the paper develops and
validates a quantitative method - a decision-theoretic audit of calibrated
posteriors - against exact likelihood geometry, and demonstrates it on a
decade of public surveillance data, the combination of methodological rigour
and applied relevance the journal looks for. The work should inspire and
equip researchers beyond influenza: any calibrated posterior that feeds a
recommendation can be audited the same way, and the negative result that
parameter-level non-identifiability does not predict decision instability
gives the field a concrete reason to move decision-level diagnostics into
routine practice alongside calibration checks. Every statistic is traceable
to a seeded, CPU-only pipeline; all analysis code is provided as supporting
information, the data are from the public WHO FluNet and CDC FluView
surveillance systems, and no funding was received for this study.

The manuscript is not under consideration elsewhere. Disclosure of
AI-assisted work: During the preparation of this work the author used Kimi K3
(Moonshot AI) for code development and language editing. The author reviewed
and edited all AI-generated content and takes full responsibility for the
integrity and accuracy of the publication.

Sincerely,
Tian-Yu Wang
Medical Office, General Affairs Office, Guangdong Peizheng College
Guangzhou 510830, Guangdong, China
E-mail: wangty0709@hotmail.com
ORCID: 0000-0002-5124-7965
"""


def build_cover_letter() -> str:
    doc = new_doc()
    text = COVER_LETTER.format(title=paper_title())
    for para in [p for p in text.split("\n\n") if p.strip()]:
        p = doc.add_paragraph(); p.paragraph_format.line_spacing = 2.0
        add_inline(p, para.replace("\n", " "))
    out = os.path.join(OUT, "Cover_Letter.docx")
    doc.save(out)
    return out


def build_figure_captions() -> str:
    doc = new_doc()
    caps = open(os.path.join(MD, "figure_captions.md"), encoding="utf-8").read()
    render_markdown(doc, caps)
    out = os.path.join(OUT, "Figure_captions.docx")
    doc.save(out)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    made = [
        build_manuscript(False),
        build_manuscript(True),
        build_title_page(),
        build_cover_letter(),
        build_figure_captions(),
    ]
    print("=== built ===")
    for m in made:
        print(f"  {os.path.basename(m)}  {os.path.getsize(m):,} B")
    mt, ab, aus = word_counts()
    print(f"word counts (build-time): main={mt:,} abstract={ab} "
          f"author_summary={aus}")


if __name__ == "__main__":
    main()
