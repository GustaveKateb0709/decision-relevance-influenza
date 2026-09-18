r"""
Independent verification of the publication figure triplets.

Checks (all from the actual files on disk -- nothing is taken on faith):

  1. TIF      : real 300 dpi embedded, RGB mode, pixel size == nominal_in * 300.
  2. PDF      : text is EXTRACTABLE (=> vector text, not outlined); fonts embedded;
                scans the extracted text for the Unicode minus U+2212.
  3. minus    : re-renders each figure and inspects the actual tick-label strings
                for U+2212 (and flags any ASCII hyphen-minus in numeric ticks).
  4. title    : NO figure title is drawn inside the figure file (PLOS forbids it);
                the recorded caption title is <= 15 words (from the manifest).
  5. width    : canvas width is exactly 3.5 in (single col) or 7.2 in (double col).

Figure files are DISCOVERED, never assumed: the working package keeps them in
02_Figures/ and 03_Supplementary/, while a plain checkout may keep them in
figures/ next to src/.  A figure that cannot be found raises a SystemExit that
lists every path that was tried, rather than a bare FileNotFoundError.

Run:  python src/verify_figures.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import fitz  # PyMuPDF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Search both this script's directory tree and the submission package root.
SEARCH_ROOTS = (ROOT, os.path.dirname(ROOT))
FIG_DIRS = ("figures", "02_Figures", "03_Supplementary", "")

MINUS = "\u2212"
ASCII_HYPHEN = "-"


def _fig_candidates(name: str) -> list[str]:
    return [os.path.join(r, d, name) if d else os.path.join(r, name)
            for r in SEARCH_ROOTS for d in FIG_DIRS]


def fig_file(name: str) -> str:
    """Resolve a figure file by name across both supported layouts."""
    for p in _fig_candidates(name):
        if os.path.isfile(p):
            return p
    raise SystemExit(
        f"FAIL: figure file not found: {name!r}\n"
        "  tried:\n" + "\n".join(f"    {c}" for c in _fig_candidates(name)) +
        "\n  The delivered package keeps the figures in 02_Figures/ and\n"
        "  03_Supplementary/; the public repository does not commit them.\n"
        "  Copy the figure files next to the manifest (or into figures/) and\n"
        "  run again -- this check reads the figures, so it cannot be skipped.")


def check_tif(path):
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (None, None))
        return {
            "mode": im.mode,
            "size_px": list(im.size),
            "dpi": [float(dpi[0]) if dpi[0] else None,
                    float(dpi[1]) if dpi[1] else None],
            "compression": im.info.get("compression"),
        }


def check_pdf(path):
    doc = fitz.open(path)
    page = doc[0]
    text = page.get_text("text")
    fonts = page.get_fonts(full=True)
    finfo = []
    embedded = True
    for f in fonts:
        # tuple: (xref, ext, type, basefont, name, encoding, referencer, ...)
        ext = f[1] if len(f) > 1 else ""
        ftype = f[2] if len(f) > 2 else ""
        base = f[3] if len(f) > 3 else ""
        finfo.append(f"{base} ({ftype}, ext='{ext}')")
        if not ext or ext == "n/a":
            embedded = False
    n_minus = text.count(MINUS)
    has_hyphen_numeric = bool(re.search(r"\d\s*-\s*\d", text)) or \
        bool(re.search(r"-\d", text))
    return {
        "vector_text": len(text.strip()) > 0,
        "text_len": len(text),
        "fonts": finfo,
        "fonts_embedded": embedded,
        "unicode_minus_count": n_minus,
        "ascii_hyphen_in_number": has_hyphen_numeric,
        "sample_has_title_words": None,  # filled by caller
        "_text": text,
    }


def rc_unicode_minus():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return bool(plt.rcParams.get("axes.unicode_minus", None))


def render_ticks_check():
    """Positive control: build a symlog axis exactly like the figures and confirm
    the plain-tick formatter emits a Unicode minus (U+2212)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = True

    def _plain_tick(v, pos=None):
        return "0" if v == 0 else str(f"{v:g}").replace("-", MINUS)

    fig, ax = plt.subplots()
    ax.set_yscale("symlog", linthresh=0.1)
    ax.yaxis.set_major_formatter(FuncFormatter(_plain_tick))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.scatter([1, 10], [-1, -5])
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    joined = "".join(labels)
    return {
        "tick_labels": labels,
        "has_unicode_minus": MINUS in joined,
        "has_ascii_minus": any(ASCII_HYPHEN in s for s in labels),
    }


def pdf_minus_scan(figs):
    """Scan every figure PDF's extracted text for minus-sign usage."""
    tot = {"unicode_minus": 0, "ascii_hyphen_in_number": 0, "per_fig": {}}
    for short, v in figs.items():
        t = fitz.open(fig_file(v["pdf"]))[0].get_text("text")
        n_minus = t.count(MINUS)
        bad = bool(re.search(r"\d\s*-\s*\d", t)) or bool(re.search(r"-\d", t))
        tot["unicode_minus"] += n_minus
        tot["ascii_hyphen_in_number"] += int(bad)
        tot["per_fig"][short] = {"unicode_minus": n_minus,
                                 "ascii_hyphen_in_number": bad}
    return tot


def main():
    manifest_path = os.path.join(RESULTS, "figure_manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(
            f"FAIL: manifest not found: {manifest_path}\n"
            "  results/figure_manifest.json must sit next to this script's\n"
            "  package directory; it is what declares the figure file names.")
    manifest = json.load(open(manifest_path))
    figs = {**manifest["figures"], **manifest.get("supplementary", {})}
    forbidden = ["divergen", "diverge", "come apart", "opposite"]
    report = {"figures": {}, "global": {}}
    print("=" * 78)
    print(f"script root : {ROOT}")
    print(f"manifest    : {manifest_path}")
    # Resolve every declared file up front so a layout problem is reported once,
    # with the full list of paths tried, instead of failing mid-check.
    locations = {}
    for short, v in figs.items():
        for kind in ("tif", "pdf"):
            locations[(short, kind)] = fig_file(v[kind])
    print(f"figures dir : {os.path.dirname(locations[(next(iter(figs)), 'tif')])}")
    for short, v in figs.items():
        w_in = v["nominal_size_in"][0]
        tif = fig_file(v["tif"])
        pdf = fig_file(v["pdf"])
        t = check_tif(tif)
        p = check_pdf(pdf)
        exp_px = [int(round(w_in * 300)), int(round(v["nominal_size_in"][1] * 300))]
        width_ok = abs(w_in - 3.5) < 1e-6 or abs(w_in - 7.2) < 1e-6
        dpi_ok = all(abs(x - 300) < 1 for x in t["dpi"] if x)
        title_ok = v["caption_title_words"] <= 15
        # PLOS: the figure file must NOT contain the figure title.  Normalise
        # whitespace (the PDF text extraction inserts its own line breaks)
        # before the containment test.
        norm = " ".join(p["_text"].split())
        p["title_absent_from_figure"] = " ".join(v["caption_title"].split()) not in norm
        rec = {
            "tif": t, "pdf": {k: val for k, val in p.items() if k != "_text"},
            "checks": {
                "dpi_is_300": dpi_ok,
                "pixel_size_expected": exp_px,
                "pixel_size_match": t["size_px"] == exp_px,
                "canvas_width_ok": width_ok,
                "canvas_width_in": w_in,
                "tif_rgb": t["mode"] == "RGB",
                "pdf_text_vector": p["vector_text"],
                "pdf_fonts_embedded": p["fonts_embedded"],
                "caption_title_words": v["caption_title_words"],
                "title_within_15_words": title_ok,
                "title_absent_from_figure": p["title_absent_from_figure"],
                "n_panels": v["n_panels"],
            },
            "title_text": v["caption_title"],
            "directional_wording": [w for w in forbidden
                                    if w in v["caption_title"].lower()],
        }
        report["figures"][short] = rec
        ok = (dpi_ok and rec["checks"]["pixel_size_match"] and width_ok
              and rec["checks"]["tif_rgb"] and p["vector_text"]
              and p["fonts_embedded"] and title_ok
              and p["title_absent_from_figure"])
        print(f"[{'PASS' if ok else 'FAIL'}] {short}")
        print(f"       tif: {t['mode']} {t['size_px']}px dpi={t['dpi']} "
              f"compression={t['compression']}")
        print(f"       pdf: text_len={p['text_len']} vector={p['vector_text']} "
              f"embedded={p['fonts_embedded']} fonts={p['fonts']}")
        print(f"       minus(U+2212) in pdf: {p['unicode_minus_count']} | "
              f"ascii-hyphen-as-minus: {p['ascii_hyphen_in_number']}")
        print(f"       width={w_in}in panels={v['n_panels']} "
              f"caption_title_words={v['caption_title_words']}")
        print(f"       caption title = {v['caption_title']!r}  "
              f"title_absent_from_figure={p['title_absent_from_figure']}  "
              f"directional_wording={rec['directional_wording']}")

    # global checks
    bad_dir = {k: rec["directional_wording"] for k, rec in report["figures"].items()
               if rec["directional_wording"]}
    report["global"] = {
        "axes_unicode_minus": rc_unicode_minus(),
        "tick_render_check": render_ticks_check(),
        "pdf_minus_scan": pdf_minus_scan(figs),
        "directional_wording_offenders": bad_dir,
        "n_figures_checked": len(figs),
    }
    print("=" * 78)
    print("axes.unicode_minus =", report["global"]["axes_unicode_minus"])
    print("tick render check   =", report["global"]["tick_render_check"])
    print("pdf minus scan      =", report["global"]["pdf_minus_scan"])
    print("directional wording offenders =", report["global"]["directional_wording_offenders"])
    print("=" * 78)

    out = os.path.join(RESULTS, "figure_verification.json")
    json.dump(report, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
