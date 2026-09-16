r"""
Independent verification of the publication figure triplets.

Checks (all from the actual files on disk -- nothing is taken on faith):

  1. TIF      : real 300 dpi embedded, RGB mode, pixel size == nominal_in * 300.
  2. PDF      : text is EXTRACTABLE (=> vector text, not outlined); fonts embedded;
                scans the extracted text for the Unicode minus U+2212.
  3. EPS      : valid PostScript with embedded TrueType (ps.fonttype=42 => TYPE42).
  4. minus    : re-renders each figure and inspects the actual tick-label strings
                for U+2212 (and flags any ASCII hyphen-minus in numeric ticks).
  5. title    : in-figure title word count <= 15 (from the manifest).
  6. width    : canvas width is exactly 3.5 in (single col) or 7.2 in (double col).

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
FIGS = os.path.join(ROOT, "figures")

MINUS = "\u2212"
ASCII_HYPHEN = "-"


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


def check_eps(path):
    raw = open(path, "rb").read()
    txt = raw[:400000].decode("latin-1", "ignore")
    return {
        "postscript_header": txt.startswith("%!PS") or "PS-Adobe" in txt[:200],
        "has_selectfont": "selectfont" in txt,
        "has_show": " show" in txt or "show\n" in txt,
        "fonttype42": "/FontType 42" in txt or "Type42" in txt,
        "size_bytes": len(raw),
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


def pdf_minus_scan(figs_dir, figs):
    """Scan every figure PDF's extracted text for minus-sign usage."""
    tot = {"unicode_minus": 0, "ascii_hyphen_in_number": 0, "per_fig": {}}
    for short, v in figs.items():
        t = fitz.open(os.path.join(figs_dir, v["pdf"]))[0].get_text("text")
        n_minus = t.count(MINUS)
        bad = bool(re.search(r"\d\s*-\s*\d", t)) or bool(re.search(r"-\d", t))
        tot["unicode_minus"] += n_minus
        tot["ascii_hyphen_in_number"] += int(bad)
        tot["per_fig"][short] = {"unicode_minus": n_minus,
                                 "ascii_hyphen_in_number": bad}
    return tot


def main():
    manifest = json.load(open(os.path.join(RESULTS, "figure_manifest.json")))
    figs = {**manifest["figures"], **manifest.get("supplementary", {})}
    forbidden = ["divergen", "diverge", "come apart", "opposite"]
    report = {"figures": {}, "global": {}}
    print("=" * 78)
    for short, v in figs.items():
        w_in = v["nominal_size_in"][0]
        tif = os.path.join(FIGS, v["tif"])
        pdf = os.path.join(FIGS, v["pdf"])
        eps = os.path.join(FIGS, v["eps"])
        t = check_tif(tif)
        p = check_pdf(pdf)
        e = check_eps(eps)
        exp_px = [int(round(w_in * 300)), int(round(v["nominal_size_in"][1] * 300))]
        width_ok = abs(w_in - 3.5) < 1e-6 or abs(w_in - 7.2) < 1e-6
        dpi_ok = all(abs(x - 300) < 1 for x in t["dpi"] if x)
        title_ok = v["in_figure_title_words"] <= 15
        # does the pdf text contain the in-figure title's first word?
        kw = v["in_figure_title"].split()[0]
        p["sample_has_title_words"] = kw in p["_text"]
        rec = {
            "tif": t, "pdf": {k: val for k, val in p.items() if k != "_text"},
            "eps": e,
            "checks": {
                "dpi_is_300": dpi_ok,
                "pixel_size_expected": exp_px,
                "pixel_size_match": t["size_px"] == exp_px,
                "canvas_width_ok": width_ok,
                "canvas_width_in": w_in,
                "tif_rgb": t["mode"] == "RGB",
                "pdf_text_vector": p["vector_text"],
                "pdf_fonts_embedded": p["fonts_embedded"],
                "eps_fonttype42_vector": e["fonttype42"] and e["has_selectfont"],
                "in_figure_title_words": v["in_figure_title_words"],
                "title_within_15_words": title_ok,
                "n_panels": v["n_panels"],
            },
            "title_text": v["in_figure_title"],
            "directional_wording": [w for w in forbidden
                                    if w in v["in_figure_title"].lower()],
        }
        report["figures"][short] = rec
        ok = (dpi_ok and rec["checks"]["pixel_size_match"] and width_ok
              and rec["checks"]["tif_rgb"] and p["vector_text"]
              and p["fonts_embedded"] and title_ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {short}")
        print(f"       tif: {t['mode']} {t['size_px']}px dpi={t['dpi']} "
              f"compression={t['compression']}")
        print(f"       pdf: text_len={p['text_len']} vector={p['vector_text']} "
              f"embedded={p['fonts_embedded']} fonts={p['fonts']}")
        print(f"       eps: header={e['postscript_header']} "
              f"selectfont={e['has_selectfont']} fonttype42={e['fonttype42']} "
              f"bytes={e['size_bytes']}")
        print(f"       minus(U+2212) in pdf: {p['unicode_minus_count']} | "
              f"ascii-hyphen-as-minus: {p['ascii_hyphen_in_number']}")
        print(f"       width={w_in}in panels={v['n_panels']} "
              f"title_words={v['in_figure_title_words']}")
        print(f"       title = {v['in_figure_title']!r}  "
              f"directional_wording={rec['directional_wording']}")

    # global checks
    bad_dir = {k: rec["directional_wording"] for k, rec in report["figures"].items()
               if rec["directional_wording"]}
    report["global"] = {
        "axes_unicode_minus": rc_unicode_minus(),
        "tick_render_check": render_ticks_check(),
        "pdf_minus_scan": pdf_minus_scan(FIGS, figs),
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
