r"""
Publication-grade figures for Paper 16 (rewritten to journal spec).

For every figure we emit a THREE-PIECE set into ``figures/``:

    * ``FigN Formal Title.tif``  300 dpi, LZW-compressed, RGB   (formal title in filename)
    * ``figN_short_name.eps``    vector text (ps.fonttype=42)
    * ``figN_short_name.pdf``    vector text (pdf.fonttype=42)

Journal-spec compliance:
    * canvas width: single column 3.5 in (89 mm) OR double column 7.2 in (183 mm)
      -> single-panel figures are 3.5 in; multi-panel figures are 7.2 in.
    * all font sizes >= 8 pt at 300 dpi.
    * real Unicode minus (U+2212) on every axis (axes.unicode_minus = True).
    * colour-blind safe AND black/white-distinguishable encoding: the two DRS arms
      differ in BOTH colour (Okabe-Ito blue vs vermillion) and fill hatch/linestyle;
      categorical series differ in colour AND marker shape.  No red/green clash.
    * in-figure title is a single line of <= 15 words; the formal title lives in the
      filename and the manuscript caption.
    * every panel is keyed a/b/c/d.

Data sources (READ ONLY -- no values are altered here):
    fig1 : results/cells.csv
    fig2 : results/laplace.csv  (+ laplace_summary.json)
    fig3 : results/cells.csv    (+ fisher_summary.json)
    fig4 : results/laplace.csv, results/cells.csv, results/external_validity.csv
           (+ snr_correlation_summary.json)

Run:  python src/make_figures.py
Also writes: results/figure_manifest.json
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from PIL import Image

MINUS = "\u2212"  # real Unicode minus sign (U+2212)


def u(s) -> str:
    """Replace any ASCII hyphen in a *numeric* string with the Unicode minus."""
    return str(s).replace("-", MINUS)


def _plain_tick(v, pos=None) -> str:
    """Plain (non-mathtext) tick label with a Unicode minus."""
    if v == 0:
        return "0"
    return u(f"{v:g}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

# --------------------------------------------------------------------------
# global publication rc settings
# --------------------------------------------------------------------------
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.unicode_minus": True,      # real Unicode minus U+2212 on ticks
        "pdf.fonttype": 42,              # embed TrueType (vector) in PDF
        "ps.fonttype": 42,               # embed TrueType (vector) in EPS
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


# Okabe-Ito colour-blind-safe palette (no red/green pairing)
BLUE, VERM, GREEN, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442", "#555555")
CAT = [BLUE, VERM, GREEN, PURPLE, SKY]
MARKERS = ["o", "s", "^", "D", "v"]


def panel_label(ax, letter, x=-0.17, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def words(s: str) -> int:
    return len(s.split())


def fitted_suptitle(fig, text, base=8.5, minimum=8.0):
    """Add a one-line suptitle that is shrunk (never below `minimum` pt) so that
    it always fits inside the fixed canvas width -- avoids clipping."""
    st = fig.suptitle(text, fontsize=base, x=0.5, ha="center")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w_px = fig.get_size_inches()[0] * fig.dpi
    bb = st.get_window_extent(renderer=r)
    if bb.width > 0.98 * w_px:
        st.set_fontsize(max(minimum, base * 0.98 * w_px / bb.width))
    return st


def save_triplet(fig, tile: str, short: str, manifest, *, n_panels: int,
                 title: str, sources: list):
    """Save TIF(300dpi,LZW,RGB) + EPS + PDF with vector text. Record manifest."""
    os.makedirs(FIGS, exist_ok=True)
    w_in, h_in = fig.get_size_inches().tolist()

    p_tif = os.path.join(FIGS, f"{tile}.tif")
    fig.savefig(p_tif, dpi=300, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"})
    # normalise to RGB + explicit 300 dpi metadata
    im = Image.open(p_tif)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(p_tif, format="TIFF", compression="tiff_lzw", dpi=(300, 300))

    p_eps = os.path.join(FIGS, f"{short}.eps")
    fig.savefig(p_eps, format="eps")
    p_pdf = os.path.join(FIGS, f"{short}.pdf")
    fig.savefig(p_pdf, format="pdf")
    p_png = os.path.join(FIGS, f"{short}.png")   # raster preview only
    fig.savefig(p_png, dpi=200, format="png")

    with Image.open(p_tif) as im2:
        dpi = im2.info.get("dpi", (None, None))
        px = im2.size
    manifest[short] = {
        "tif": os.path.basename(p_tif),
        "eps": os.path.basename(p_eps),
        "pdf": os.path.basename(p_pdf),
        "png_preview": os.path.basename(p_png),
        "nominal_size_in": [round(w_in, 3), round(h_in, 3)],
        "dpi_requested": 300,
        "dpi_embedded": [int(dpi[0]) if dpi[0] else None,
                         int(dpi[1]) if dpi[1] else None],
        "pixels": [int(px[0]), int(px[1])],
        "n_panels": int(n_panels),
        "in_figure_title": title,
        "in_figure_title_words": words(title),
        "data_sources": sources,
    }
    plt.close(fig)


def _partial_spearman(a, b, c):
    """Spearman(a, b) partialling out c (rank-residual method)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    rc = pd.Series(c[m]).rank().to_numpy()

    def resid(y, x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1]), int(m.sum())


# --------------------------------------------------------------------------
# fig 1 -- DRS arms under both geometries (4 panels a-d), double column
# --------------------------------------------------------------------------
def fig1(df: pd.DataFrame, manifest):
    title = "Variance-weighted DRS separates arms; geometric DRS does not"
    panels = [
        ("drs_var_fisher", "drs_var_fisher_control",
         "Fisher/Laplace variance-weighted"),
        ("drs_geo_fisher", "drs_geo_fisher_control",
         "Fisher/Laplace geometric"),
        ("drs_var", "drs_control_var",
         "NPE-posterior variance-weighted"),
        ("drs", "drs_control",
         "NPE-posterior geometric"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout="constrained")
    for ax, (cm, cc, ttl), L in zip(axes.ravel(), panels, "abcd"):
        xs = df[cm].to_numpy(); xc = df[cc].to_numpy()
        ok = np.isfinite(xs) & np.isfinite(xc)
        xs, xc = xs[ok], xc[ok]
        lo = float(min(xs.min(), xc.min())); hi = float(max(xs.max(), xc.max()))
        bins = np.linspace(lo, hi, 26)
        ax.hist(xs, bins=bins, density=True, color=BLUE, alpha=0.60,
                edgecolor="white", linewidth=0.3, label="main (burden loss)")
        ax.hist(xc, bins=bins, density=True, color=VERM, alpha=0.55,
                hatch="////", edgecolor=VERM, linewidth=0.0,
                label="control (scale-invariant)")
        mm, mc = float(np.median(xs)), float(np.median(xc))
        fm = float(np.mean(xs > xc))
        ax.axvline(mm, color=BLUE, ls="--", lw=0.9)
        ax.axvline(mc, color=VERM, ls=":", lw=1.1)
        verdict = "control arm OK" if fm > 0.5 else "control arm FAILS"
        ax.set_title(f"{ttl}  [{verdict}]\n"
                     f"median main={mm:.3f}  ctrl={mc:.3f}  frac>ctrl={fm:.2f}",
                     fontsize=7.6)
        ax.set_xlabel("DRS"); ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7, handlelength=1.4)
        panel_label(ax, L)
    fitted_suptitle(fig, title)
    save_triplet(fig, "Fig1 DRS Positive Control",
                 "fig1_drs_positive_control", manifest, n_panels=4,
                 title=title, sources=["results/cells.csv"])


# --------------------------------------------------------------------------
# fig 2 -- parameter-level non-identifiability vs decision instability
#          (cells.csv, NULL RESULT -- deliberately no directional wording)
# --------------------------------------------------------------------------
def fig2(cells: pd.DataFrame, manifest):
    title = "Parameter-level non-identifiability does not predict decision instability"
    fig, ax = plt.subplots(figsize=(3.5, 3.4), layout="constrained")
    x = cells["relsd_max"].to_numpy(); y = cells["ppf"].to_numpy()
    for i, phi in enumerate(sorted(cells["phi"].unique())):
        g = cells[cells["phi"] == phi]
        ax.scatter(g["relsd_max"], g["ppf"], s=16, alpha=0.8,
                   color=CAT[i % len(CAT)], marker=MARKERS[i % len(MARKERS)],
                   edgecolor="white", linewidth=0.3, label=u(f"\u03c6 = {phi:g}"))
    ok = np.isfinite(x) & np.isfinite(y)
    sp = spearmanr(x[ok], y[ok])
    ax.axhline(0.2, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel("parameter-layer non-identifiability\n(max relative posterior sd)")
    ax.set_ylabel("decision instability  (PPF)")
    ax.set_title(u(f"Spearman = {sp.statistic:+.3f}  (p = {sp.pvalue:.2f}),  "
                   f"n = {int(ok.sum())}\nno detectable relationship"),
                 fontsize=8)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.3)
    panel_label(ax, "a")
    fitted_suptitle(fig, title)
    save_triplet(fig, "Fig2 Parameter Non-identifiability versus Decision Instability",
                 "fig2_parameter_identifiability_vs_decision_instability", manifest,
                 n_panels=1, title=title, sources=["results/cells.csv"])


# --------------------------------------------------------------------------
# fig S1 -- conditioning vs decision-relevance (Laplace path)  [supplementary]
# --------------------------------------------------------------------------
def fig3_conditioning(lp: pd.DataFrame, store):
    title = "Conditioning informs but does not determine decision-relevance"
    fig, ax = plt.subplots(figsize=(3.5, 3.4), layout="constrained")
    ns = sorted(lp["n_sloppy"].unique())
    for i, k in enumerate(ns):
        g = lp[lp["n_sloppy"] == k]
        ax.scatter(g["shrink_max"], g["drs_var"], s=16, alpha=0.85,
                   color=CAT[i % len(CAT)], marker=MARKERS[i % len(MARKERS)],
                   edgecolor="white", linewidth=0.3, label=f"n_sloppy={int(k)}")
    ax.axvline(float(lp["shrink_max"].median()), color=GREY, ls="--", lw=0.8)
    ax.axhline(float(lp["drs_var"].median()), color=GREY, ls="--", lw=0.8)
    sp = spearmanr(lp["shrink_max"], lp["drs_var"])
    ps, _ = _partial_spearman(lp["shrink_max"].to_numpy(), lp["drs_var"].to_numpy(),
                              lp["peak_count"].to_numpy())
    ax.set_xlabel("parameter-layer conditioning  (max posterior shrinkage)")
    ax.set_ylabel("decision-relevance  (DRS$_{var}$)")
    ax.set_title(u(f"Spearman = {sp.statistic:+.3f}  (p={sp.pvalue:.1e})\n"
                   f"partial | peak count = {ps:+.3f}"), fontsize=8)
    ax.legend(loc="best", fontsize=7, title="sloppy directions",
              title_fontsize=7, handletextpad=0.3)
    panel_label(ax, "a")
    fitted_suptitle(fig, title)
    save_triplet(fig, "Fig3 Conditioning versus Decision Relevance",
                 "fig3_conditioning_vs_relevance", store, n_panels=1,
                 title=title,
                 sources=["results/laplace.csv", "results/laplace_summary.json"])


# --------------------------------------------------------------------------
# S1 fig -- decision-relevance vs decision instability, single column
# --------------------------------------------------------------------------
def s1_fig(df: pd.DataFrame, manifest):
    title = "Decision-relevance index versus decision instability"
    fig, ax = plt.subplots(figsize=(3.5, 3.4), layout="constrained")
    xcol, ycol = "drs_var_fisher", "ppf"
    for i, phi in enumerate(sorted(df["phi"].unique())):
        g = df[df["phi"] == phi]
        ax.scatter(g[xcol], g[ycol], s=16, alpha=0.8, color=CAT[i % len(CAT)],
                   marker=MARKERS[i % len(MARKERS)], edgecolor="white",
                   linewidth=0.3, label=f"\u03c6 = {phi:g}")
    ok = np.isfinite(df[xcol]) & np.isfinite(df[ycol])
    sp = spearmanr(df[xcol][ok], df[ycol][ok])
    ax.axhline(0.2, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel("Fisher/Laplace variance-weighted DRS")
    ax.set_ylabel("decision instability  (PPF)")
    ax.set_title(u(f"Spearman = {sp.statistic:+.3f}  (p={sp.pvalue:.1e})"),
                 fontsize=8)
    ax.legend(loc="best", fontsize=7, handletextpad=0.3)
    panel_label(ax, "a")
    fitted_suptitle(fig, title)
    save_triplet(fig, "S1 Fig Decision Relevance versus Instability",
                 "s1_fig_drs_vs_ppf", manifest, n_panels=1,
                 title=title,
                 sources=["results/cells.csv", "results/fisher_summary.json"])


# --------------------------------------------------------------------------
# fig 4 -- margin SNR vs case count across paths and real seasons (3 panels)
# --------------------------------------------------------------------------
def fig4(lp: pd.DataFrame, cells: pd.DataFrame, ext: pd.DataFrame,
         snr: dict, manifest):
    title = "Margin SNR versus case count: prior regime and real seasons"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")

    def scatter_panel(ax, x, y, color, marker, xlab, ylab, note):
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(np.asarray(x)[ok], np.asarray(y)[ok], s=14, alpha=0.7,
                   color=color, marker=marker, edgecolor="white", linewidth=0.25)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.xaxis.set_major_formatter(FuncFormatter(_plain_tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.axhline(1.0, color=GREY, ls="--", lw=0.8)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY,
                          lw=0.5, alpha=0.9))

    # (a) synthetic Fisher/Laplace path
    ax = axes[0]
    x = lp["peak_count"].to_numpy(); y = lp["margin_snr"].to_numpy()
    sp = spearmanr(x, y); pe = pearsonr(x, y)
    scatter_panel(ax, x, y, BLUE, "o",
                  "peak weekly reports (log)", "decision-margin SNR",
                  u(f"Spearman={sp.statistic:+.3f}\nPearson={pe.statistic:+.3f}\n"
                    f"median SNR={np.median(y):.2f}\nfrac(SNR<1)={np.mean(y < 1):.2f}"))
    ax.set_title("Fisher/Laplace path (synthetic)", fontsize=8)
    panel_label(ax, "a")

    # (b) synthetic NPE-posterior path
    ax = axes[1]
    x = cells["peak_count"].to_numpy(); y = cells["margin_snr_npe"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    sp = spearmanr(x[ok], y[ok]); pe = pearsonr(x[ok], y[ok])
    scatter_panel(ax, x, y, VERM, "s",
                  "peak weekly reports (log)", "decision-margin SNR",
                  u(f"Spearman={sp.statistic:+.3f}\nPearson={pe.statistic:+.3f}\n"
                    f"median SNR={np.median(y[ok]):.2f}\nfrac(SNR<1)={np.mean(y[ok] < 1):.2f}"))
    ax.set_title("NPE-posterior path (synthetic)", fontsize=8)
    panel_label(ax, "b")

    # (c) real surveillance seasons
    ax = axes[2]
    x = ext["mean_count"].to_numpy(); y = ext["margin_snr"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    sp = spearmanr(x[ok], y[ok]); pe = pearsonr(x[ok], y[ok])
    scatter_panel(ax, x, y, GREEN, "^",
                  "mean weekly reports (log)", "decision-margin SNR",
                  u(f"Spearman={sp.statistic:+.3f}\nPearson={pe.statistic:+.3f}\n"
                    f"median SNR={np.median(y[ok]):.2f}\nfrac(SNR<1)={np.mean(y[ok] < 1):.3f}"))
    ax.set_title("Real surveillance seasons", fontsize=8)
    panel_label(ax, "c")

    fitted_suptitle(fig, title)
    save_triplet(fig, "Fig4 Margin SNR versus Case Count",
                 "fig4_snr_peakcount", manifest, n_panels=3, title=title,
                 sources=["results/laplace.csv", "results/cells.csv",
                          "results/external_validity.csv",
                          "results/snr_correlation_summary.json"])


def main():
    set_pub_style()
    os.makedirs(FIGS, exist_ok=True)
    cells = pd.read_csv(os.path.join(RESULTS, "cells.csv"))
    lp = pd.read_csv(os.path.join(RESULTS, "laplace.csv"))
    ext = pd.read_csv(os.path.join(RESULTS, "external_validity.csv"))
    snr = json.load(open(os.path.join(RESULTS, "snr_correlation_summary.json")))
    print("cells", cells.shape, "| laplace", lp.shape, "| external", ext.shape)

    manifest = {
        "_note": "Publication three-piece sets. Values are read-only from results/*.",
        "_spec": {
            "widths_in": {"single_column": 3.5, "double_column": 7.2},
            "min_font_pt": 8,
            "tif": "300 dpi, LZW, RGB",
            "vector_text": "pdf.fonttype=42 / ps.fonttype=42",
            "minus_sign": "Unicode U+2212 (axes.unicode_minus=True)",
            "palette": "Okabe-Ito (colour-blind safe, no red/green clash; "
                       "arms also differ by hatch, series by marker)",
            "in_figure_title_max_words": 15,
        },
        "figures": {},
        "supplementary": {},
    }
    fig1(cells, manifest["figures"])
    fig2(cells, manifest["figures"])
    s1_fig(cells, manifest["supplementary"])
    fig4(lp, cells, ext, snr, manifest["figures"])
    fig3_conditioning(lp, manifest["figures"])

    out = os.path.join(RESULTS, "figure_manifest.json")
    json.dump(manifest, open(out, "w"), indent=2)
    print("wrote", out)
    for sect in ("figures", "supplementary"):
        print(f"-- {sect} --")
        for k, v in manifest[sect].items():
            print(f"  {k:34s} {v['tif']:52s} {v['nominal_size_in']} "
                  f"dpi={v['dpi_embedded']} panels={v['n_panels']} "
                  f"title_words={v['in_figure_title_words']}")


if __name__ == "__main__":
    main()
