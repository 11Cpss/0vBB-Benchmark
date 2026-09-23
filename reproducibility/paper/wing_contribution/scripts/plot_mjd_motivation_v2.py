#!/usr/bin/env python3
"""Render manuscript Figure 2 from archived data in a publication layout.

Run from any directory:
    python plot_mjd_motivation_v2.py

The default output stem is figures/mjd_motivation_main_v2_generated.
PDF and SVG contain vector curves and editable text; PNG is a 600-dpi preview.
Neither of the supplied reference PDFs is read, embedded, or overwritten.

Panel (a): MJD clean test-event waveforms, low/high energy examples.
Panels (b,c): SuperNEMO 0nu/Bi214 test-partition energy diagnostic.
The default blue shading represents the exact retained 5-keV bins.
--shade-mode reference reproduces the approximate 1100--3000 keV band in
the professor's original mock-up; that band is NOT the measured support.

Visual conventions adapted from XENONnT/xenon_plot_style: thin lines,
inward ticks, frame-free legends, its blue/red/gray palette, and fixed-size
export. Typography remains the manuscript's Nimbus Roman / Computer Modern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mjd-v2-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from paper_style import PAPER_WIDTH_IN, apply_paper_style

WAVEFORMS = ROOT / "evaluation/mjd_style/data/mjd_example_waveforms.csv"
DATA = ROOT / "figures/data"
BLUE, RED, GRAY = "#4067B1", "#B9123E", "#555555"
SHADE = "#6CCEF5"
FONT_SIZE = 9
LINE_WIDTH = 1.0
FIGURE_HEIGHT_IN = 2.55
AXES_LEFT_PT = (38, 168, 298)
AXES_BOTTOM_PT, AXES_WIDTH_PT, AXES_HEIGHT_PT = 38, 90, 120
STYLE_REFERENCE = "https://github.com/XENONnT/xenon_plot_style"
STYLE_SOURCE = STYLE_REFERENCE + "/blob/master/xenonnt_plot_style/styles/xenonnt.mplstyle"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data():
    """Validate the plotted arrays without rerunning or modifying evaluation."""
    source_paths = [
        WAVEFORMS,
        DATA / "supernemo_matching_50keV.csv",
        DATA / "supernemo_matching_roc.csv",
        DATA / "supernemo_matching_evidence.json",
    ]
    waveform_rows = read_csv(source_paths[0])
    evidence = json.loads(source_paths[3].read_text())
    waves = {}
    for cohort in ("Low energy", "High energy"):
        rows = [r for r in waveform_rows if r["cohort"] == cohort]
        t = np.array([int(r["sample_index"]) for r in rows])
        amplitude = np.array([float(r["baseline_subtracted_adc"]) for r in rows])
        np.testing.assert_array_equal(t, np.arange(3800))
        assert np.isfinite(amplitude).all()
        assert len({r["event_id"] for r in rows}) == 1
        waves[cohort] = (t, amplitude, int(rows[0]["event_id"]), float(rows[0]["energy_keV"]))

    histogram_rows = read_csv(source_paths[1])
    histograms = {}
    for category in ("0nubb", "Bi214"):
        rows = [r for r in histogram_rows
                if r["distribution"] == "original" and r["category"] == category]
        edges = np.array([float(r["left_keV"]) for r in rows]
                         + [float(rows[-1]["right_keV"])])
        density = np.array([float(r["density_per_keV"]) for r in rows])
        np.testing.assert_allclose(np.diff(edges), 50, rtol=0, atol=0)
        np.testing.assert_array_equal([float(r["right_keV"]) for r in rows], edges[1:])
        assert np.all(density >= 0)
        np.testing.assert_allclose(np.dot(density, np.diff(edges)), 1, atol=1e-12, rtol=0)
        histograms[category] = (edges, density * 1000)

    roc_rows = read_csv(source_paths[2])
    curves = {}
    retained_spans = []
    for kind in ("original", "retained", "matched"):
        rows = [r for r in roc_rows if r["distribution"] == kind]
        x = np.array([float(r["fpr"]) for r in rows])
        y = np.array([float(r["tpr"]) for r in rows])
        auc = float(np.trapezoid(y, x))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0)
        np.testing.assert_allclose([x[0], y[0], x[-1], y[-1]], [0, 0, 1, 1], atol=1e-12)
        np.testing.assert_allclose(auc, evidence["stages"][kind]["auc"], atol=1e-10, rtol=0)
        curves[kind] = (x, y, auc)
        if kind == "retained":
            thresholds = np.array([float(r["threshold_keV"]) for r in rows])
            bins = np.unique(np.floor(thresholds[np.isfinite(thresholds)] / 5).astype(int))
            assert len(bins) == evidence["valid_matching_bins"]
            # Every occupied retained energy appears among the unsampled ROC
            # thresholds. Merge consecutive 5-keV bins, preserving sparse gaps.
            runs = np.split(bins, np.flatnonzero(np.diff(bins) != 1) + 1)
            low, high = evidence["common_support_keV"]
            retained_spans = [[max(float(run[0] * 5), low),
                               min(float((run[-1] + 1) * 5), high)] for run in runs]
        if kind == "original":
            operating_points = [r for r in rows
                                if float(r["threshold_keV"]) == evidence["threshold"]["keV"]]
            assert len(operating_points) == 1
            point = operating_points[0]
            np.testing.assert_allclose(
                [float(point["fpr"]), float(point["tpr"])],
                [evidence["threshold"]["fpr"], evidence["threshold"]["tpr"]],
                atol=1e-12, rtol=0,
            )
    return waves, histograms, curves, evidence, source_paths, retained_spans


def set_style():
    apply_paper_style()
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#262626",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "lines.dash_capstyle": "butt",
        "lines.solid_capstyle": "round",
        "svg.fonttype": "none",
        "path.simplify": False,
    })


def point_text(fig, x, y, label, **kwargs):
    """Place annotation baselines in physical points, independent of axes data."""
    return fig.text(x / (PAPER_WIDTH_IN * 72), y / (FIGURE_HEIGHT_IN * 72),
                    label, fontsize=FONT_SIZE, va="baseline", **kwargs)


def axes_text(ax, x, y, label, **kwargs):
    """Position text in physical points inside its panel."""
    return ax.text(x / AXES_WIDTH_PT, y / AXES_HEIGHT_PT, label,
                   transform=ax.transAxes, fontsize=FONT_SIZE, va="baseline",
                   zorder=7, **kwargs)


def legend_row(ax, y, label, color, style="-", value=None, shaded=False):
    """Place an inset legend row, with AUC values in a right-aligned column."""
    width, height = AXES_WIDTH_PT, AXES_HEIGHT_PT
    handle_left, handle_right, label_left = (7, 12, 26) if value is not None else (5, 13, 17)
    if shaded:
        ax.add_patch(Rectangle((handle_left / width, (y - 0.4) / height),
                               (handle_right - handle_left) / width, 5 / height,
                               transform=ax.transAxes, facecolor=SHADE,
                               edgecolor="none", alpha=0.20, zorder=7))
    else:
        # Compact AUC keys must still show both dashes and dots in grayscale.
        key_style = ({"-.": (0, (2.3, 0.8, 0.6, 0.8)),
                      "--": (0, (1.8, 0.9))}.get(style, style)
                     if value is not None else style)
        ax.add_artist(Line2D([handle_left / width, handle_right / width],
                             [(y + 2.3) / height] * 2, transform=ax.transAxes,
                             color=color, linestyle=key_style, linewidth=LINE_WIDTH,
                             zorder=7))
    axes_text(ax, label_left, y, label)
    if value is not None:
        axes_text(ax, width - 4, y, f"{value:.4f}", ha="right")


def draw_figure(waves, histograms, curves, evidence, shade_spans, shade_label="Retained bins"):
    set_style()
    # Physical dimensions are final-print dimensions. Do not use bbox='tight':
    # cropping would change the font size when LaTeX scales to \linewidth.
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, FIGURE_HEIGHT_IN))
    width, height = fig.get_size_inches() * 72
    axes = [fig.add_axes([left / width, AXES_BOTTOM_PT / height,
                         AXES_WIDTH_PT / width, AXES_HEIGHT_PT / height])
            for left in AXES_LEFT_PT]
    wave, spectrum, roc = axes

    for cohort, color, style in [("Low energy", BLUE, "-"),
                                  ("High energy", RED, "--")]:
        t, amplitude, _, _ = waves[cohort]
        wave.plot(t, amplitude, color=color, linestyle=style,
                  linewidth=LINE_WIDTH, zorder=3)
    wave.set(xlim=(0, 3800), ylim=(-120, 3600))
    wave.set_xticks([0, 2000, 3800])
    wave.set_yticks([0, 1000, 2000, 3000])
    wave.set_xlabel("Sample index")
    wave.set_ylabel("ADC counts")
    wave.xaxis.set_minor_locator(MultipleLocator(500))
    wave.yaxis.set_minor_locator(MultipleLocator(500))

    for span in shade_spans:
        spectrum.axvspan(*span, facecolor=SHADE, edgecolor="none", alpha=0.20, zorder=0)
    for category, label, color, style in [
        ("0nubb", "Signal-like", BLUE, "-"),
        ("Bi214", "Background-like", RED, "--"),
    ]:
        edges, density = histograms[category]
        spectrum.stairs(density, edges, color=color, linestyle=style,
                        linewidth=LINE_WIDTH, label=label, zorder=3)
    spectrum.set(xlim=(0, 3300), ylim=(0, 3.1), xlabel="Energy (keV)")
    spectrum.set_ylabel(r"Density ($10^{-3}$ keV$^{-1}$)")
    spectrum.set_xticks([0, 1000, 2000, 3000])
    spectrum.set_yticks([0, 1, 2, 3])
    spectrum.xaxis.set_minor_locator(MultipleLocator(500))
    spectrum.yaxis.set_minor_locator(MultipleLocator(0.5))

    for kind, label, color, style in [
        ("original", "Original", BLUE, "-"),
        ("retained", "Retained", GRAY, "-."),
        ("matched", "Matched", RED, "--"),
    ]:
        fpr, tpr, auc = curves[kind]
        roc.plot(fpr, tpr, color=color, linestyle=style, linewidth=LINE_WIDTH,
                 label=f"{label} {auc:.4f}", zorder=3)
    # The measured matched curve already traces the chance diagonal. Keep its
    # own line visible instead of overprinting it with a second reference line.
    point = evidence["threshold"]
    roc.scatter([point["fpr"]], [point["tpr"]], color=BLUE, edgecolors="white",
                s=18, linewidths=0.6, zorder=6)
    roc.annotate(f'{point["keV"]:.0f} keV', xy=(point["fpr"], point["tpr"]),
                 xytext=(0.26, 0.72), fontsize=FONT_SIZE,
                 arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.65,
                                 shrinkA=3, shrinkB=4))
    roc.set(xlim=(0, 1.02), ylim=(0, 1.04), xlabel="Background acceptance", ylabel="Signal efficiency")
    roc.set_xticks([0, 0.5, 1])
    roc.set_yticks([0, 0.5, 1])
    roc.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    roc.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    roc.xaxis.set_minor_locator(MultipleLocator(0.1))
    roc.yaxis.set_minor_locator(MultipleLocator(0.1))

    for index, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.tick_params(which="both", top=False, right=True)
        ax.yaxis.labelpad = 3
        ax.xaxis.set_label_coords(0.5, -16.2 / AXES_HEIGHT_PT)
        title = ("MJD waveforms", "Energy spectra", "Energy-only ROC")[index]
        point_text(fig, AXES_LEFT_PT[index], AXES_BOTTOM_PT + AXES_HEIGHT_PT + 9,
                   f"({chr(97 + index)}) {title}")

    # Taller equal panels make room for readable 9-pt inset legends. The AUC
    # table occupies the lower-right triangle; the diagonal passes only
    # through the whitespace between its line samples and text columns.
    for y, cohort, color, style in [(110, "Low energy", BLUE, "-"),
                                     (99, "High energy", RED, "--")]:
        energy = waves[cohort][3]
        legend_row(wave, y, f'{cohort.split()[0]}: {energy:.0f} keV', color, style)
    legend_row(spectrum, 110, "Signal-like", BLUE)
    legend_row(spectrum, 98, "Background-", RED, "--")
    axes_text(spectrum, 17, 88, "like")
    if shade_spans:
        legend_row(spectrum, 72, shade_label, SHADE, shaded=True)
    axes_text(roc, AXES_WIDTH_PT - 4, 38, "AUC", ha="right")
    for y, kind, label, color, style in [
        (25, "original", "Original", BLUE, "-"),
        (15, "retained", "Retained", GRAY, "-."),
        (5, "matched", "Matched", RED, "--"),
    ]:
        legend_row(roc, y, label, color, style, curves[kind][2])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT / "figures/mjd_motivation_main_v2_generated")
    parser.add_argument("--shade-mode", choices=["reference", "retained", "common-support", "none"],
                        default="retained")
    parser.add_argument("--shade-range", nargs=2, type=float, metavar=("LOW_KEV", "HIGH_KEV"),
                        help="Override the blue band's endpoints.")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    stem = args.output_stem.resolve()
    protected = {(ROOT / f"figures/{name}").resolve() for name in
                 ("mjd_motivation_main.pdf", "mjd_motivation_main_v2.pdf")}
    if stem.with_suffix(".pdf") in protected:
        parser.error("Use a new output name to preserve the two supplied reference PDFs.")
    waves, histograms, curves, evidence, sources, retained_spans = load_data()
    shade_spans = {"reference": [[1100., 3000.]],
                   "retained": retained_spans,
                   "common-support": [evidence["common_support_keV"]],
                   "none": []}[args.shade_mode]
    if args.shade_range is not None:
        shade_spans = [args.shade_range]
    shade_label = {"reference": "Reference band", "retained": "Retained bins",
                   "common-support": "Common support", "none": ""}[args.shade_mode]
    if args.shade_range is not None:
        shade_label = "Energy interval"
    if any(not (0 <= span[0] < span[1] <= 3300) for span in shade_spans):
        parser.error("The shaded range must satisfy 0 <= LOW < HIGH <= 3300.")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_figure(waves, histograms, curves, evidence, shade_spans, shade_label)
    for extension in ("pdf", "png", "svg"):
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        if extension == "svg":
            metadata = {"Date": None}
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, metadata=metadata)
    plt.close(fig)
    receipt = {
        "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources},
        "panels": {"a": "MJD clean test examples; baseline-subtracted ADC",
                   "b_c": "SuperNEMO 0nu/Bi214 test-partition energy diagnostic"},
        "style": {"font": "Nimbus Roman", "math_font": "Computer Modern",
                  "font_size_pt": FONT_SIZE, "figure_width_in": PAPER_WIDTH_IN,
                  "figure_height_in": FIGURE_HEIGHT_IN, "curve_linewidth_pt": LINE_WIDTH,
                  "panel_headings": "above panels",
                  "legends": "inside panels; AUC table lower right",
                  "style_reference": STYLE_SOURCE,
                  "palette": {"blue": BLUE, "red": RED, "gray": GRAY, "shade": SHADE},
                  "axes_bounds_points": [[left, AXES_BOTTOM_PT, AXES_WIDTH_PT, AXES_HEIGHT_PT]
                                         for left in AXES_LEFT_PT]},
        "waveforms": {cohort: {"event_id": value[2], "energy_keV": value[3], "samples": len(value[0])}
                      for cohort, value in waves.items()},
        "auc_from_full_roc": {kind: value[2] for kind, value in curves.items()},
        "threshold": evidence["threshold"],
        "shade_mode": "custom" if args.shade_range else args.shade_mode,
        "shade_label": shade_label,
        "shaded_spans_keV": shade_spans,
        "retained_bin_spans_keV": retained_spans,
        "recorded_common_support_keV": evidence["common_support_keV"],
        "shade_note": "Default retained shading uses the 283 measured 5-keV bins, "
                      "with gaps preserved and the upper edge clipped to common support. "
                      "The optional reference band is only a mock-up reproduction. "
                      "Common support includes sparse bins that are subsequently excluded.",
        "outputs": [str(stem.with_suffix(f".{ext}")) for ext in ("pdf", "png", "svg")],
    }
    stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("outputs", "auc_from_full_roc", "shaded_spans_keV")},
                     indent=2))


if __name__ == "__main__":
    main()
