#!/usr/bin/env python3
"""Render the three active SuperNEMO appendix figures.
Pass --extent-cdf code/figure_sources/data/supernemo_cohort_ecdf.csv.gz
from the reproduction root to include the complete tracker-extent CDF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/energybench-appendix-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from paper_style import PAPER_WIDTH_IN, apply_paper_style

DATA = ROOT / "evaluation/appendix_figures/data"
BLUE, ORANGE, GRAY = "#286A9B", "#C47723", "#555555"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name, output_dir, preview_dir):
    fig.savefig(output_dir / f"{name}.pdf", dpi=240,
                metadata={"CreationDate": None, "ModDate": None})
    if preview_dir:
        fig.savefig(preview_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def style_axes(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=.5)
    ax.set_axisbelow(True)


def spectrum(output_dir, preview_dir):
    data = pd.read_csv(DATA / "energy_shortcut_histograms.csv")
    fig, ax = plt.subplots(figsize=(.90 * PAPER_WIDTH_IN, 2.65))
    for cat, label, color, style in [
        ("0nubb", r"$0\nu\beta\beta$", BLUE, "-"),
        ("Bi214", r"$^{214}$Bi", ORANGE, "--"),
    ]:
        rows = data[data.category.eq(cat)]
        n = int(rows.n_events.iloc[0])
        assert rows["count"].sum() == n
        edges = np.r_[rows.left_keV, rows.right_keV.iloc[-1]]
        density = rows.density_per_keV.to_numpy()
        assert np.isclose(np.dot(density, np.diff(edges)), 1.)
        ax.stairs(1000 * density, edges, color=color, linestyle=style,
                  linewidth=1.25, label=label + f" ($n={n:,}$)")
    ax.axvline(2200, color=GRAY, linestyle=":", linewidth=1.)
    ax.text(2140, 2.98, "2200 keV", ha="right", va="top", fontsize=9, color=GRAY)
    ax.set(xlim=(0, 3300), ylim=(0, 3.1),
           xlabel=r"Reconstructed energy $E_1+E_2$ (keV)",
           ylabel=r"Density ($10^{-3}$ keV$^{-1}$)")
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.subplots_adjust(left=.125, right=.985, bottom=.20, top=.98)
    save(fig, "energy_bias_spectrum", output_dir, preview_dir)


def threshold(output_dir, preview_dir):
    data = pd.read_csv(DATA / "energy_threshold_scan.csv")
    cut = data[data.threshold_keV.eq(2200)].iloc[0]
    fig, ax = plt.subplots(figsize=(.85 * PAPER_WIDTH_IN, 2.70))
    for key, label, color, style in [
        ("signal_efficiency", r"Signal efficiency ($0\nu\beta\beta$)", BLUE, "-"),
        ("background_rejection", r"Background rejection ($^{214}$Bi)", ORANGE, "--"),
    ]:
        ax.plot(data.threshold_keV, data[key], color=color, linestyle=style,
                linewidth=1.25, label=label)
        ax.scatter([2200], [cut[key]], s=22, color=color,
                   edgecolors="white", linewidths=.6, zorder=5)
    ax.axvline(2200, color=GRAY, linestyle=":", linewidth=1.)
    ax.text(110, .83,
            f"At 2200 keV:\nSignal efficiency = {100 * cut.signal_efficiency:.1f}%\n"
            f"Background rejection = {100 * cut.background_rejection:.1f}%",
            fontsize=8, va="top")
    ax.set(xlim=(0, 3300), ylim=(0, 1.03),
           xlabel=r"Energy threshold $t$ (keV); signal if $E\geq t$",
           ylabel="Event fraction")
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_yticks([0, .25, .5, .75, 1.])
    fig.legend(*ax.get_legend_handles_labels(), frameon=False,
               loc="upper left", bbox_to_anchor=(.12, 1.0),
               borderaxespad=0, labelspacing=.2)
    style_axes(ax)
    fig.subplots_adjust(left=.12, right=.985, bottom=.20, top=.78)
    save(fig, "energy_threshold_tradeoff", output_dir, preview_dir)


def extent(cdf_path, output_dir, preview_dir):
    manifest = json.loads((DATA / "sources.json").read_text())
    assert sha(cdf_path) == manifest["external_extent_cdf"]["sha256"], "CDF source changed"
    data = pd.read_csv(DATA / "supernemo_conditional_extent_histogram.csv")
    medians = pd.read_csv(DATA / "extent_energy_medians.csv")
    cdf = pd.read_csv(cdf_path)
    summary = json.loads((DATA / "extent_summary.json").read_text())
    e_edges = np.array(summary["energy_edges_keV"])
    r_edges = np.array(summary["radius_edges_mm"])
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 4.35))
    gs = fig.add_gridspec(2, 2, left=.105, right=.875, top=.94, bottom=.23,
                         hspace=.54, wspace=.20, height_ratios=[1., 1.])
    cmap = LinearSegmentedColormap.from_list(
        "tracker_blue", ["#FFFFFF", "#D4E5EF", "#7AB1CF", "#2474A5", "#14415F"])
    cmap.set_bad("#EDEDED")
    for col, (cat, label) in enumerate([("Bi214", r"$^{214}$Bi"),
                                       ("0nubb", r"$0\nu\beta\beta$")]):
        stats = summary["classes"][cat]
        rows = data[data.category.eq(cat)]
        counts = rows["count"].to_numpy().reshape(len(e_edges) - 1, len(r_edges) - 1)
        probabilities = rows.probability_given_energy_bin.to_numpy().reshape(counts.shape)
        supported = rows.display_supported.to_numpy().reshape(counts.shape)
        assert counts.sum() == stats["n_events"]
        assert np.allclose(probabilities[counts.sum(axis=1) > 0].sum(axis=1), 1.)
        masked = np.ma.masked_array(probabilities.T, mask=~supported.T)
        ax = fig.add_subplot(gs[0, col])
        mesh = ax.pcolormesh(e_edges, r_edges, masked, cmap=cmap,
                            norm=Normalize(0, .13), rasterized=True, shading="flat")
        med = medians[medians.category.eq(cat)]
        ax.plot(med.energy_center_keV, med.median_radius_mm,
                color="#242424", linewidth=1.15, label="Bin median")
        ax.set(xlim=(300, 3200), ylim=(0, 800), xlabel="Energy (keV)",
               ylabel=r"Tracker extent $R_g$ (mm)" if col == 0 else "")
        ax.set_title(f"({chr(97 + col)}) " + label + f" ($n={stats['n_events']:,}$)",
                     loc="left", pad=6)
        ax.set_xticks([500, 1500, 2500])
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.legend(loc="upper right", frameon=True, facecolor="white",
                  edgecolor="none", framealpha=.85, handlelength=1.2,
                  borderpad=.2, handletextpad=.4)
        if col:
            ax.tick_params(labelleft=False)
        ec = fig.add_subplot(gs[1, col])
        for cohort, name, color, linestyle in [
            ("low", "Lowest", "#2474A5", "-"),
            ("high", "Highest", "#D5772B", "--"),
        ]:
            values = cdf[cdf.category.eq(cat) & cdf.energy_cohort.eq(cohort)]
            cs = stats["cohorts"][cohort]
            assert values.cohort_n.eq(cs["energy"]["n"]).all()
            assert np.isclose(values.ecdf.iloc[-1], 1.)
            lo, hi = cs["energy"]["minimum"], cs["energy"]["maximum"]
            ec.step(values.radius_gyration_mm, values.ecdf, where="post",
                    color=color, linestyle=linestyle, linewidth=1.25,
                    label=f"{name}: {lo:.0f}–{hi:.0f} keV")
        ec.axhline(.5, color="#BBBBBB", linewidth=.6, zorder=0)
        ec.set(xlim=(60, 2250), ylim=(0, 1.015), xscale="log",
               xlabel=r"$R_g$ (mm; log scale)",
               ylabel="Cumulative fraction" if col == 0 else "")
        ec.set_title(f"({chr(99 + col)}) {stats['cohorts']['low']['energy']['n']:,} per quartile",
                     loc="left", pad=6)
        ec.set_xticks([100, 300, 1000, 2000])
        ec.xaxis.set_major_formatter(ScalarFormatter())
        ec.set_yticks([0, .5, 1.])
        ec.minorticks_off()
        ec.legend(loc="upper left", bbox_to_anchor=(0., -.40),
                  frameon=False, handlelength=1.3, borderaxespad=0,
                  handletextpad=.4, labelspacing=.25)
        if col:
            ec.tick_params(labelleft=False)
    heatmap_pos = ax.get_position()
    cax = fig.add_axes([.899, heatmap_pos.y0, .015, heatmap_pos.height])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Probability per 20 mm bin", fontsize=8, labelpad=3)
    cb.set_ticks([0, .04, .08, .12])
    cb.ax.tick_params(labelsize=8, pad=2, length=2)
    save(fig, "supernemo_extent_energy_population", output_dir, preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--extent-cdf", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "sources.json").read_text())
    for name, source in manifest.items():
        if name != "external_extent_cdf":
            assert sha(DATA / name) == source["sha256"], f"Plot source changed: {name}"
    apply_paper_style()
    spectrum(args.output_dir, args.preview_dir)
    threshold(args.output_dir, args.preview_dir)
    if args.extent_cdf:
        extent(args.extent_cdf, args.output_dir, args.preview_dir)
    else:
        print("Extent figure skipped: supply its local-only --extent-cdf input.")


if __name__ == "__main__":
    main()
