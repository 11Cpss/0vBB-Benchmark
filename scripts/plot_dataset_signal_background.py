"""Publication-style signal/background figures for MJD, NEXT, and SuperNEMO.

Produces one signal figure and one background figure per dataset (six PDF +
PNG files total) with a shared font and a shared signal/background color
pair, sized to stay legible when placed as subfigures in a paper. MJD is a
1D waveform plot; NEXT and SuperNEMO are 3D event-topology displays (NEXT:
hit cloud colored by energy; SuperNEMO: two-track "V" from the tracker hits).

Usage:
    python scripts/plot_dataset_signal_background.py \
        --data-root /vast/pvenkata/vis224 --out-dir docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Shared categorical pair (dataviz skill default palette, slots 1 and 2):
# validated adjacent-pair CVD safety on a light surface.
SIGNAL_COLOR = "#2a78d6"      # blue
BACKGROUND_COLOR = "#eb6834"  # orange

SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "signal_blue", ["#d8e8fa", SIGNAL_COLOR, "#0d2f52"]
)
BACKGROUND_CMAP = LinearSegmentedColormap.from_list(
    "background_orange", ["#fbe2d2", BACKGROUND_COLOR, "#7a2c0c"]
)

# Punchier versions for the NEXT hit clouds: most hits are low-energy, so a
# colormap that starts near-white reads as washed out. These drop the pale
# end and are combined with a log color norm (hit energies span ~3 decades).
NEXT_SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "next_signal_blue", ["#8fbde8", SIGNAL_COLOR, "#0a2540"]
)
NEXT_BACKGROUND_CMAP = LinearSegmentedColormap.from_list(
    "next_background_orange", ["#f5ac74", BACKGROUND_COLOR, "#6b2408"]
)

# White-floored ramps for the EXO wire-vs-time images, so sub-threshold noise
# stays white and only real pulses take on the class color.
EXO_SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "exo_signal_blue", ["#ffffff", "#8fbde8", SIGNAL_COLOR, "#0a2540"]
)
EXO_BACKGROUND_CMAP = LinearSegmentedColormap.from_list(
    "exo_background_orange", ["#ffffff", "#f5ac74", BACKGROUND_COLOR, "#6b2408"]
)

FIGSIZE = (5.5, 4.3)
FIGSIZE_3D = (6.3, 5.6)
TITLE_SIZE = 19
LABEL_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 13


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf/.png")


# --------------------------------------------------------------------------
# MJD: baseline-subtracted, amplitude-normalized example waveforms.
# --------------------------------------------------------------------------

RISE_WINDOW = (900, 1550)


def _mjd_normalize(waveforms: np.ndarray, baseline_samples: int = 200) -> np.ndarray:
    baseline = waveforms[:, :baseline_samples].mean(axis=1, keepdims=True)
    waveforms = waveforms - baseline
    peak = np.max(np.abs(waveforms), axis=1, keepdims=True)
    return waveforms / peak


def _mjd_signal_examples(path: Path, n: int, baseline_samples: int = 200) -> np.ndarray:
    with h5py.File(path, "r") as f:
        clean = (
            f["psd_label_dcr"][:]
            & f["psd_label_high_avse"][:]
            & f["psd_label_low_avse"][:]
            & f["psd_label_lq"][:]
        )
        energy = f["energy_label"][:]
        candidates = np.flatnonzero(clean)
        target = np.percentile(energy[candidates], 70)
        order = candidates[np.argsort(np.abs(energy[candidates] - target))]
        picked = np.sort(order[:n])
        waveforms = f["raw_waveform"][picked].astype(np.float64)
    return _mjd_normalize(waveforms, baseline_samples)


def _mjd_multisite_background_examples(
    path: Path, n: int, baseline_samples: int = 200, scan_limit: int = 4000
) -> np.ndarray:
    """Pick background pulses that fail only the low-AvsE (multi-site) cut and
    show a visibly stepped rising edge, rather than a generic PSD reject that
    looks like a signal pulse once normalized."""

    with h5py.File(path, "r") as f:
        dcr = f["psd_label_dcr"][:]
        high = f["psd_label_high_avse"][:]
        low = f["psd_label_low_avse"][:]
        lq = f["psd_label_lq"][:]
        energy = f["energy_label"][:]
        fail_low_only = (~low) & high & dcr & lq

        candidates = np.flatnonzero(fail_low_only & (energy > 400) & (energy < 1500))
        candidates = np.sort(candidates)[:scan_limit]
        waveforms = f["raw_waveform"][candidates].astype(np.float64)

    normalized = _mjd_normalize(waveforms, baseline_samples)

    # Score each pulse by how large a second rise (shoulder) shows up in the
    # smoothed derivative away from the main rise: near 1.0 means a
    # genuine two-step, multi-site pulse.
    lo, hi = RISE_WINDOW
    kernel = np.ones(15) / 15
    scores = np.empty(normalized.shape[0])
    for i, waveform in enumerate(normalized):
        deriv = np.convolve(np.diff(waveform), kernel, mode="same")[lo:hi]
        peak_pos = int(np.argmax(deriv))
        masked = deriv.copy()
        masked[max(0, peak_pos - 30) : peak_pos + 30] = 0.0
        primary = deriv[peak_pos]
        scores[i] = float(np.max(masked) / primary) if primary > 0 else 0.0

    top = np.argsort(scores)[::-1][:n]
    return normalized[np.sort(top)]


def plot_mjd(data_root: Path, out_dir: Path) -> None:
    path = data_root / "MJD" / "MJD_Train_0.hdf5"
    signal = _mjd_signal_examples(path, n=4)
    background = _mjd_multisite_background_examples(path, n=4)
    specs = [
        (signal, SIGNAL_COLOR, "mjd_signal", "MJD: Signal"),
        (background, BACKGROUND_COLOR, "mjd_background", "MJD: Background"),
    ]
    for waveforms, color, name, title in specs:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sample_index = np.arange(waveforms.shape[1])
        for waveform in waveforms:
            ax.plot(sample_index, waveform, color=color, linewidth=1.8, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Normalized Amplitude (a.u.)")
        ax.set_xlim(*RISE_WINDOW)
        ax.set_ylim(-0.15, 1.15)
        save(fig, out_dir, name)


# --------------------------------------------------------------------------
# NEXT: single-event hit display, projected onto the event's principal axes.
# --------------------------------------------------------------------------

def _next_event_hits(path: Path, max_diameter_mm: float = 600.0, top_n: int = 30):
    with h5py.File(path, "r") as f:
        table = f["MC/hits/table"][:]
        event_id = table["values_block_0"][:, 0]
        vals = table["values_block_1"]
        x, y, z, energy = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

    # Prefer a high-statistics, spatially compact event so the plotted
    # topology isn't dominated by a single far-flung outlier hit.
    uniq, counts = np.unique(event_id, return_counts=True)
    order = np.argsort(counts)[::-1][:top_n]
    chosen = None
    for candidate in uniq[order]:
        mask = event_id == candidate
        pts = np.column_stack([x[mask], y[mask], z[mask]])
        diam = 2.0 * np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
        if diam <= max_diameter_mm:
            chosen = candidate
            break
    if chosen is None:
        chosen = uniq[order[0]]

    mask = event_id == chosen
    return x[mask], y[mask], z[mask], energy[mask]


def _next_multisite_event_hits(
    path: Path,
    gap_mm: float = 15.0,
    min_secondary_hits: int = 100,
    max_extent_mm: float = 600.0,
):
    """Pick an event whose hits split into two (or more) spatially disjoint
    clusters, i.e. a multi-site background topology with a visible gap
    between energy deposits, rather than one continuous track."""

    with h5py.File(path, "r") as f:
        table = f["MC/hits/table"][:]
        event_id = table["values_block_0"][:, 0]
        vals = table["values_block_1"]
        x, y, z, energy = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

    best_score = -1
    best_mask = None
    for candidate in np.unique(event_id):
        mask = event_id == candidate
        if mask.sum() < 500:
            continue
        pts = np.column_stack([x[mask], y[mask], z[mask]])
        extent = 2.0 * np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
        if extent > max_extent_mm:
            continue

        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=gap_mm, output_type="ndarray")
        if len(pairs) == 0:
            continue
        adjacency = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(pts), len(pts)),
        )
        _, labels = connected_components(adjacency, directed=False)
        sizes = np.sort(np.bincount(labels))[::-1]
        if len(sizes) < 2 or sizes[1] < min_secondary_hits:
            continue

        score = int(sizes[1])
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError(f"no clean multi-site event found in {path}")
    return x[best_mask], y[best_mask], z[best_mask], energy[best_mask]


def _principal_axis_projection_3d(x, y, z):
    points = np.column_stack([x, y, z]).astype(np.float64)
    points -= points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    projected = points @ vt.T
    return projected[:, 0], projected[:, 1], projected[:, 2]


def plot_next(data_root: Path, out_dir: Path) -> None:
    specs = [
        (
            data_root / "NEXT" / "0nubb_part_1" / "ATPC_0nubb_5bar_Efilt_5.0percent_smear_0.h5",
            _next_event_hits,
            NEXT_SIGNAL_CMAP,
            "next_signal",
            r"NEXT: Signal (0$\nu\beta\beta$)",
        ),
        (
            data_root / "NEXT" / "Bi_part_1" / "ATPC_Bi_ion_5bar_Efilt_5.0percent_smear_0.h5",
            _next_multisite_event_hits,
            NEXT_BACKGROUND_CMAP,
            "next_background",
            r"NEXT: Background ($^{214}$Bi)",
        ),
    ]
    for path, event_selector, cmap, name, title in specs:
        x, y, z, energy = event_selector(path)
        px, py, pz = _principal_axis_projection_3d(x, y, z)
        energy_kev = energy * 1000.0
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            px,
            py,
            pz,
            c=energy_kev,
            cmap=cmap,
            norm=LogNorm(vmin=energy_kev.min(), vmax=energy_kev.max()),
            s=9,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.65)
        cbar.set_label("Hit Energy [keV]", fontsize=LABEL_SIZE - 2)
        cbar.ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.set_title(title)
        ax.set_xlabel("PC 1 [mm]", labelpad=12)
        ax.set_ylabel("PC 2 [mm]", labelpad=12)
        ax.set_zlabel("PC 3 [mm]", labelpad=6)
        ax.tick_params(labelsize=TICK_SIZE - 3)
        ax.view_init(elev=18, azim=-60)
        save(fig, out_dir, name)


# --------------------------------------------------------------------------
# SuperNEMO: two-track event topology from tracker hit positions (tX, tY, tZ).
# --------------------------------------------------------------------------

def _kmeans2(points: np.ndarray, iters: int = 10) -> np.ndarray:
    """Tiny dependency-free 2-means: seeded by the farthest pair of points, to
    split one event's tracker hits into its two electron tracks."""

    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    centroids = np.stack([points[i], points[j]])
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iters):
        d0 = np.linalg.norm(points - centroids[0], axis=1)
        d1 = np.linalg.norm(points - centroids[1], axis=1)
        labels = (d1 < d0).astype(int)
        for k in (0, 1):
            if np.any(labels == k):
                centroids[k] = points[labels == k].mean(axis=0)
    return labels


# Signal: among candidate two-track events (filtered by hit count and by
# requiring the two tracker-hit clusters to meet at a tight, shared vertex),
# hand-picked by inspection for a clean, well-separated "V" that reads
# clearly at small size.
#
# Background: real 0nubb/2nubb candidates all reconstruct as a shared-vertex
# "V" much like the signal (the two-track selection is common to every merged
# file), so a same-style example wouldn't visibly differ. Bi214/Tl208 do,
# however, have a much heavier tail of events where the two tracks' closest
# points are far apart -- i.e. no common vertex, an accidental/mismatched
# pairing typical of these decays (99th percentile gap ~500mm vs ~220mm for
# 0nubb/2nubb). This event was hand-picked from that large-gap population for
# a dramatic, unambiguous separation.
SUPERNEMO_EXAMPLE_EVENTS = {
    "data_0nubb_merged.h5": 4683,
    "data_Bi214_merged.h5": 8579,
}
SUPERNEMO_COMMON_VERTEX = {
    "data_0nubb_merged.h5": True,
    "data_Bi214_merged.h5": False,
}


def _supernemo_event_tracks(
    path: Path, event_id: int, row_limit: int = 400_000
) -> tuple[np.ndarray, np.ndarray]:
    """Split one event's tracker hits into its two electron tracks and order
    each outward from their closest mutual approach so the line traces a
    clean path (the two tracks' nearest points become each track's start)."""

    with h5py.File(path, "r") as f:
        n = min(row_limit, f["ev_no"].shape[0])
        ev_no = f["ev_no"][:n]
        tx = f["tX"][:n]
        ty = f["tY"][:n]
        tz = f["tZ"][:n]

    mask = ev_no == event_id
    points = np.column_stack([tx[mask], ty[mask], tz[mask]])
    labels = _kmeans2(points)
    group_a, group_b = points[labels == 0], points[labels == 1]

    inter = np.linalg.norm(group_a[:, None, :] - group_b[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmin(inter), inter.shape)
    near_a, near_b = group_a[i], group_b[j]

    group_a = group_a[np.argsort(np.linalg.norm(group_a - near_a, axis=1))]
    group_b = group_b[np.argsort(np.linalg.norm(group_b - near_b, axis=1))]
    return group_a, group_b


def plot_supernemo(data_root: Path, out_dir: Path) -> None:
    root = data_root / "SuperNEMO"
    specs = [
        (
            root / "data_0nubb_merged.h5",
            SIGNAL_CMAP,
            "supernemo_signal",
            r"SuperNEMO: Signal (0$\nu\beta\beta$)",
        ),
        (
            root / "data_Bi214_merged.h5",
            BACKGROUND_CMAP,
            "supernemo_background",
            r"SuperNEMO: Background ($^{214}$Bi)",
        ),
    ]
    for path, cmap, name, title in specs:
        track_a, track_b = _supernemo_event_tracks(path, SUPERNEMO_EXAMPLE_EVENTS[path.name])

        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        for track, shade in ((track_a, 0.55), (track_b, 0.95)):
            ax.plot(
                track[:, 0],
                track[:, 1],
                track[:, 2],
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=cmap(shade),
            )

        if SUPERNEMO_COMMON_VERTEX[path.name]:
            vertex = 0.5 * (track_a[0] + track_b[0])
            ax.scatter(*vertex, color="black", marker="*", s=140, zorder=5, label="Vertex")
        else:
            gap = np.linalg.norm(track_a[0] - track_b[0])
            ax.plot(
                *zip(track_a[0], track_b[0]),
                color="black",
                linestyle="--",
                linewidth=1.3,
                marker="x",
                markersize=6,
                label=f"No shared vertex ({gap:.0f} mm gap)",
            )

        ax.set_title(title)
        ax.set_xlabel("X [mm]", labelpad=12)
        ax.set_ylabel("Y [mm]", labelpad=12)
        ax.set_zlabel("Z [mm]", labelpad=6)
        ax.tick_params(labelsize=TICK_SIZE - 3)
        ax.legend(loc="upper left", fontsize=LEGEND_SIZE - 1, frameon=False)
        ax.view_init(elev=18, azim=-60)
        save(fig, out_dir, name)


# --------------------------------------------------------------------------
# EXO-200: U-wire (charge-collection) channel-vs-time event display.
# --------------------------------------------------------------------------

# Waveforms are [226 channels, 600 samples]; U wires are channels 0:38 (positive
# side) and 76:114 (negative side). Each charge cluster shows up as its own
# collection pulse on the U wires, so a single-site (Charge_Cluster_Number==1)
# event has one pulse and a multi-site event has several, spread over wires
# and time. The file stores only one (x, y, z) per event, so a true 3D
# multi-cluster display isn't available; wire-vs-time is the spatial view.
#
# Hand-picked in run 8968 among events with Rotated_Energy in 2.3-2.6 MeV (so
# both classes sit near Q_bb): signal is a clean single pulse; background is a
# 4-cluster event whose U-wire pulses are well separated across both TPC
# halves and drift times.
EXO_EXAMPLE_EVENTS = {"signal": 101, "background": 217}
EXO_U_WIRES = (slice(0, 38), slice(76, 114))
EXO_BASELINE_SAMPLES = 150
EXO_SMOOTH_SAMPLES = 5
EXO_TIME_WINDOW = (180, 600)
# Display-only: a pulse is one wire (row) tall and vanishes when the figure is
# shrunk into a subfigure, so each half's image is max-filtered over this many
# neighboring wires. Set to 1 for the raw one-row-per-wire image.
EXO_DILATE_CHANNELS = 3


def _exo_u_wire_image(path: Path, event_index: int) -> tuple[np.ndarray, float, dict]:
    with h5py.File(path, "r") as f:
        waveform = f["Waveforms"][event_index].astype(np.float32)
        meta = {
            "clusters": int(f["Charge_Cluster_Number"][event_index]),
            "energy": float(f["Rotated_Energy"][event_index]),
        }

    u = np.concatenate([waveform[s] for s in EXO_U_WIRES], axis=0)
    u = u - u[:, :EXO_BASELINE_SAMPLES].mean(axis=1, keepdims=True)
    # Collection pulses go negative; flip so collected charge is positive.
    image = uniform_filter1d(-u, EXO_SMOOTH_SAMPLES, axis=1)
    noise_floor = 3.0 * float(np.median(image[:, 10:EXO_BASELINE_SAMPLES - 10].std(axis=1)))

    if EXO_DILATE_CHANNELS > 1:
        half = image.shape[0] // 2
        image = np.concatenate(
            [
                maximum_filter1d(image[:half], EXO_DILATE_CHANNELS, axis=0, mode="nearest"),
                maximum_filter1d(image[half:], EXO_DILATE_CHANNELS, axis=0, mode="nearest"),
            ],
            axis=0,
        )
    return image, noise_floor, meta


def plot_exo(data_root: Path, out_dir: Path) -> None:
    path = data_root / "EXO" / "8968.h5"
    specs = [
        ("signal", EXO_SIGNAL_CMAP, "exo_signal", "EXO-200: Signal (SS)"),
        ("background", EXO_BACKGROUND_CMAP, "exo_background", "EXO-200: Background (MS)"),
    ]
    for kind, cmap, name, title in specs:
        image, noise_floor, meta = _exo_u_wire_image(path, EXO_EXAMPLE_EVENTS[kind])
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.grid(False)
        artist = ax.imshow(
            image,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=PowerNorm(0.6, vmin=noise_floor, vmax=np.percentile(image, 99.9)),
            interpolation="nearest",
            extent=(0, image.shape[1], -0.5, image.shape[0] - 0.5),
            rasterized=True,
        )
        ax.axhline(37.5, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_xlim(*EXO_TIME_WINDOW)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("U-wire Channel")
        ax.set_yticks([0, 20, 38, 58, 75])
        cbar = fig.colorbar(artist, ax=ax, pad=0.02)
        cbar.set_label("Charge [ADC]", fontsize=LABEL_SIZE - 2)
        cbar.ax.tick_params(labelsize=TICK_SIZE - 2)
        print(f"{name}: event {EXO_EXAMPLE_EVENTS[kind]} {meta}")
        save(fig, out_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/vast/pvenkata/vis224"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    configure_style()
    plot_mjd(args.data_root, args.out_dir)
    plot_next(args.data_root, args.out_dir)
    plot_supernemo(args.data_root, args.out_dir)
    plot_exo(args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
