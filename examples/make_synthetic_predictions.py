#!/usr/bin/env python3
"""Create a small prediction bundle for the quick-start tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from energybench.data import PredictionBundle, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_predictions.npz")
    parser.add_argument("--events", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_signal = args.events // 2
    n_background_a = args.events // 4
    n_background_b = args.events - n_signal - n_background_a

    label = np.r_[
        np.ones(n_signal, dtype=int),
        np.zeros(n_background_a + n_background_b, dtype=int),
    ]
    category = np.asarray(
        ["signal"] * n_signal
        + ["background_A"] * n_background_a
        + ["background_B"] * n_background_b
    )
    # Deliberately different class spectra: inclusive AUC receives an energy
    # shortcut, while the matched AUC measures the remaining topology term.
    energy = np.r_[
        rng.normal(2.45, 0.32, n_signal),
        rng.normal(1.95, 0.48, n_background_a),
        rng.normal(2.85, 0.55, n_background_b),
    ]
    topology = rng.normal(0.75 * label - 0.2, 1.0, args.events)
    score = topology + 0.85 * (energy - 2.2)
    energy_pred = energy + rng.normal(0.0, 0.08 + 0.025 * energy, args.events)

    bundle = PredictionBundle(
        {
            "event_id": np.asarray(
                ["synthetic:%07d" % index for index in range(args.events)]
            ),
            "label": label,
            "category": category,
            "energy_condition": energy,
            "energy_target": energy,
            "score": score,
            "energy_pred": energy_pred,
            "sample_weight": np.ones(args.events, dtype=float),
            "split": np.asarray(["test"] * args.events),
        },
        metadata={
            "experiment": "synthetic",
            "dataset_id": "energybench-demo",
            "dataset_version": "1",
            "energy_unit": "MeV",
            "score_space": "logit",
            "seed": args.seed,
        },
    )
    destination = save_bundle(bundle, Path(args.output))
    print("Wrote %d events to %s" % (bundle.n_events, destination))


if __name__ == "__main__":
    main()
