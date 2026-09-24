# MJD waveform figure

From the `reproducibility/` root:

```bash
python paper/wing_contribution/evaluation/mjd_style/render.py --output-dir outputs/mjd
```

This renders the active low/high waveform appendix figure. `data/` contains the
exact two selected waveforms, the 410-event detector/run population metadata,
and the deterministic selection rule. Waveform values are not recomputed from
rounded CSV values: the renderer uses `selected_waveforms_exact.npz`.

The main motivation figure uses the waveform CSV through
`paper/wing_contribution/scripts/plot_mjd_motivation_v2.py`.
