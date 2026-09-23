# MJD waveform figures

Replays the original two MJD waveform figures from frozen plotted data, with normal-weight titles. No detector-file access, model inference, event reselection or benchmark evaluation is needed.

From the Overleaf repository root:

```bash
python wing_contribution/evaluation/mjd_style/render.py \
  --output-dir wing_contribution/figures
```

Requires Python with NumPy, SciPy and Matplotlib. The default title weight is `normal`. The optional `--title-weight bold` exists only to reproduce the audited style control. The command writes only `mjd_motivation_main.pdf` and `mjd_low_high_waveforms.pdf` in the selected output directory.

The main plot uses the unchanged archived waveform/population CSVs. The low/high plot uses exact float32 centered and normalized arrays for the already-selected events 2407207 and 2585508. All 410 population energies, quartile boundaries, selected event identities, plotted arrays and axis limits are retained. `data_provenance.json` records original source-file SHA256 values; `data/selected_waveforms_exact.npz` is a lossless subset of the archived arrays.

Both figures use Nimbus Roman text and Computer Modern mathematical labels
from `../paper_style.py`, matching the manuscript's text and math fonts.
Their canvases match the final paper widths (5.5 and 5.225 inches), with 9 pt
axis/panel text, 8 pt ticks and legends, and smaller inset labels. Positions,
line breaks and margins are adjusted for these final print sizes.
