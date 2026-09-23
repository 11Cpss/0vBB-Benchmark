#!/usr/bin/env python3
"""Compare isolated rerendered active PDFs with the frozen paper assets."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'mjd_motivation_main_v2_generated.pdf':'mjd_motivation_main_v2_generated.pdf','mjd_low_high_waveforms.pdf':'mjd/mjd_low_high_waveforms.pdf','next_capacity_scores.pdf':'next/next_capacity_scores.pdf','energy_bias_spectrum.pdf':'appendix/energy_bias_spectrum.pdf','energy_threshold_tradeoff.pdf':'appendix/energy_threshold_tradeoff.pdf','supernemo_extent_energy_population.pdf':'appendix/supernemo_extent_energy_population.pdf'}
records=[]
for name,rel in paths.items():
 p=B/'paper/wing_contribution/figures'/name;q=B/'validation/figures'/rel
 records.append({'paper_asset':str(p.relative_to(B)),'isolated_output':str(q.relative_to(B)),'source_sha256':sha(p),'generated_sha256':sha(q),'byte_identical':sha(p)==sha(q)})
r={'all_pass':all(x['byte_identical'] for x in records),'active_figures_regenerated':6,'archived_dataset_panels_not_regenerated':8,'records':records,'note':'Byte identity is demonstrated in the recorded local environment. Different Matplotlib/font versions may change PDF serialization while preserving data; the asset-only panels do not have a recovered generator.'}
(B/'validation/figures/render_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['all_pass'] else 1)
