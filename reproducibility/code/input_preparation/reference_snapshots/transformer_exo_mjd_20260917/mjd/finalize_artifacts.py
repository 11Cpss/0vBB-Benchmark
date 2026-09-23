#!/usr/bin/env python3
"""Compact ROC-only legacy arrays and finish metadata verification (no new metrics)."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
sys.dont_write_bytecode = True
from recompute_mjd import HERE, save, sha, omit_roc_arrays


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    identity = json.loads((HERE/'event_identity_audit.json').read_text())
    with np.load(HERE/'test_metadata.npz') as fresh, np.load(identity['classic_comparison']['path']) as old:
        for key in ['energy_label','label','physical_event_id','id','run_number','detector','tp0']:
            fresh_key = {'energy_label':'energy_keV','physical_event_id':'event_id'}.get(key,key)
            equal = bool(np.array_equal(old[key], fresh[fresh_key]))
            assert equal, key
            identity['classic_comparison'][key+'_exact_equal'] = equal
    save(HERE/'event_identity_audit.json', identity)
    for record in manifest['ready']:
        run = Path(record['source_predictions']).parent.parent
        checkpoint = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        meta = dict(epoch=int(checkpoint['epoch']),validation_score=float(checkpoint['score']))
        source_summary = json.loads((run/'run_summary.json').read_text())
        assert meta['epoch'] == source_summary['best_epoch']
        assert meta['validation_score'] == source_summary['best_validation_score']
        record['checkpoint_metadata'] = meta
        assert sha(record['standardized_predictions']) == record['standardized_sha256']
        replay_path = HERE/'legacy_replay'/('MJD_'+record['model_key']+'.json')
        replay = json.loads(replay_path.read_text())
        replay['matching'] = omit_roc_arrays(replay['matching'])
        save(replay_path,replay)
        print(record['model_key'],meta,flush=True)
    manifest['recompute_script_sha256'] = sha(HERE/'recompute_mjd.py')
    save(HERE/'manifest.json',manifest)


if __name__ == '__main__':
    main()
