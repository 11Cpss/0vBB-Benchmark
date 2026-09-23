#!/usr/bin/env python3
"""Maintainer helper: freeze checksums of the assembled bundle's static files."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SELECTED=('benchmark','paper','code','reproduction','results','provenance','environments','event_data')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
files=[ROOT/'README.md']
for name in SELECTED:
    files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='bundle_manifest.json')
rows=[]
for p in sorted(set(files)):
    if p.is_symlink():raise AssertionError('Unexpected source symlink: '+str(p))
    rows.append({'path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p),'optional_event_data':p.relative_to(ROOT).parts[0]=='event_data'})
(ROOT/'provenance/bundle_manifest.json').write_text(json.dumps({'schema_version':1,'source_snapshots_preserved':True,'mutable_validation_outputs_excluded':True,'files':rows},indent=2)+'\n')
print('Frozen',len(rows),'file hashes.')
