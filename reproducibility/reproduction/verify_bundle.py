#!/usr/bin/env python3
"""Verify immutable bundle files without reading original machine paths."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--with-event-data',action='store_true');a=p.parse_args()
    manifest=ROOT/'provenance/bundle_manifest.json'
    if not manifest.exists():raise FileNotFoundError('Bundle assembly manifest has not been finalized')
    data=json.loads(manifest.read_text());failures=[];checked=0
    for r in data['files']:
        if r.get('optional_event_data') and not a.with_event_data:continue
        path=ROOT/r['path']
        if path.is_symlink():failures.append({'path':r['path'],'error':'packaged file is a symlink'});continue
        if not path.is_file():failures.append({'path':r['path'],'error':'missing'});continue
        if digest(path)!=r['sha256']:failures.append({'path':r['path'],'error':'SHA256 mismatch'})
        checked+=1
    # User-created virtual environments contain normal Python symlinks; only
    # the immutable packaged files are subject to bundle symlink checks.
    result={'checked_files':checked,'include_event_data':a.with_event_data,'all_pass':not failures,'failures':failures}
    print(json.dumps(result,indent=2))
    if failures:sys.exit(1)
if __name__=='__main__':main()
