#!/usr/bin/env python3
"""Create separate code and event-input ZIPs from the finalized local bundle."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def keep_validation(rel):
    parts=rel.parts
    if '__pycache__' in parts or any('mpl' in p for p in parts):return False
    if len(parts)>1 and parts[1] in ('initial_render',):return False
    if parts[:3]==('validation','compiled','manuscript'):return False
    if parts[:2]==('validation','metrics') and rel.name!='receipt.json':return False
    # Numerical and layout evidence is retained; duplicate image exports are not.
    if rel.suffix in ('.png','.svg','.pdf'):
        return rel.as_posix()=='validation/compiled/paper.pdf'
    return True

def main():
    OUT.mkdir(exist_ok=True)
    static=json.loads((ROOT/'provenance/bundle_manifest.json').read_text())['files']
    code=[ROOT/r['path'] for r in static if not r['optional_event_data']]
    code.append(ROOT/'provenance/bundle_manifest.json')
    code.extend(p for p in (ROOT/'validation').rglob('*') if p.is_file() and keep_validation(p.relative_to(ROOT)))
    data=[ROOT/r['path'] for r in static if r['optional_event_data']]
    records=[]
    for name,files in [('EnergyBench_code_20260923.zip',code),('EnergyBench_event_data_20260923.zip',data)]:
        path=OUT/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p in sorted(set(files)):
                z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(path) as z:
            bad=z.testzip()
            if bad:raise AssertionError('Corrupted archive member: '+bad)
            count=len(z.infolist())
        records.append({'file':name,'size_bytes':path.stat().st_size,'sha256':sha(path),'members':count})
        print(name,f'{path.stat().st_size/2**20:.2f} MiB',count,'members',flush=True)
    (OUT/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['file']+'\n' for r in records))
    (OUT/'delivery_manifest.json').write_text(json.dumps({'archives':records,'extract_into_same_parent_directory':True,'code_archive_excludes_raw_event_companion':True,'original_sources_modified':False},indent=2)+'\n')
if __name__=='__main__':main()
