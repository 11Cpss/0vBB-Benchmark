"""Record and copy the working-tree manuscript without touching its originals."""
from pathlib import Path
import hashlib,json,shutil,subprocess
BUNDLE=Path(__file__).resolve().parents[1]
SOURCE=BUNDLE.parent/'overleaf'
IGNORE_DIRS={'.git','.runtime','__pycache__','.venv-energybench'}
IGNORE_SUFFIX={'.aux','.blg','.log','.out','.synctex.gz','.fls','.fdb_latexmk','.pyc'}
records=[]
for src in sorted(SOURCE.rglob('*')):
    rel=src.relative_to(SOURCE)
    if any(part in IGNORE_DIRS for part in rel.parts) or not src.is_file():continue
    if any(src.name.endswith(s) for s in IGNORE_SUFFIX):continue
    dst=BUNDLE/'paper'/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    records.append({'source':str(src),'bundle':str(dst.relative_to(BUNDLE)),'size_bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(BUNDLE/'provenance/paper_file_manifest.json').write_text(json.dumps({'files':records},indent=2)+'\n')
(BUNDLE/'provenance/paper_uncommitted_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=SOURCE))
print(f'Copied {len(records)} working-tree files; {sum(r["size_bytes"] for r in records)/2**20:.1f} MiB.')
