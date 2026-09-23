#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess
B=Path(__file__).resolve().parents[2]
A=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['paper_file_manifest.json','classic_source_manifest.json','transformer_source_manifest.json','input_preparation_source_manifest.json','figure_source_manifest.json']:
 manifest=json.loads((B/'provenance'/name).read_text())
 records=manifest.get('files',manifest.get('copied_external_sources',[]))
 for r in records:
  source=Path(r.get('source',r.get('source_path','')));target=B/r.get('bundle',r.get('bundle_path',''))
  expected=r.get('sha256',r.get('source_sha256'));target_expected=r.get('bundle_sha256',expected)
  c={'manifest':name,'bundle_path':str(target.relative_to(B)),'source_path':str(source),'source_exists':source.is_file(),'bundle_exists':target.is_file()}
  c['source_hash_unchanged']=source.is_file() and sha(source)==expected
  c['bundle_hash_matches']=target.is_file() and sha(target)==target_expected
  c['byte_identical']=c['source_hash_unchanged'] and c['bundle_hash_matches'] and expected==target_expected
  checks.append(c)
links=[]
for root,dirs,files in os.walk(B,followlinks=False):
 for name in dirs+files:
  p=Path(root)/name
  if p.is_symlink():links.append(str(p.relative_to(B)))
state=json.loads((B/'provenance/source_paper_state.json').read_text());original=Path(state['source'])
current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()
current_status=subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True)
events=json.loads((B/'provenance/event_inputs.json').read_text())['records'];event_results=[]
for r in events:
 p=B/'event_data'/r['relative_input'];event_results.append({'id':r['id'],'exists':p.is_file(),'hash_match':p.is_file() and sha(p)==r['input_sha256']})
historical=[]
for r in json.loads((B/'provenance/transformer_historical_inputs.json').read_text())['records']:
 p=B/r['bundle_file'];original_input=Path(r['source_file'])
 historical.append({'id':r['dataset']+'__'+r['model_key'],'exists':p.is_file(),'hash_match':p.is_file() and sha(p)==r['sha256'],'original_source_hash_unchanged':original_input.is_file() and sha(original_input)==r['sha256']})
result={'all_pass':all(x['byte_identical'] for x in checks) and not links and current_head==state['head'] and current_status==state['status_porcelain'] and all(x['hash_match'] for x in event_results) and all(x['hash_match'] and x['original_source_hash_unchanged'] for x in historical),'source_copy_files_checked':len(checks),'checks':checks,'symlinks':links,'original_paper_head_unchanged':current_head==state['head'],'original_paper_git_status_unchanged':current_status==state['status_porcelain'],'bundled_event_count':len(events)+len(historical),'profile_event_count':len(events),'historical_event_count':len(historical),'event_inputs':event_results,'historical_event_inputs':historical}
(A/'source_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','event_inputs','historical_event_inputs']},indent=2));print('FAILURES',[x for x in checks if not x['byte_identical']]);raise SystemExit(0 if result['all_pass'] else 1)
