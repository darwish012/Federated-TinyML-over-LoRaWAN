"""Causal cadence and wider healthy-only representation study runner.

This module is intentionally executable only when explicitly invoked.  Tests
exercise the pure combination/evaluation helpers; the full study is not run by
importing this module.
"""
from __future__ import annotations
import csv, hashlib, json, os, pickle, sys, time
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_k]='1'
import numpy as np
HERE = __import__('pathlib').Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent/'healthy_only_suite_20260907'))
import wider_models as wider
import cadence
import run_suite as old_suite
import models as old_models
import run_ensembles as ensemble

OUT=ROOT/'results/healthy_only_causal_20260908'; CACHE=ROOT/'results/healthy_only_suite_20260907/cache'
OLD=ROOT/'results/healthy_only_suite_20260907'; PROTOCOL=HERE/'protocol.json'
RULES=('global_h4','global_h12','personalized_h12','svdd_h12','pca','feature_envelope','pca_and_global_h4','pca_and_global_h12','pca_and_personalized_h12','pca_and_svdd_h12','pca_and_global_h4_and_envelope','pca_and_global_h12_and_envelope','pca_and_personalized_h12_and_envelope','pca_and_svdd_h12_and_envelope')
def read(p): return json.loads(__import__('pathlib').Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with __import__('pathlib').Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def combine(flags,rule):
 if rule in ('global_h4','global_h12','personalized_h12','svdd_h12','pca','feature_envelope'): return np.asarray(flags[rule],bool)
 base=rule.removeprefix('pca_and_'); env=base.endswith('_and_envelope'); base=base.removesuffix('_and_envelope')
 out=np.asarray(flags['pca'],bool)&np.asarray(flags[base],bool)
 return out & np.asarray(flags['feature_envelope'],bool) if env else out
def evaluate(meta,arrays,flags,metadata=None):
 h,f,u=[np.asarray(arrays[k],dtype=int) for k in ('healthy_test_indices','unhealthy_test_indices','unscored_test_indices')]
 flags=np.asarray(flags,bool); ha=bool(np.any(flags[h])); fh=np.flatnonzero(flags[f]); first=np.flatnonzero(flags)
 expected='fault' if meta['truth'] is not None else ('healthy_control' if meta['truth_kind']=='healthy_control' else 'unknown')
 return {'machine_id':meta['machine_id'],'dataset':meta['dataset'],'expected':expected,'scored':expected!='unknown','strict_success':expected!='unknown' and not ha and (bool(len(fh)) or expected=='healthy_control'),'healthy_alarm':ha,'fault_detected':bool(len(fh)),'fault_delay_acquisitions':int(fh[0]) if len(fh) else None,'first_alarm_index':int(first[0]) if len(first) else None,'unknown_alarm':bool(np.any(flags[u])),'training_health_verified':meta['training_health_verified'],'cadence_metadata':metadata or {}}
def summarize(rows):
 s=[r for r in rows if r['scored']]; f=[r for r in s if r['expected']=='fault']; c=[r for r in s if r['expected']=='healthy_control']; d=[r['fault_delay_acquisitions'] for r in f if r['fault_detected']]
 return {'strict_successes':sum(r['strict_success'] for r in s),'scored':len(s),'strict_percent':100*sum(r['strict_success'] for r in s)/max(1,len(s)),'faults_detected':sum(r['fault_detected'] for r in f),'faults':len(f),'healthy_alarm_devices':sum(r['healthy_alarm'] for r in s),'control_alarms':sum(r['healthy_alarm'] for r in c),'controls':len(c),'fault_delay_median':float(np.median(d)) if d else None}
def _deps(manifest):
 paths=[PROTOCOL,HERE/'wider_models.py',HERE/'cadence.py',HERE/'run_study.py',HERE/'test_study.py',HERE/'../healthy_only_suite_20260907/run_suite.py',HERE/'../healthy_only_suite_20260907/run_ensembles.py',HERE/'../healthy_only_suite_20260907/calibration.py',HERE/'../healthy_only_suite_20260907/models.py',OLD/'summary.json',OLD/'fusion/summary.json',OLD/'cache/manifest.json']
 paths += [__import__('pathlib').Path(p) for p in manifest['files']]
 paths += [OLD/f'trials/global_ae_seed{s}.pkl' for s in (17070,17071,17072)]
 return {str(__import__('pathlib').Path(p).resolve()):sha(p) for p in paths}
def _check_lock(deps):
 lock=OUT/'study_lock.json'; OUT.mkdir(parents=True,exist_ok=True)
 if lock.exists():
  if read(lock)['dependencies']!=deps: raise RuntimeError('dependency drift; refusing resume')
 else: (lock).write_text(json.dumps({'dependencies':deps,'classification':read(PROTOCOL)['classification']},indent=2)+'\n',encoding='utf-8')
def _model_state(path):
 with path.open('rb') as f:return pickle.load(f)
def _metrics(rows): return summarize(rows)
def run():
 spec=read(PROTOCOL); manifest=read(CACHE/'manifest.json'); old_summary=read(OLD/'summary.json')
 if not old_summary.get('complete'): raise RuntimeError('old suite incomplete')
 if any(sha(p)!=v for p,v in old_summary.get('dependencies',{}).items()): raise RuntimeError('historical suite dependency drift')
 fusion_summary=read(OLD/'fusion/summary.json')
 if any(sha(p)!=v for p,v in fusion_summary.get('dependencies',{}).items()): raise RuntimeError('fusion dependency drift')
 if any(sha(p)!=v for p,v in manifest.get('config',{}).get('provenance_sha256',{}).items()): raise RuntimeError('cache provenance drift')
 if any(sha(p)!=v for p,v in manifest['files'].items()): raise RuntimeError('cache hash drift')
 deps=_deps(manifest); _check_lock(deps)
 devices=sorted(manifest['devices'],key=lambda x:x['machine_id']); arrays={}; clients={}
 if len(devices)!=70 or len({d['machine_id'] for d in devices})!=70: raise RuntimeError('cohort identity count mismatch')
 if sum(d['truth'] is not None for d in devices)!=59 or sum(d['truth_kind']=='healthy_control' for d in devices)!=6 or sum(d['truth'] is None and d['truth_kind']!='healthy_control' for d in devices)!=5: raise RuntimeError('cohort scored/fault/control/unknown counts mismatch')
 for meta in devices:
  with np.load(meta['path'],allow_pickle=False) as z: arrays[meta['machine_id']]={k:z[k].copy() for k in z.files}
  clients[meta['machine_id']]=arrays[meta['machine_id']]['x'][arrays[meta['machine_id']]['train_indices'].astype(int)]
 states={}
 for seed in spec['seeds']:
  for variant in spec['new_models']:
   p=OUT/f'{variant}_seed{seed}.pkl'; cp=OUT/f'{variant}_seed{seed}_checkpoint.json'
   if p.exists():
    if not cp.exists() or read(cp).get('input_dependencies')!=deps: raise RuntimeError(f'state checkpoint drift {variant} seed{seed}')
    if read(cp).get('state_sha256')!=sha(p): raise RuntimeError(f'saved model hash drift {variant} seed{seed}')
    states[(variant,seed)]=_model_state(p)
   else:
    states[(variant,seed)]=wider.fit(clients,variant,seed)
    with p.open('wb') as f:pickle.dump(states[(variant,seed)],f,protocol=5)
   checkpoint={'variant':variant,'seed':seed,'clients':len(clients),'input_dependencies':deps,'state_sha256':sha(p),'communication':wider.communication(states[(variant,seed)]),'mean_training_reconstruction_mse':float(np.mean([np.mean(wider.reconstruction_scores(states[(variant,seed)],mid,clients[mid])) for mid in clients]))}
   cp.write_text(json.dumps(checkpoint,indent=2)+'\n',encoding='utf-8')
   print(f'ready {variant} seed{seed}, train MSE {checkpoint["mean_training_reconstruction_mse"]:.6f}',flush=True)
 artifact_manifest={str(OUT/f'{v}_seed{s}.pkl'):sha(OUT/f'{v}_seed{s}.pkl') for s in spec['seeds'] for v in spec['new_models']}
 artifact_manifest.update({str(OUT/f'{v}_seed{s}_checkpoint.json'):sha(OUT/f'{v}_seed{s}_checkpoint.json') for s in spec['seeds'] for v in spec['new_models']})
 rows=[]
 for seed in spec['seeds']:
  h4=_model_state(OLD/f'trials/global_ae_seed{seed}.pkl')
  for meta in devices:
   mid=meta['machine_id']; a=arrays[mid]; x=a['x']; ti=a['train_indices'].astype(int); ci=a['calibration_indices'].astype(int)
   scores={'global_h4':old_models.score(h4,mid,x),'global_h12':wider.score(states[('global_h12',seed)],mid,x),'personalized_h12':wider.score(states[('personalized_h12',seed)],mid,x),'svdd_h12':wider.score(states[('svdd_h12',seed)],mid,x),'pca':old_suite.reference_scores('local_pca',x,x[ti],None)}
   scores['feature_envelope']=np.max(x/np.maximum(np.max(x[np.concatenate((ti,ci))],axis=0),1e-8),axis=1)
   if any(not np.all(np.isfinite(v)) for v in scores.values()): raise RuntimeError(f'nonfinite scores {mid}')
   for mode in ('dense','history_phase'):
    flag={}; cadmeta={}
    for key,v in scores.items():
     mult=1.25 if key=='feature_envelope' else 2.0
     if mode=='dense': flag[key]=cadence.dense_flags(v,ti,ci,21,13,mult); cadmeta[key]={'stride':1,'decision_count':len(v)}
     else:
      q=cadence.phase_vote(v,ti,ci,cadence.stride_from_history(ti,ci,100),21,13,mult); flag[key]=q['flags']; cadmeta[key]=q['metadata']; cadmeta[key]['decision_count']=int(sum(q['metadata']['phase_decisioncounts']))
    for rule in RULES:
     rr=evaluate(meta,a,combine(flag,rule),{'mode':mode,'routes':cadmeta}); rows.append({'rule':rule,'cadence':mode,'seed':seed,**rr})
  print(f'completed seed {seed}',flush=True)
  
 # Reference replication is a hard stop and intentionally happens before output summaries.
 fusion=read(OLD/'fusion/summary.json')
 for seed in spec['seeds']:
  got=summarize([r for r in rows if r['seed']==seed and r['rule']=='global_h4' and r['cadence']=='dense'])
  ref=next(c['metrics'] for c in fusion['grid'] if c['seed']==seed and c['window']==21 and c['k']==13 and c['multiplier']==2.0 and c['rule']=='ae')
  if got != ref: raise RuntimeError(f'h4 replication mismatch seed {seed}: {got} != {ref}')
 if any(summarize([r for r in rows if r['seed']==s and r['rule']=='pca' and r['cadence']=='dense'])['strict_successes']!=48 for s in spec['seeds']): raise RuntimeError('PCA replication mismatch')
 flat=[]
 for seed in spec['seeds']:
  for mode in ('dense','history_phase'):
   for rule in RULES:
     rr=[r for r in rows if r['seed']==seed and r['cadence']==mode and r['rule']==rule]
     if len(rr)!=70 or sum(r['scored'] for r in rr)!=65 or sum(r['expected']=='fault' for r in rr)!=59 or sum(r['expected']=='healthy_control' for r in rr)!=6: raise RuntimeError(f'config cohort mismatch {seed} {mode} {rule}')
     flat.append({'rule':rule,'cadence':mode,'seed':seed,'metrics':summarize(rr),'per_dataset':{d:summarize([r for r in rr if r['dataset']==d]) for d in sorted({r['dataset'] for r in rr})},'devices':rr})
 if len(flat)!=84: raise RuntimeError('configuration count mismatch')
 primary=[summarize([r for r in rows if r['seed']==s and r['rule']==spec['primary']['rule'] and r['cadence']==spec['primary']['cadence']]) for s in spec['seeds']]
 goal={'passes':all(x['strict_successes']>=59 and x['faults_detected']>=56 and x['control_alarms']==0 for x in primary),'primary':spec['primary'],'per_seed':primary}
 goal['all_fixed_pairs']=[{'rule':rule,'cadence':mode,'passes':all(c['metrics']['strict_successes']>=59 and c['metrics']['faults_detected']>=56 and c['metrics']['control_alarms']==0 for c in flat if c['rule']==rule and c['cadence']==mode)} for rule in RULES for mode in ('dense','history_phase')]
 if deps != _deps(manifest) or any(sha(p)!=v for p,v in old_summary['dependencies'].items()): raise RuntimeError('dependency drift after evaluation')
 if any(sha(p)!=v for p,v in fusion_summary['dependencies'].items()): raise RuntimeError('fusion dependency drift after evaluation')
 if any(sha(p)!=v for p,v in artifact_manifest.items()): raise RuntimeError('new model artifact drift after evaluation')
 summary={'complete':True,'classification':spec['classification'],'configuration_count':len(flat),'dependencies':deps,'artifact_hashes':artifact_manifest,'goal':goal,'results':flat}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['rule','cadence','seed']+list(flat[0]['metrics'])); w.writeheader(); [w.writerow({'rule':x['rule'],'cadence':x['cadence'],'seed':x['seed'],**x['metrics']}) for x in flat]
  report=['# Causal cadence and wider representation study','',spec['classification'],'','| Seed | Primary strict | Detected | Healthy alarms | Controls |','|---:|---:|---:|---:|---:|']
  report += [f"| {s} | {x['strict_successes']}/65 | {x['faults_detected']}/59 | {x['healthy_alarm_devices']} | {x['control_alarms']}/6 |" for s,x in zip(spec['seeds'],primary)]
  report += ['',f"Goal gate: {'PASS' if goal['passes'] else 'FAIL'} (fixed primary across all three seeds).",'All 84 configurations are retained in summary.json; no winner is selected.']
  (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 return summary
if __name__=='__main__': run()
