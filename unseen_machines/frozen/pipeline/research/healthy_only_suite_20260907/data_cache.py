"""Memory-bounded healthy-only 70-machine cache exporter."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
import raw_online_federated_v64_baselines as raw
import raw_online_federated_v62 as v62
import raw_online_federated_v45_cohort as cohort
import thesis_70_improvement as fix
import thesis_70_device_federated as old
import thesis_participating_node_scaling as trainer
AUDIT=ROOT/'results/thesis_70_improvement_dev1/partition_audit.csv'
MODEL=ROOT/'results/thesis_70_improvement_dev1/corrected_model.npz'
OUTPUT=ROOT/'results/healthy_only_suite_20260907/cache'
FEATURE_INDICES=np.asarray(fix.read_json(fix.PROTOCOL)['features']['raw_feature_indices'],dtype=np.int64)
FIELDS=('train_indices','calibration_indices','healthy_test_indices','unhealthy_test_indices','unscored_test_indices')
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def array_sha256(a):
 a=np.ascontiguousarray(a); return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
def audit():
 with AUDIT.open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
 if len(r)!=70 or len({x['machine_id'] for x in r})!=70: raise RuntimeError('audit must contain 70 unique machines')
 return r
def metadata():
 reg={x['client_id']:x for x in cohort.stream_registry()}; groups=cohort.group_by_experiment(list(reg)); out={}
 for d0 in v62._development_devices():
  d=dict(d0); mid=str(d['machine_id']); d['feature_paths']=([raw._feature_path(mid)] if d['tier']=='ldm_upb_complete_lifecycle' else [Path(reg[m]['path']) for m in groups[mid]]); out[mid]=d
 transitions=fix.read_json(old.TRANSITIONS)['transitions']
 for m in old.ADDED: out[f'ldm_upb::{m}']={'machine_id':f'ldm_upb::{m}','tier':'ldm_upb_complete_lifecycle','truth_kind':'published_transition','truth':int(transitions[m]),'feature_paths':[old.LDM_ROOT/'features'/f'ldm_upb__{m}.npz']}
 for b in ('K003','K004','K005','K006'):
  p=raw.V63_ROOT/'paderborn_predictions'/f'paderborn__{b}.npz'; out[f'paderborn::{b}']={'machine_id':f'paderborn::{b}','tier':'paderborn_healthy_snapshot','truth_kind':'healthy_control','truth':None,'feature_paths':[p]}
 if len(out)!=70: raise RuntimeError('metadata cohort is not 70')
 return out,reg
def streams(meta,reg):
 if meta['tier'] in ('ldm_upb_complete_lifecycle','paderborn_healthy_snapshot'):
  with np.load(meta['feature_paths'][0],allow_pickle=False) as z:a=[np.asarray(z['features'],dtype=np.float64)]
 else:
  a=[np.asarray(cohort.load_stream(next(v for v in reg.values() if Path(v['path']).resolve()==p.resolve()))[0],dtype=np.float64) for p in meta['feature_paths']]
 n=min(map(len,a)); return [x[:n] for x in a]
def physical_matrix(stream_list,train_indices):
 out=[]; ti=np.asarray(train_indices,dtype=np.int64)
 for s in stream_list:
  v=np.asarray(s,dtype=np.float64)[:,FEATURE_INDICES]; fit=v[ti]; c=np.median(fit,axis=0); mad=1.4826*np.median(np.abs(fit-c),axis=0); scale=np.where(np.maximum(mad,np.std(fit,axis=0))>1e-8,np.maximum(mad,np.std(fit,axis=0)),1.0); out.append(np.abs(np.arcsinh((v-c)/scale)))
 x=np.max(np.stack(out),axis=0)
 if x.dtype!=np.float64 or not np.all(np.isfinite(x)): raise RuntimeError('non-finite physical matrix')
 return x
def indices(row):
 counts=[int(row[k]) for k in ('healthy_train_rows','healthy_calibration_rows','healthy_test_rows','unhealthy_test_rows','unscored_prediction_rows')]; out={}; start=0
 for k,n in zip(FIELDS,counts): out[k]=np.arange(start,start+n,dtype=np.int64); start+=n
 return out
def export(output=OUTPUT):
 rows=audit(); meta,reg=metadata(); output=Path(output); output.mkdir(parents=True,exist_ok=True); devices=[]
 if {r['machine_id'] for r in rows}!=set(meta): raise RuntimeError('identity mismatch')
 for number,row in enumerate(rows,1):
  mid=row['machine_id']; ix=indices(row); ss=streams(meta[mid],reg)
  if sum(map(len,ix.values()))!=len(ss[0]): raise RuntimeError(f'partition mismatch {mid}')
  x=physical_matrix(ss,ix['train_indices']); p=output/(mid.replace('::','__')+'.npz'); np.savez_compressed(p,x=x,**ix)
  truth=None if meta[mid].get('truth') is None else int(meta[mid]['truth']); src={str(q.resolve()):sha256(q) for q in meta[mid]['feature_paths']}
  devices.append({'machine_id':mid,'dataset':mid.split('::',1)[0],'truth_kind':str(meta[mid].get('truth_kind',row['truth_kind'])),'truth':truth,'training_health_verified':bool(truth is not None or meta[mid].get('truth_kind')=='healthy_control'),'path':str(p.resolve()),'source_hashes':src,'counts':{k:int(len(v)) for k,v in ix.items()},'x_sha256':array_sha256(x)})
  del x,ss
  if number==1 or number%10==0 or number==70: print(f'cached {number}/70 {mid}',flush=True)
 prov={str(Path(p).resolve()):sha256(p) for p in (AUDIT,MODEL,fix.PROTOCOL,old.PROTOCOL,Path(raw.__file__),Path(cohort.__file__),Path(fix.__file__),Path(__file__))}
 manifest={'schema':'healthy_only_suite_cache_v1','devices':devices,'config':{'audit':str(AUDIT.resolve()),'feature_indices':FEATURE_INDICES.tolist(),'transform':'per-stream median/max(1.4826*MAD,std), abs(asinh), max streams','source_only_existing_npz':True,'memory_strategy':'one machine at a time','provenance_sha256':prov}}
 manifest['files']={d['path']:sha256(d['path']) for d in devices}; (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return manifest
def validate(m):
 with np.load(MODEL,allow_pickle=False) as z:model=trainer.TinyAE(*(z[k].copy() for k in ('w1','b1','w2','b2')))
 protocol=fix.read_json(fix.PROTOCOL); arm=next(a for a in protocol['arms'] if a['name']=='physical_point_quantile_3of5'); totals={'fixed':[0,0,0,0],'zero':[0,0,0,0]}
 for d in m['devices']:
  with np.load(d['path'],allow_pickle=False) as z:
   p=type('P',(),{'machine_id':d['machine_id'],'dataset':d['dataset'],'truth_kind':d['truth_kind'],'truth':d['truth'],**{k:np.asarray(z[k],dtype=np.int64) for k in FIELDS}})(); x=np.asarray(z['x'],dtype=np.float64)
   for key,s in (('fixed',model.scores(x)),('zero',np.mean(x*x,axis=1))):
    r=fix.evaluate_scores(s,p,arm,protocol); totals[key][0]+=int(r['machine_success']); totals[key][1]+=int(r['fault_detected']) if r['expected']=='fault' else 0; totals[key][2]+=int(r['healthy_persistent_alarm']) if r['scored'] else 0; totals[key][3]+=int(r['healthy_persistent_alarm']) if r['expected']=='healthy_control' else 0
 if totals['fixed']!=[24,56,39,1] or totals['zero']!=[26,56,37,1]: raise RuntimeError(f'parity mismatch {totals}')
 print(json.dumps({'fixed_corrected_model':totals['fixed'],'zero_reference':totals['zero']}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=OUTPUT); a=ap.parse_args(); validate(export(a.output))
