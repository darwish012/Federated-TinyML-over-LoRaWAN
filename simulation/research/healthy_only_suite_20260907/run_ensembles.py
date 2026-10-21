"""Fixed causal combination extension. Reuses trained models; never tunes them."""
from pathlib import Path
import json
import pickle
import numpy as np
import models
import calibration
import run_suite as suite

HERE = Path(__file__).resolve().parent
OUT = suite.OUT / 'ensembles'


def combine(flags, rule):
    ae, svdd, pca = [np.asarray(flags[k], dtype=bool) for k in ('ae','svdd','pca')]
    assert ae.ndim == 1 and ae.shape == svdd.shape == pca.shape
    if rule == 'pca_ae_and': return pca & ae
    if rule == 'pca_svdd_and': return pca & svdd
    if rule == 'ae_svdd_and': return ae & svdd
    if rule == 'two_of_three': return ae.astype(int)+svdd.astype(int)+pca.astype(int) >= 2
    if rule == 'all_three': return ae & svdd & pca
    if rule == 'any_of_three': return ae | svdd | pca
    if rule == 'pca_ae_corroboration5':
        recent = lambda a: np.convolve(a.astype(int),np.ones(5,dtype=int),'full')[:len(a)] > 0
        return (ae | pca) & recent(ae) & recent(pca)
    raise ValueError(rule)


def evaluate(meta, arrays, flags):
    h, f, u = [arrays[k].astype(int) for k in ('healthy_test_indices','unhealthy_test_indices','unscored_test_indices')]
    ha = bool(np.any(flags[h])); fh = np.flatnonzero(flags[f]); first = np.flatnonzero(flags)
    expected = 'fault' if meta['truth'] is not None else ('healthy_control' if meta['truth_kind']=='healthy_control' else 'unknown')
    return dict(machine_id=meta['machine_id'], dataset=meta['dataset'], expected=expected, scored=expected!='unknown',
                strict_success=expected!='unknown' and not ha and (bool(len(fh)) or expected=='healthy_control'),
                healthy_alarm=ha, fault_detected=bool(len(fh)), fault_delay_acquisitions=int(fh[0]) if len(fh) else None,
                first_alarm_index=int(first[0]) if len(first) else None, unknown_alarm=bool(np.any(flags[u])),
                training_health_verified=meta['training_health_verified'])


def run():
    spec = suite.read(HERE/'ensemble_protocol.json')
    original = suite.read(suite.OUT/'summary.json')
    assert original['complete']
    assert all(suite.sha(p)==v for p,v in original['dependencies'].items()), 'Original suite drift'
    manifest = suite.read(suite.OUT/'cache/manifest.json')
    paths = [HERE/'ensemble_protocol.json',Path(__file__),suite.OUT/'summary.json']
    paths += [suite.OUT/f'trials/{name}_seed{seed}.pkl' for seed in spec['seeds'] for name in ('global_ae','ae_svdd')]
    deps={str(p):suite.sha(p) for p in paths}
    OUT.mkdir(exist_ok=True)
    lock=OUT/'lock.json'
    if lock.exists(): assert suite.read(lock)['dependencies']==deps
    else: suite.write_json(lock,{'dependencies':deps,'classification':spec['classification']})
    results=[]
    for seed in spec['seeds']:
        states={}
        for key, name in [('ae','global_ae'),('svdd','ae_svdd')]:
            with (suite.OUT/f'trials/{name}_seed{seed}.pkl').open('rb') as handle: states[key]=pickle.load(handle)
        rows={rule:[] for rule in spec['rules']}
        components={key:[] for key in ('ae','svdd','pca')}
        for meta in manifest['devices']:
            with np.load(meta['path'],allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
            x=arrays['x']; ti=arrays['train_indices'].astype(int); ci=arrays['calibration_indices'].astype(int)
            scores={key:models.score(state,meta['machine_id'],x) for key,state in states.items()}
            scores['pca']=suite.reference_scores('local_pca',x,x[ti],None)
            flags={key:calibration.predict(s,ti,ci,'point995')['flags'] for key,s in scores.items()}
            for key in components: components[key].append(evaluate(meta,arrays,flags[key]))
            for rule in spec['rules']: rows[rule].append(evaluate(meta,arrays,combine(flags,rule)))
        for key,name in [('ae','global_ae'),('svdd','ae_svdd'),('pca','local_pca')]:
            reference=next(r for r in original['results'] if r['model']==name and r['calibration']=='point995' and r['seed']==(17070 if key=='pca' else seed))
            assert suite.summarize(components[key])==reference['metrics'], 'Component parity mismatch'
        for rule, rr in rows.items():
            metrics=suite.summarize(rr)
            pca={r['machine_id']:r for r in components['pca']}
            result={'rule':rule,'seed':seed,'metrics':metrics,'per_dataset':{d:suite.summarize([r for r in rr if r['dataset']==d]) for d in sorted({r['dataset'] for r in rr})},
                    'strict_gains_vs_pca':[r['machine_id'] for r in rr if r['strict_success'] and not pca[r['machine_id']]['strict_success']],
                    'strict_losses_vs_pca':[r['machine_id'] for r in rr if not r['strict_success'] and pca[r['machine_id']]['strict_success']], 'devices':rr}
            results.append(result)
            print(f"{rule} seed{seed}: {metrics['strict_successes']}/65 strict, {metrics['faults_detected']}/59 detected, {metrics['healthy_alarm_devices']} healthy alarms, {metrics['control_alarms']}/6 controls",flush=True)
    advancement=[]
    for rule in spec['rules']:
        rr=[r for r in results if r['rule']==rule]
        advancement.append({'rule':rule,'passes':all(r['metrics']['strict_successes']>30 and r['metrics']['faults_detected']>=58 and r['metrics']['control_alarms']==0 for r in rr)})
    assert deps=={str(p):suite.sha(p) for p in paths}
    assert all(suite.sha(p)==v for p,v in original['dependencies'].items()), 'End original suite drift'
    suite.write_json(OUT/'summary.json',{'complete':True,'dependencies':deps,'configuration_count':21,'advancement':advancement,'results':results})
    print('COMPLETE '+str(OUT/'summary.json'),flush=True)


if __name__=='__main__': run()
