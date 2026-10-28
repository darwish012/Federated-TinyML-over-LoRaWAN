"""Synthetic validation only. D: data access is forbidden for this process."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.drive.lower()=='d:':
            raise AssertionError(f'Forbidden target access during synthetic tests: {path}')
sys.addaudithook(guard)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

r=load('runner_under_test',HERE/'run_comparators.py')
published=load('published_under_test',HERE/'frozen/pipeline/research/published_methods_20261021/run_comparison.py')
import numpy as np
from threadpoolctl import threadpool_limits

tests=[]
def passed(name):
    tests.append(name)
    print('PASS:',name,flush=True)

def expect_stop(fn,needle):
    try:
        fn()
    except r.StopRun as exc:
        assert needle in str(exc),(needle,str(exc))
    else:
        raise AssertionError('Expected StopRun: '+needle)

def run():
    r.EXTERNAL=HERE/'synthetic_validation'
    (r.EXTERNAL/'results_v3/scores').mkdir(parents=True,exist_ok=True)
    seed=17070
    rng=np.random.default_rng(97)
    unit={'unit':'synthetic','dataset':'MathWorks_HSS','turbine':None,'event':None,'primary':True,
          'evaluation_status':'evaluable','control':False,'x':rng.normal(size=(50,22)),
          'train_indices':np.arange(15),'calibration_indices':np.arange(15,20),
          'healthy_test_indices':np.arange(20,44),'unhealthy_test_indices':np.arange(44,50),
          'coordinate':np.arange(1,51),'source_rows':np.arange(1,51),'onset_years':None,
          'historical_onset_sensitivity':{'onset_index':None,'analysis_status':'insufficient_post_commission_data'}}
    for day,strict in ((None,False),(21,False),(44,False),(45,True),(50,True)):
        flags=np.zeros(50,bool)
        if day: flags[day-1]=True
        out=r.outcome(unit,seed,'test',flags,np)
        assert out['strict_success']==strict and out['first_alarm_day']==day
        assert out['hss_proxy_strict_success'] is None
    flags=np.zeros(50,bool); flags[[43,44]]=True
    assert not r.outcome(unit,seed,'test',flags,np)['strict_success']
    passed('HSS yardstick boundary days 21,44,45,50; premature alarm remains failure; no-onset is null')

    s=np.ones(50); s[20:]=100
    flags=published.cadence.dense_flags(s,unit['train_indices'],unit['calibration_indices'])
    hits=np.flatnonzero(flags)
    assert hits[0]==32
    assert not flags[:20].any()
    passed('HSS short-history calibration fallback and causal 13-of-21 first alarm at record-day 33')

    s=rng.lognormal(size=50)
    assert r.PRIMARY_METHODS==('proposed_fedadam_h12','wang_3sigma','gev_regime2',
                               'javanmardi_2means','fedavg_h12','fedprox_h12')
    assert r.ALL_METHODS==r.PRIMARY_METHODS+('wang_4sigma','wang_6sigma','gev_no_clustering')
    original_point_flags=published.point_flags
    pointwise_calls=[]
    def pointwise_only(scores,threshold,a,persist=False):
        assert persist is False,'Published comparator persistence must never be computed'
        pointwise_calls.append(True)
        return original_point_flags(scores,threshold,a,persist=False)
    published.point_flags=pointwise_only
    try:
        out=r.decision_rules(published,s,unit,seed,np)
    finally:
        published.point_flags=original_point_flags
    assert len(pointwise_calls)==5
    assert set(out)==set(r.ALL_METHODS)-{'proposed_fedadam_h12','fedavg_h12','fedprox_h12'}
    for sigma in (3,4,6):
        name=f'wang_{sigma}sigma'
        expected,fit=published.wang(s,unit,sigma,persist=False)
        assert np.array_equal(out[name][0],expected) and out[name][1]==fit
    flipped=s.copy(); flipped[20:]*=1e6
    assert published.wang(s,unit,3)[1]==published.wang(flipped,unit,3)[1]
    for regimes in (1,2):
        a,detail=published.gev(s,unit,regimes,seed)
        b,detail2=published.gev(flipped,unit,regimes,seed)
        assert detail==detail2 and np.array_equal(a,b)
        name='gev_regime2' if regimes==2 else 'gev_no_clustering'
        assert np.array_equal(out[name][0],published.point_flags(s,a,unit,persist=False))
        assert out[name][1]==detail
    expected,fit=published.onset(s,unit,seed)
    assert np.array_equal(out['javanmardi_2means'][0],expected) and out['javanmardi_2means'][1]==fit
    assert all(not flags[:20].any() for flags,_ in out.values())
    passed('Exact six-primary/three-secondary roster; C1 rules match frozen pointwise/onset functions, never enable persistence; Wang/GEV fits ignore held-out scores')

    units=[]
    for turbine in (1,2,3,4,5,6):
        u=dict(unit,unit=f'SND::Turbine{turbine}'+('::event1' if turbine==5 else ''),dataset='SND',
               turbine=turbine,event='event1' if turbine==5 else None,control=turbine!=5,
               healthy_test_indices=np.arange(20,40),unhealthy_test_indices=np.arange(40,50) if turbine==5 else np.array([],int),
               coordinate=np.linspace(.1,1.19,50),replacement_years=1.2 if turbine==5 else None,
               onset_years=1.0 if turbine==5 else None)
        units.append(u)
    units.append(dict(units[4],unit='SND::Turbine5::event2',event='event2',primary=False,
                      coordinate=np.linspace(1.21,1.99,50),replacement_years=2.,onset_years=1.8))
    units.append(dict(unit,unit='MathWorks_HSS::bearing'))
    score_by_unit={}
    refs=[]
    expected={}
    for u in units:
        scores=np.ones(50)
        if u['event']=='event2' or u['dataset']=='MathWorks_HSS': scores[32:]=100
        flags=published.cadence.dense_flags(scores,u['train_indices'],u['calibration_indices'])
        score_by_unit[u['unit']]=scores
        file=r.EXTERNAL/'results_v3/scores'/(u['unit'].replace('::','__')+'__'+r.ARM+f'_seed{seed}.csv')
        records=[]
        for i in range(50):
            part='training' if i<15 else 'calibration' if i<20 else 'test'
            if u['dataset']=='SND' and i>=20:
                part='healthy_test' if i in u['healthy_test_indices'] else 'fault_test' if i in u['unhealthy_test_indices'] else 'test'
            records.append(dict(index_0based=i,source_row_or_day_1based=int(u['source_rows'][i]),
                                coordinate=float(u['coordinate'][i]),partition=part,score=float(scores[i]),alarm=bool(flags[i])))
        with file.open('w',newline='',encoding='utf-8') as stream:
            w=csv.DictWriter(stream,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        row=r.outcome(u,seed,'proposed_fedadam_h12',flags,np)
        hist=np.concatenate((u['train_indices'],u['calibration_indices']))
        statistic=published.cadence._rolling(scores[hist],21,13)
        finite=statistic[np.isfinite(statistic)]
        reference=float(np.max(finite)) if len(finite) else float(np.quantile(scores[u['calibration_indices']],.995))
        row.update(reference=reference,threshold=2*reference,
                   threshold_method='history_max_persistent' if len(finite) else 'calibration_point_quantile_0.995')
        ref=dict(row)
        if u['dataset']=='MathWorks_HSS':
            for key in ('healthy_alarm','fault_detected','strict_success'): ref[key]=None
            ref['healthy_test_rows']=0;ref['fault_test_rows']=0
        refs.append(ref)
        if u['event']=='event2':
            expected['event2_first_alarm_years']=row['first_alarm_coordinate']
            expected['event2_lead_days']=row['lead_time_days']
        if u['dataset']=='MathWorks_HSS':expected['hss_first_alarm_day']=row['first_alarm_day']
    fields=list(dict.fromkeys(k for row in refs for k in row))
    with (r.EXTERNAL/'results_v3/per_unit.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(refs)
    expected.update(snd_strict_successes=5,controls_silent=5)
    class Wider:
        @staticmethod
        def score(state,mid,x):
            return score_by_unit[mid.removesuffix('::'+r.ARM)].copy()
    mods={'cadence':published.cadence,'wider_models':Wider}
    protocol={'public_v3_reference':{str(seed):expected}}
    cached,rows,evidence=r.baseline_gate(units,[seed],{seed:None},mods,np,protocol)
    assert evidence['pass'] and len(evidence['units'])==8 and len(rows)==8
    passed('End-to-end synthetic baseline gate: score CSV schema, memberships, exact scores/alarms, v3 per-unit and public reference')
    score_by_unit[units[0]['unit']][0]+=1e-12
    expect_stop(lambda:r.baseline_gate(units,[seed],{seed:None},mods,np,protocol),'reconstruction-score mismatch')
    score_by_unit[units[0]['unit']][0]-=1e-12
    passed('Any reconstruction-score discrepancy, including 1e-12, stops before comparator evaluation')
    u=units[0]
    _,flags,_=r.load_saved_scores(u,seed,np)
    original=published.cadence.dense_flags
    published.cadence.dense_flags=lambda *a,**kw:np.logical_not(original(*a,**kw))
    # Restore exact synthetic score (floating subtraction need not round-trip).
    score_by_unit[u['unit']][0]=1.
    expect_stop(lambda:r.baseline_gate(units,[seed],{seed:None},mods,np,protocol),'alarm-vector mismatch')
    published.cadence.dense_flags=original
    passed('Alarm-vector mismatch stops before comparator evaluation')

    allrows=[]
    for method in r.ALL_METHODS:
        for repeat_seed in (17070,17071,17072):
            for u in units:
                scores=score_by_unit[u['unit']]
                flags=published.cadence.dense_flags(scores,u['train_indices'],u['calibration_indices'])
                allrows.append(r.outcome(u,repeat_seed,method,flags,np))
    cells=r.aggregate(allrows,r.ALL_METHODS,(17070,17071,17072))
    assert len(cells)==27 and all(c['strict_successes']==6 for c in cells)
    reportroot=HERE/'synthetic_report'
    reportroot.mkdir(exist_ok=True)
    summary={'complete':True,'disclosure':'Synthetic fixture only','seeds':[17070,17071,17072],
             'cells':cells,'rows':allrows}
    r.write_results(reportroot,summary,allrows)
    assert {p.name for p in reportroot.iterdir()}=={'summary.json','per_unit.csv','REPORT.md'}
    assert len(json.loads((reportroot/'summary.json').read_text())['rows'])==216
    text=(reportroot/'REPORT.md').read_text()
    assert 'C3: PCA + personalized H12 | not applicable' in text and 'HSS historical-onset sensitivity' in text
    assert '_p13of21' not in text
    passed('All 216 method/seed/unit outcomes (27 cells) aggregate with the common /7 denominator and render into exactly three result files without removed variants')

    process=subprocess.run([sys.executable,'-I','-B',str(HERE/'run_comparators.py'),'--dry-run'],capture_output=True,text=True)
    assert process.returncode==0 and 'no protocol, source, score or waveform' in process.stdout
    process=subprocess.run([sys.executable,'-I','-B',str(HERE/'run_comparators.py'),'--run'],capture_output=True,text=True,cwd=HERE)
    assert process.returncode!=0 and 'only from D:' in process.stderr
    passed('Dry run parses successfully; actual analysis refuses staging before any target read')
    code = '''import importlib.util; from pathlib import Path
s=importlib.util.spec_from_file_location('r',Path.cwd()/'run_comparators.py'); r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
r.install_write_boundary(Path.cwd()/'synthetic_boundary')
try: (Path.cwd()/'forbidden_test_write.txt').write_text('forbidden')
except PermissionError: print('BLOCKED')
else: raise AssertionError('outside write permitted')
'''
    process=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True,cwd=HERE)
    assert process.returncode==0 and 'BLOCKED' in process.stdout,process.stderr
    assert not (HERE/'forbidden_test_write.txt').exists()
    passed('Runtime audit hook blocks writes outside the three registered result files')
    result={'status':'PASS','synthetic_only':True,'target_numeric_data_read':False,'checks':tests}
    (HERE/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
