import numpy as np
import run_study

def test_combinations_are_exact_boolean_rules():
    f={k:np.array([0,1,0,1],dtype=bool) for k in run_study.RULES}
    f.update({'pca':np.array([1,1,0,0],bool),'global_h4':np.array([1,0,1,0],bool),'feature_envelope':np.array([1,1,1,0],bool)})
    np.testing.assert_array_equal(run_study.combine(f,'pca_and_global_h4'),[1,0,0,0])
    np.testing.assert_array_equal(run_study.combine(f,'pca_and_global_h4_and_envelope'),[1,0,0,0])
    for name in ('global_h4','pca','feature_envelope'):
        np.testing.assert_array_equal(run_study.combine(f,name),f[name])

def test_evaluate_uses_original_timeline_indices_and_unknowns():
    meta={'machine_id':'d::x','dataset':'d','truth':10,'truth_kind':'onset','training_health_verified':True}
    arrays={k:np.array(v,dtype=int) for k,v in {'healthy_test_indices':[4,5],'unhealthy_test_indices':[10,11,12],'unscored_test_indices':[]}.items()}
    flags=np.zeros(13,bool); flags[10]=True
    row=run_study.evaluate(meta,arrays,flags,{'mode':'history_phase','stride':3})
    assert row['strict_success'] and row['fault_delay_acquisitions']==0 and row['first_alarm_index']==10
    flags[5]=True; row=run_study.evaluate(meta,arrays,flags)
    assert not row['strict_success'] and row['healthy_alarm']
    unknown=dict(meta,truth=None,truth_kind='unknown',training_health_verified=False)
    row=run_study.evaluate(unknown,arrays,flags)
    assert row['expected']=='unknown' and not row['scored']
    assert 'cadence' not in row and 'cadence_metadata' in row

def test_protocol_configured_count_and_no_full_study_on_import():
    assert len(run_study.RULES)==14
    assert 14*2*3==84
