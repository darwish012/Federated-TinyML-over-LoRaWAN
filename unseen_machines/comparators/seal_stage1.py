"""Build the stage-1 protocol and seals using allowed documents and development artifacts only."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import os
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
EXTERNAL=Path(r'D:\external_unseen')
ROOT=Path(r'E:\masters\autoencoder_cwru_release\full_pipeline_simulation')
sys.dont_write_bytecode=True

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.drive.lower()=='d:' and path not in {EXTERNAL/'protocol.json',EXTERNAL/'run_unseen.py',
                EXTERNAL/'PROTOCOL.md',EXTERNAL/'DEVIATIONS.md',EXTERNAL/'results_v3/REPORT.md'}:
            raise PermissionError('Sealing never reads target numeric data: '+str(path))
        mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not path.is_relative_to(HERE):raise PermissionError('Write outside staging')
sys.addaudithook(guard)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    evidence=read(HERE/'development_validation.json')
    assert evidence['status']=='PASS'
    assert read(HERE/'synthetic_test_results.json')['status']=='PASS'
    v3=read(EXTERNAL/'protocol.json')
    spec=importlib.util.spec_from_file_location('sealed_runner',HERE/'run_comparators.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    disclosure=("The proposed framework's unseen-machine results were already known to the author and to the addendum "
                "preparers when this addendum was written. The comparators' unseen-machine results were not known. "
                "This is a prospective specification of additional comparators after disclosure of the proposed method, "
                "not a claim that this addendum preceded the original unseen test. No target numeric data were read during stage 1.")
    primary=list(runner.PRIMARY_METHODS)
    source_pairs={
        ROOT/'research/published_methods_20261021/run_comparison.py':'frozen/pipeline/research/published_methods_20261021/run_comparison.py',
        ROOT/'research/published_methods_20261021/protocol.json':'frozen/pipeline/research/published_methods_20261021/protocol.json',
        ROOT/'research/healthy_only_causal_20260908/cadence.py':'frozen/pipeline/research/healthy_only_causal_20260908/cadence.py',
        ROOT/'research/healthy_only_causal_20260908/wider_models.py':'frozen/pipeline/research/healthy_only_causal_20260908/wider_models.py',
        ROOT/'research/fl_algorithm_comparison_20260929/fl.py':'frozen/evidence/fl.py',
        ROOT/'research/fl_algorithm_comparison_20260929/run_study.py':'frozen/evidence/run_study.py',
        ROOT/'research/fl_algorithm_comparison_20260929/protocol.json':'frozen/evidence/protocol.json',
        ROOT/'research/healthy_only_causal_no_ldm_20260922/run_study.py':'frozen/evidence/no_ldm_run_study.py',
        ROOT/'research/healthy_only_causal_no_ldm_20260922/protocol.json':'frozen/evidence/no_ldm_protocol.json',
        ROOT/'research/healthy_only_causal_20260908/run_study.py':'frozen/evidence/original_causal_run_study.py',
        Path(r'E:\masters\paper\main.tex'):'frozen/evidence/paper_main.tex',
        EXTERNAL/'protocol.json':'frozen/evidence/V3_PROTOCOL.json',
        EXTERNAL/'PROTOCOL.md':'frozen/evidence/V3_PROTOCOL.md',
        EXTERNAL/'results_v3/REPORT.md':'frozen/evidence/V3_PUBLIC_REPORT.md',
    }
    sources={}
    for path,relative in source_pairs.items():
        digest=sha(path)
        assert digest==sha(HERE/relative),(path,relative)
        sources[str(path)]={'sha256':digest,'snapshot':relative}
    sources[str(EXTERNAL/'run_unseen.py')]={'sha256':sha(EXTERNAL/'run_unseen.py'),'runtime_import':True}
    models=[]
    for alg in ('fedavg','fedprox'):
        for seed,expected in zip((17070,17071,17072),(46,44,43)):
            path=HERE/'frozen/models'/f'{alg}_h12_seed{seed}.pkl'
            cp=path.with_name(path.stem+'_checkpoint.json')
            checkpoint=read(cp)
            assert checkpoint['state_sha256']==sha(path)
            assert checkpoint['development_metrics']['strict_successes']==expected
            models.append({'algorithm':alg,'seed':seed,'model':path.relative_to(HERE).as_posix(),
                           'sha256':sha(path),'checkpoint':cp.relative_to(HERE).as_posix(),
                           'checkpoint_sha256':sha(cp),'development_strict_successes':expected,'scored_units':49})
    reference={str(seed):{'snd_strict_successes':5,'controls_silent':5,'primary_with_hss_strict_successes':6,
                        'event1_fault_detected':False,'event1_first_alarm_years':None,
                        'hss_first_alarm_day':day,'event2_first_alarm_years':year,'event2_lead_days':lead}
               for seed,day,year,lead in ((17070,49,1.523988818692,173.86308397274698),
                                        (17071,47,1.523988818692,173.86308397274698),
                                        (17072,45,1.52672542646071,172.8635379852257))}
    protocol={
        'protocol_id':'SND_HSS_COMPARATOR_ADDENDUM_20261027_STAGE1','version':1,
        'created_utc':datetime.now(timezone.utc).isoformat(),'status':'SEALED_STAGE1_COMPLETE',
        'disclosure':disclosure,'staging_root':str(HERE),'delivery_root':str(EXTERNAL/'comparators'),
        'stage2_working_directory':str(EXTERNAL/'comparators'),
        'stage2_command':r'python -I -B .\run_comparators.py --protocol .\comparators_protocol.json --run',
        'seeds':[17070,17071,17072],
        'stage1_data_boundary':{'no_SND_CSV_fields_read':True,'no_HSS_MAT_signals_read':True,
                               'no_results_scores_files_opened_or_hashed':True,'no_target_per_unit_values_read':True,
                               'development_only_training':True,'target_numeric_data_read':False},
        'methods':{'primary':primary,'all_applicable':list(runner.ALL_METHODS),'secondary': [m for m in runner.ALL_METHODS if m not in primary]},
        'C1':{'implementation':'frozen/pipeline/research/published_methods_20261021/run_comparison.py',
              'settings':read(HERE/'frozen/pipeline/research/published_methods_20261021/protocol.json'),
              'front_end':'same v3 frozen FedAdam H12 scores; x from exactly the v3 feature/scaling pipeline for regime clustering',
              'call_only':['wang','gev','point_flags','onset'],'never_call_source_runner':True,
              'all_variants':'primary: Wang 3-sigma, regime-2 GEV, two-means; secondary: pointwise Wang 4/6-sigma and unclustered GEV; no persistence-augmented variants',
              'author_reason':'Published comparators are evaluated unmodified.',
              'healthy_test_rows_never_fit_thresholds':True,'two_means_exception':'intentional full-trajectory retrospective clustering'},
        'deviations':['Amendment C-1 (2026-10-27, before any comparator result): persistence-augmented variants of published rules removed by author decision.'],
        'C2':{'algorithms':['fedavg','fedprox'],'mu':{'fedavg':0.0,'fedprox':0.01},'rounds':12,
              'architecture':[22,12,22],'parameters':562,'hidden':'tanh','output':'linear','dtype':'float64',
              'initialisation':'fl.init, one default_rng(seed), sequential normal draws; zero biases',
              'local_full_batch_steps':2,'local_learning_rate':0.03,'weight_matrix_L2':0.0001,
              'aggregation':'equal-client mean of local deltas; no FedAdam server step',
              'clients':54,'cohort':'exact selected non-LDM development cache; 43 fault units, 6 controls, 5 unscored',
              'expected_strict_counts':[46,44,43],'observed_fedavg':[46,44,43],'observed_fedprox':[46,44,43],
              'gate':'all three frozen FedAdam weights identical (max difference zero), strict counts 45/44/44; then C2 exact reproduction or stop',
              'no_unseen_weight_updates':True,'models':models,
              'unseen_scoring':'v3 local training-only physical_matrix; calibration + training H; unchanged cadence.dense_flags(21,13,2)'},
        'applicability':{
            'C1':'applicable: fixed decision rules on frozen H12 scores',
            'C2':'applicable: development reproduction gate passed exactly for both optimizers',
            'C3':{'status':'not applicable','declared_rule':'pca_and_personalized_h12','cadence':'history_phase',
                  'historical_development_strict_counts':[40,43,41],'independently_retrained_in_stage1':False,
                  'reason':'wider_models.fit keeps a client decoder across 12 federated rounds and scores development IDs with private decoders. _model falls back to the global decoder for unseen IDs. No frozen new-device decoder commissioning routine exists. Replaying federation, fitting a new decoder, choosing epochs/initialization, or using the global fallback would change the declared method. PCA and cadence alone do not resolve this. No such design was invented.',
                  'evidence':['frozen/evidence/no_ldm_run_study.py','frozen/evidence/original_causal_run_study.py',
                              'frozen/pipeline/research/healthy_only_causal_20260908/wider_models.py']},
            'C4':{'status':'optional comparator not included','reason':'Frozen loges(channels) hard-codes rfftfreq(...,1/25600) and development two-channel PRONOSTIA/XJTU raw loaders. It has no supplied sampling-rate parameter for 12800 Hz SND or 97656 Hz HSS. A new adapter/sample-rate implementation would be required; the optional arm is omitted before target access.',
                  'new_choices_introduced':False}},
        'partitions':{'arm':'A: RMS strictly >0.5 g for SND; no filter for HSS','v3_version':3,'v3_amendment':2,
                      'SND':v3['partitions'],'SND_onset':v3['onset'],
                      'event1':'T5 time <1.2; documented onset 1.0; unchanged 60/20/20 split of included pre-onset rows',
                      'event2':'separate T5 1.2<time<2.0 unit; original V19 proxy and stop/nonfatal no-onset rules, separate commissioning',
                      'controls':'full RMS-filtered record, 60/20/20; evaluate last 20%',
                      'HSS':{'training_days':[1,15],'calibration_days':[16,20], 'test_days':[21,50],
                             'all_50_record_days':True,'two_March17_records_preserved':True,'tach_loaded':False}},
        'alarm':v3['alarm'],
        'outcomes':{'primary_denominator':7,'primary_units':['Turbine1','Turbine2','Turbine3','Turbine4','Turbine6','Turbine5_event1','HSS'],
                    'SND_control_strict':'no alarm anywhere in held-out healthy test',
                    'SND_fault_strict':'no alarm anywhere in held-out healthy test AND >=1 alarm in assigned fault period',
                    'HSS_strict':'no alarm on record-days 21-44 inclusive AND >=1 alarm on record-days 45-50 inclusive',
                    'HSS_yardstick_provenance':'Author-specified paper yardstick in this request; original v3 protocol reported HSS timing without strict success. Apply identically to all methods, including proposed.',
                    'HSS_alarm_on_day50':'counts for strict detection; does not count for v3 advance warning (first alarm <50)',
                    'event2':'secondary; strict, healthy alarm, fault detected, first alarm, lead days; excluded from /7',
                    'SND_lead_days':'(nominal replacement year - first post-commissioning alarm year)*365.25; include premature alarms',
                    'HSS_lead':'50-first alarm record-day; record-days, not calendar-days','no_alarm_timing':None,
                    'controls':'silent count out of 5 for each method/seed','not_evaluable':'null outcomes, keep registered denominators; do not rescue'},
        'HSS_historical_onset_sensitivity':{'estimator':'same frozen_v19_eda_features.acquisition_features and frozen_v19_analytic_onset.estimate',
                    'original_minimum_acquisitions':105,'available_record_days':50,
                    'expected_status_from_guard':'insufficient_post_commission_data; onset null',
                    'no_guard_adjustment':True,'no_repartition_or_model_refit':True,
                    'if_onset_found':'assess fixed alarm vectors on test days 21-50: pre-onset is healthy, onset onwards fault; empty evaluation interval gives null; original commissioning retained',
                    'methods':'all applicable methods including proposed; never substitute a fixed onset for this sensitivity'},
        'public_v3_reference':reference,
        'baseline_gate':{'before_comparator_outcomes':True,'cached_scores':'results_v3/scores Arm A only',
                         'checks':['exact reconstructed score arrays','exact alarm vectors','source rows','coordinates','partition labels',
                                   'per-unit strict/healthy/fault outcomes and timing','threshold references and fallback methods',
                                   'public REPORT primary/secondary reference numbers'],
                         'difference_policy':'stop, no numeric tolerance, no tuning or rescue'},
        'source_files_sha256':sources,
        'external_files_sha256':{str(path):sha(path) for path in (EXTERNAL/'protocol.json',EXTERNAL/'run_unseen.py',EXTERNAL/'PROTOCOL.md',EXTERNAL/'results_v3/REPORT.md')},
        'v3_frozen_files_sha256':v3['frozen_files_sha256'],
        'stage1_package_versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','scikit-learn','threadpoolctl')},
        'outputs':{'only_write_directory':str(EXTERNAL/'comparators/results'),'files':['summary.json','per_unit.csv','REPORT.md'],
                   'existing_results':'refuse rerun/overwrite/resume','facts_only':True},
        'bundle_files_sha256':{}
    }
    names=['run_comparators.py','train_development.py','test_stage1.py','seal_stage1.py','DELIVER.ps1',
           'STAGE1_VALIDATION.md','development_validation.json','development_training.log','synthetic_test_results.json','synthetic_validation.log']
    names += [p.relative_to(HERE).as_posix() for p in (HERE/'frozen').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protocol['bundle_files_sha256']={name:sha(HERE/name) for name in sorted(set(names))}
    (HERE/'comparators_protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    md=['# Comparator addendum: frozen stage-1 protocol','',f'Protocol: `{protocol["protocol_id"]}`, version 1. Sealed at {protocol["created_utc"]}.','',
        disclosure,'',
        '## Scope and stage boundary','',
        'All writes in stage 1 are in the registered C: staging directory. No commit and no delivery to D: occur in stage 1. No SND CSV fields, HSS MAT signal values, results_v3/scores file contents (including hashes), or target per_unit.csv values were read. The existing scores directory was listed by filename only. Public v3 documents, frozen code/models, and the development cohort on E: were used.','',
        '## Comparators and frozen settings','',
        '**C1 applies.** Import the byte-identical saved run_comparison.py functions; never run its original runner. Use the same v3 FedAdam H12 scores, local standardized 22-feature x, and original train/calibration partitions. Report Wang 3-sigma, regime-2 GEV, and retrospective scalar-score two-means as primary comparators. Secondary variants are pointwise Wang 4/6-sigma and unclustered GEV. No persistence-augmented variants of published rules are computed or reported. Published comparators are evaluated unmodified.','',
        protocol['deviations'][0],'',
        'Wang baseline is first 100 training+calibration acquisitions when available, otherwise first min(30,H). Threshold is mean + sigma*population std, strict >. GEV clusters commissioning x with KMeans(n_init=10, random_state=seed), uses chronological block maxima (size 10, final partial block included), SciPy GEV MLE and 0.99 return level. Fewer than three maxima, degenerate maxima or invalid fit use the existing 0.995 empirical quantile fallback. Save fits, maxima, warnings and fallbacks. Two-means clusters the full scalar score trajectory, labels acquisition 0 healthy, and holds an alarm from the first changed label at zero-based index >=10, masked through calibration. Its retrospective nature is retained. For the proposed method and FedAvg/FedProx only, persistence needs a full 21 rows, 13 exceedances, and carries across commissioning/evaluation boundaries.','',
        '**C2 applies.** Train both optimizers afresh on exactly 54 non-LDM development clients for 12 rounds, seeds 17070/17071/17072, architecture 22-12-22 tanh/linear, float64. Use the original initialization, two full-batch local steps at 0.03 with weight L2=1e-4, equal-client aggregation; FedProx mu=0.01. No target data trains weights. The FedAdam replication gate passed with zero maximum absolute weight difference and 45/44/44 strict successes. FedAvg and FedProx each reproduced 46/44/43 on the same 49 scored development units. Stop on any discrepancy rather than proceeding. Apply C2 with unchanged v3 local scaling and causal dense 13-of-21, alpha=2.','',
        '**C3: not applicable.** The declared pca_and_personalized_h12/history_phase method has historical development counts 40/43/41. Its private decoder is trained per participating client across federated rounds. The frozen code has no procedure to commission a new private decoder; unknown IDs fall back to the global decoder. Initialization, training schedule or federation replay for new units would be new choices. No C3 outcomes are computed and no global fallback is relabeled personalized.','',
        '**C4: optional arm not included.** The existing logES function hard-codes 25600 Hz and its loaders select two-channel PRONOSTIA/XJTU data. Applying it at 12800 Hz SND or 97656 Hz HSS would require a sampling-rate/front-end adapter change. No new choice or waveform access was made for this optional arm.','',
        '## Units, partitions and alarm rules','',
        'Use v3 Amendment 2 Arm A only, authors\' raw RMS >0.5 g inclusion for SND. Preserve identity assumptions, chronology, nominal replacement cutoffs, event-1 documented onset 1.0 year, event-2 historical onset, training-only feature normalization, and the original 60/20/20 SND partitions and guards. Event 2 is an independent processing unit, not an independent machine. No event-1 rows at/after 1.2; event 2 strictly 1.2<time<2.0. Five controls use their complete filtered record with only the final ~20% evaluated.','',
        'HSS uses all 50 ordered record-days including both March 17 records, training days 1-15, calibration 16-20 and testing 21-50. Load vibration only, never tach. For the project rule, 20 commissioning scores cannot fill a 21-score window: theta = 2*q0.995(calibration scores). Persistence continues through commissioning and all later boundaries with no latch/reset.','',
        '## Outcomes and common yardstick','',
        'Primary: strict successes out of 7 machines per seed/method: Turbines 1-4 and 6, T5 event 1, HSS. Controls succeed iff silent across all held-out testing. T5 event 1 succeeds iff silent in its healthy test and at least one fault-period alarm. HSS succeeds iff no alarm on record-days 21-44 inclusive AND at least one alarm on record-days 45-50 inclusive. The author specifies this as the paper yardstick; original v3 itself reported HSS timing without a strict score. Apply the addendum yardstick identically to every method. Alarm on day 50 counts for this strict yardstick but fails the original advance-warning criterion (<50).','',
        'Secondary for every method: event-2 healthy alarm/detection/strict success and lead days, controls silent /5, HSS first-alarm day and lead in record-days. SND lead=(replacement year-first post-commissioning alarm year)*365.25, including premature alarms. No alarm means null timing. Event 2 stays outside /7; not-evaluable outcomes remain null without denominator reduction.','',
        'HSS onset sensitivity invokes the unchanged historical V19 estimator on its waveform-derived descriptors, independent of detector scores. Its original minimum is 100+5=105 acquisitions, so 50 HSS records produce insufficient_post_commission_data and no onset. Retain null sensitivity outcomes for every method; never shorten the guard. If an onset is returned, assess the same fixed alarm vectors on days 21-50 before/from onset, without refitting commissioning/models.','',
        '## Mandatory proposed-method reference gate','',
        'Before computing comparator outcomes, reconstruct proposed scores and flags and require exact equality to the original v3 Arm A score CSVs; check source rows, coordinate, partition labels, per-unit outcomes, timing and thresholds/fallbacks. Verify the public reference below. Any mismatch stops the run; no tolerance, tuning, rescue or automatic rerun.','',
        '| Seed | SND strict /6 | Addendum strict /7 | Controls silent /5 | HSS first alarm | T5 event-2 first alarm (y) | Event-2 lead days |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for seed in protocol['seeds']:
        ref=reference[str(seed)]
        md.append(f'| {seed} | 5 | 6 | 5 | {ref["hss_first_alarm_day"]} | {ref["event2_first_alarm_years"]} | {ref["event2_lead_days"]} |')
    md+=['','Event 1 was missed for all three seeds with no post-commissioning alarm. Event 2 had no healthy-test alarm and was detected for all three Arm A seeds. These are known public reference values, not newly opened comparator results.','',
         '## Source/model integrity and execution','',
         'The JSON contains exact source-file hashes with byte-identical snapshot locations, the v3 frozen dependency hash map, all model/checkpoint hashes, package versions, complete settings, and the bundle artifact hash map. Development cache/provenance SHA-256 values and per-unit reproduction outcomes are retained in development_validation.json and every model checkpoint. The two protocol files are sealed by COMPARATORS_SHA256.txt. The runner verifies the seal and every bundle artifact, then verifies the original v3 protocol/source/model seals before target processing.','',
         '| C2 model | SHA-256 |','|---|---|']
    md += [f'| {m["model"]} | `{m["sha256"]}` |' for m in models]
    md+=['','Only Opus delivers with `DELIVER.ps1`; its `-VerifyOnly` mode reads staging only. Delivery refuses an existing destination, copies only sealed artifacts, checks resolved paths stay within the fixed roots and verifies every copied hash. It never copies data or results.','',
         'Stage 2 refuses analysis anywhere except `D:\\external_unseen\\comparators` as both script directory and working directory. It reads original data/frozen/results files read-only and writes only `results/summary.json`, `results/per_unit.csv`, and `results/REPORT.md`. An existing results directory causes refusal. Runtime write guards block other writes and subprocesses. Failures produce an incomplete factual report, not a completed comparator table.','',
         '```powershell',r'Set-Location -LiteralPath D:\external_unseen\comparators',protocol['stage2_command'],'```','',
         'The report retains all failures, nulls, applicability limits, GEV diagnostics and the retrospective two-means caveat. No promotional language, outcome-driven selection, target tuning, additional operating arms or significance tests are registered. Seed repetitions, one-farm dependence, assumed identity, approximate replacements and retrospective onset labels remain limitations.']
    (HERE/'COMPARATORS_PROTOCOL.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (HERE/'COMPARATORS_SHA256.txt').write_text(''.join(f'{sha(HERE/name)}  {name}\n' for name in ('COMPARATORS_PROTOCOL.md','comparators_protocol.json')),encoding='utf-8')
    print('Sealed protocol and',len(protocol['bundle_files_sha256']),'bundle artifacts; no target numeric data read.')

if __name__=='__main__':main()
