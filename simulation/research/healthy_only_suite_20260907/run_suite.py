"""Fixed healthy-only development suite. No downloads or historical writes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import thesis_participating_node_scaling as trainer
import models
import calibration

OUT = ROOT / 'results/healthy_only_suite_20260907'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows):
    scored = [r for r in rows if r['scored']]
    faults = [r for r in rows if r['expected'] == 'fault']
    controls = [r for r in rows if r['expected'] == 'healthy_control']
    delays = [r['fault_delay_acquisitions'] for r in faults if r['fault_detected']]
    return dict(strict_successes=sum(r['strict_success'] for r in scored), scored=len(scored),
                strict_percent=100 * sum(r['strict_success'] for r in scored) / max(1, len(scored)),
                faults_detected=sum(r['fault_detected'] for r in faults), faults=len(faults),
                healthy_alarm_devices=sum(r['healthy_alarm'] for r in scored),
                control_alarms=sum(r['healthy_alarm'] for r in controls), controls=len(controls),
                fault_delay_median=float(np.median(delays)) if delays else None)


def evaluate(meta, arrays, scores, mode):
    prediction = calibration.predict(scores, arrays['train_indices'], arrays['calibration_indices'], mode)
    flags = prediction['flags']
    h = arrays['healthy_test_indices'].astype(int)
    f = arrays['unhealthy_test_indices'].astype(int)
    u = arrays['unscored_test_indices'].astype(int)
    h_alarm = bool(np.any(flags[h]))
    fault_hits = np.flatnonzero(flags[f])
    f_alarm = bool(len(fault_hits))
    expected = 'fault' if meta['truth'] is not None else ('healthy_control' if meta['truth_kind'] == 'healthy_control' else 'unknown')
    first = np.flatnonzero(flags)
    return dict(machine_id=meta['machine_id'], dataset=meta['dataset'], expected=expected,
                scored=expected != 'unknown', strict_success=expected != 'unknown' and not h_alarm and (f_alarm or expected == 'healthy_control'),
                healthy_alarm=h_alarm, fault_detected=f_alarm,
                fault_delay_acquisitions=int(fault_hits[0]) if f_alarm else None,
                first_alarm_index=int(first[0]) if len(first) else None,
                healthy_test_rows=len(h), fault_test_rows=len(f), unknown_rows=len(u),
                unknown_alarm=bool(np.any(flags[u])), training_health_verified=meta['training_health_verified'],
                calibration_metadata=prediction['metadata'])


def reference_scores(name, x, train, initial):
    if name == 'untrained_ae':
        return initial.scores(x)
    if name == 'zero_reference':
        return np.mean(x*x, axis=1)
    center = np.mean(train, axis=0)
    y = x - center
    if name == 'local_centroid':
        return np.mean(y*y, axis=1)
    if name == 'local_pca':
        _, _, vh = np.linalg.svd(train - center, full_matrices=False)
        basis = vh[:min(4, len(train)-1, x.shape[1])]
        residual = y - (y @ basis.T) @ basis
        return np.mean(residual*residual, axis=1)
    raise ValueError(name)


def run():
    protocol = read(HERE / 'protocol.json')
    manifest_path = OUT / 'cache/manifest.json'
    manifest = read(manifest_path)
    assert all(sha(p) == expected for p, expected in manifest['files'].items()), 'Cache file drift'
    assert all(sha(p) == expected for p, expected in manifest['config']['provenance_sha256'].items()), 'Cache provenance drift'
    devices = sorted(manifest['devices'], key=lambda d: d['machine_id'])
    assert len(devices) == len({d['machine_id'] for d in devices}) == 70
    dependency_paths = [HERE / n for n in ('protocol.json', 'run_suite.py', 'models.py', 'calibration.py', 'data_cache.py')]
    dependency_paths += [manifest_path, ROOT / 'thesis_70_device_federated_protocol.json', ROOT / 'results/thesis_70_improvement_dev1/corrected_model.npz']
    dependency_paths += [Path(p) for p in manifest['files']]
    deps = {str(p): sha(p) for p in dependency_paths}
    out = OUT / 'trials'
    out.mkdir(parents=True, exist_ok=True)
    lockpath = OUT / 'suite_lock.json'
    if lockpath.exists():
        assert read(lockpath)['dependencies'] == deps, 'Suite dependency drift'
    else:
        write_json(lockpath, {'dependencies': deps, 'classification': protocol['classification'], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    clients = {}
    for meta in devices:
        with np.load(meta['path'], allow_pickle=False) as data:
            x = data['x']
            clients[meta['machine_id']] = x[data['train_indices'].astype(int)].copy()
    print(f"Loaded {len(clients)} clients, {sum(x.nbytes for x in clients.values()) / 1e6:.2f} MB training arrays", flush=True)
    train_spec = read(ROOT / 'thesis_70_device_federated_protocol.json')
    initial = trainer.initial_model(train_spec)
    jobs = [(name, 17070) for name in protocol['reference_models']]
    jobs += [(name, seed) for seed in protocol['train_seeds'] for name in protocol['learned_models']]
    all_results = []
    for name, seed in jobs:
        path = out / f'{name}_seed{seed}.json'
        if path.exists():
            saved = read(path)
            assert saved['dependencies'] == deps
            all_results.extend(saved['results'])
            print(f'Resumed {name} seed{seed}', flush=True)
            continue
        started = time.monotonic()
        is_learned = name in protocol['learned_models']
        state = models.fit(clients, name, seed) if is_learned else None
        if name == 'global_ae' and seed == 17070:
            with np.load(ROOT / 'results/thesis_70_improvement_dev1/corrected_model.npz', allow_pickle=False) as frozen:
                for key, arr in zip(('w1', 'b1', 'w2', 'b2'), models.global_arrays(state)):
                    np.testing.assert_allclose(arr, frozen[key], rtol=1e-11, atol=1e-12, err_msg='Global replication drift')
        grouped = {mode: [] for mode in protocol['calibration_modes']}
        train_losses = []
        for meta in devices:
            mid = meta['machine_id']
            with np.load(meta['path'], allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            x = arrays['x']
            scores = models.score(state, mid, x) if is_learned else reference_scores(name, x, clients[mid], initial)
            assert scores.shape == (len(x),) and np.all(np.isfinite(scores))
            for mode in grouped:
                grouped[mode].append(evaluate(meta, arrays, scores, mode))
            if is_learned:
                train_losses.append(float(np.mean(models.reconstruction_scores(state, mid, clients[mid]))))
        results = []
        for mode, rows in grouped.items():
            metrics = summarize(rows)
            assert (metrics['scored'], metrics['faults'], metrics['controls']) == (65, 59, 6)
            if seed == 17070 and mode == 'point995':
                if name == 'global_ae':
                    assert [metrics[k] for k in ('strict_successes','faults_detected','healthy_alarm_devices','control_alarms')] == [24,56,39,1]
                if name == 'zero_reference':
                    assert metrics['strict_successes'] == 26
            result = {'model': name, 'seed': seed, 'calibration': mode, 'metrics': metrics,
                      'per_dataset': {d: summarize([r for r in rows if r['dataset'] == d]) for d in sorted({r['dataset'] for r in rows})},
                      'mean_client_training_reconstruction_mse': float(np.mean(train_losses)) if train_losses else None,
                      'communication': models.communication(state) if is_learned else {'training': 'No federated neural training'}, 'devices': rows}
            results.append(result)
            print(f"{name} seed{seed} {mode}: strict {metrics['strict_successes']}/65, detect {metrics['faults_detected']}/59, healthy alarms {metrics['healthy_alarm_devices']}, controls {metrics['control_alarms']}/6", flush=True)
        if is_learned:
            with (out / f'{name}_seed{seed}.pkl').open('wb') as handle:
                pickle.dump(state, handle, protocol=5)
        write_json(path, {'dependencies': deps, 'elapsed_seconds': time.monotonic()-started, 'results': results})
        all_results.extend(results)
    assert len(all_results) == 66
    refs = {mode: max(r['metrics']['strict_successes'] for r in all_results if r['model'] in protocol['reference_models'] and r['calibration'] == mode) for mode in protocol['calibration_modes']}
    advancement = []
    for name in protocol['learned_models']:
        for mode in protocol['calibration_modes']:
            entries = [r for r in all_results if r['model'] == name and r['calibration'] == mode]
            passes = all(r['metrics']['strict_successes'] > max(26, refs[mode]) and r['metrics']['faults_detected'] >= 56 and r['metrics']['control_alarms'] <= 1 for r in entries)
            advancement.append({'model': name, 'calibration': mode, 'passes': passes, 'reference_strict': refs[mode], 'strict_range': [min(r['metrics']['strict_successes'] for r in entries),max(r['metrics']['strict_successes'] for r in entries)]})
    assert deps == {str(p): sha(p) for p in dependency_paths}, 'End-of-run dependency drift'
    summary = {'complete': True, 'classification': protocol['classification'], 'configuration_count': len(all_results),
               'dependencies': deps, 'advancement': advancement,
               'results': [{k:v for k,v in r.items() if k != 'devices'} for r in all_results]}
    write_json(OUT / 'summary.json', summary)
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        flat = [dict(model=r['model'], seed=r['seed'], calibration=r['calibration'], **r['metrics']) for r in all_results]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    lines = ['# Healthy-only improvement suite', '', protocol['classification'] + '.', '',
             'All 66 fixed model/seed/calibration configurations completed. Each predicts all 70 clients: 65 scored, 59 fault lifecycles, 6 controls, 5 unknown.', '',
             '| Model | Calibration | Strict successes across seeds | Advancement gate |', '|---|---|---|---|']
    for item in advancement:
        lo, hi = item['strict_range']
        lines.append(f"| {item['model']} | {item['calibration']} | {lo}–{hi}/65 | {'PASS' if item['passes'] else 'FAIL'} |")
    lines += ['', '## Interpretation limits', '',
              '- Model fitting uses only commissioning-training rows. Five unknown clients have assumed, not verified, healthy commissioning.',
              '- Fault labels define retrospective splits and evaluation, not training gradients. Previously inspected outcomes make this development, not untouched validation.',
              '- Generic feature-space synthetic negatives are not physically validated bearing waveforms.',
              '- Calibration guards and online adaptation have no guaranteed lifetime false-alarm bound.',
              '- Payload counts are analytical float32 parameter estimates, not an RF or wire-codec test.',
              '- V108 38/42 is a different detector, training regime and denominator; it is not contradicted or replicated here.',
              '- Full node-scaling results were not replaced by this 70-client endpoint study.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('COMPLETE: ' + str(OUT / 'summary.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
