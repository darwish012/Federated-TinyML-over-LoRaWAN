"""Stage-1 development-only reproduction gate; never executes a source runner."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

# Enforce the stage-1 boundary for imports, provenance checks, and training.
def boundary(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and not p.is_relative_to(HERE):
        raise PermissionError(f'Write outside staging: {p}')
    if p.drive.lower() == 'd:':
        frozen = Path(r'D:\external_unseen\frozen').resolve()
        if not p.is_relative_to(frozen):
            raise PermissionError(f'Stage-1 target read denied: {p}')

sys.addaudithook(boundary)

import hashlib
import importlib.util
import json
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(r'E:\masters\autoencoder_cwru_release\full_pipeline_simulation')
RESEARCH = ROOT / 'research'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    start = time.time()
    sys.path.insert(0, str(RESEARCH / 'fl_algorithm_comparison_20260929'))
    study = load('fl_study', RESEARCH / 'fl_algorithm_comparison_20260929/run_study.py')
    manifest_path = study.CACHE / 'manifest.json'
    manifest = study.read(manifest_path)
    devices = study.noldm.select_devices(manifest)
    study.noldm.verify_selected_cache_integrity(manifest, devices)
    arrays, clients = {}, {}
    for meta in devices:
        with np.load(meta['path'], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        mid = meta['machine_id']
        arrays[mid] = a
        clients[mid] = a['x'][a['train_indices'].astype(int)]
    assert len(clients) == 54
    dependencies = {str(p.resolve()): sha(p) for p in study.noldm._selected_paths(manifest, devices)}
    for name in ('run_study.py', 'fl.py', 'protocol.json'):
        p = RESEARCH / 'fl_algorithm_comparison_20260929' / name
        dependencies[str(p)] = sha(p)
    out = HERE / 'frozen/models'
    out.mkdir(parents=True, exist_ok=True)
    evidence = {'status': 'running', 'training_clients': 54, 'scored_units': 49,
                'seeds': [17070,17071,17072], 'input_dependencies': dependencies,
                'fedadam_gate': [], 'comparators': [],
                'target_numeric_data_read': False}
    evidence_path = HERE / 'development_validation.json'
    for seed, expected in zip(evidence['seeds'], (45,44,44)):
        server = study.fl.Server('fedadam', study.fl.init([22,12,22], seed))
        for _ in range(12):
            server.round(clients)
        refpath = Path(r'D:\external_unseen\frozen\models') / f'global_h12_seed{seed}.pkl'
        with refpath.open('rb') as f:
            reference = pickle.load(f)['global']
        diff = max(float(np.max(np.abs(a-b))) for a,b in zip(server.p, reference))
        metrics, rows = study.evaluate_params(server.p, devices, arrays)
        passed = diff <= 1e-10 and metrics['strict_successes'] == expected and metrics['scored'] == 49
        evidence['fedadam_gate'].append({'seed':seed,'max_abs_weight_difference':diff,
            'strict_successes': metrics['strict_successes'], 'expected':expected,'pass':passed,
            'frozen_reference_sha256':sha(refpath)})
        print(f'FedAdam gate {seed}: {metrics["strict_successes"]}/49, maxdiff={diff:.3g}, pass={passed}', flush=True)
        write(evidence_path, evidence)
        if not passed:
            evidence['status'] = 'STOPPED_FEDADAM_GATE_DISCREPANCY'
            write(evidence_path,evidence)
            raise SystemExit(2)
    for algorithm in ('fedavg','fedprox'):
        for seed, expected in zip(evidence['seeds'], (46,44,43)):
            server = study.fl.Server(algorithm,study.fl.init([22,12,22],seed),mu=.01 if algorithm=='fedprox' else 0)
            for _ in range(12):
                server.round(clients)
            metrics, rows = study.evaluate_params(server.p,devices,arrays)
            passed = metrics['strict_successes']==expected and metrics['scored']==49
            entry = {'algorithm':algorithm,'seed':seed,'rounds':12,'mu':server.mu,
                'metrics':metrics,'expected_strict_successes':expected,'pass':passed,'devices':rows}
            evidence['comparators'].append(entry)
            print(f'{algorithm} {seed}: {metrics["strict_successes"]}/49, expect={expected}, pass={passed}',flush=True)
            if not passed:
                evidence['status']='STOPPED_C2_REPRODUCTION_DISCREPANCANCY'
                write(evidence_path,evidence)
                raise SystemExit(3)
            state = {'variant':'global_h12','seed':seed,'clients':tuple(clients),
                'global':server.p,'private':{},'centers':{},'algorithm':algorithm,'mu':server.mu,'rounds':12}
            path = out / f'{algorithm}_h12_seed{seed}.pkl'
            path.write_bytes(pickle.dumps(state,protocol=5))
            cp = {'algorithm':algorithm,'seed':seed,'rounds':12,'mu':server.mu,'clients':54,
                'layers':[22,12,22],'state_sha256':sha(path),'input_dependencies':dependencies,
                'development_metrics':metrics,'expected_strict_successes':expected,'reproduction_pass':passed}
            write(path.with_name(path.stem+'_checkpoint.json'),cp)
            entry['model_sha256']=sha(path)
            write(evidence_path,evidence)
    evidence['status']='PASS'
    evidence['runtime_seconds']=time.time()-start
    write(evidence_path,evidence)
    print(f'PASS in {time.time()-start:.1f}s; no target numeric data read.',flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
