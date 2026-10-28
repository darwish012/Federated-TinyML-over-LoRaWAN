"""Sealed comparator addendum. Analysis runs only after delivery to its fixed D: directory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

EXTERNAL = Path(r'D:\external_unseen')
DESTINATION = EXTERNAL / 'comparators'
ARM = 'authors_rms_gt_0p5g'
PRIMARY_METHODS = ('proposed_fedadam_h12', 'wang_3sigma',
                   'gev_regime2', 'javanmardi_2means', 'fedavg_h12', 'fedprox_h12')
ALL_METHODS = PRIMARY_METHODS + ('wang_4sigma', 'wang_6sigma', 'gev_no_clustering')


class StopRun(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise StopRun(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).resolve().parent / 'comparators_protocol.json')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true')
    action.add_argument('--dry-run', action='store_true', help='Parse arguments only; no files or target data opened')
    action.add_argument('--verify-only', action='store_true', help='Verify the local bundle only; no target data opened')
    return parser.parse_args(argv)


def verify_bundle(root, protocol_path):
    require(protocol_path.resolve() == root / 'comparators_protocol.json', 'Protocol must be the adjacent sealed JSON')
    pairs = []
    for line in (root / 'COMPARATORS_SHA256.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, filename = line.split(None, 1)
            pairs.append((filename.lstrip(' *'), digest))
    require(len(pairs) == 2 and {p[0] for p in pairs} ==
            {'COMPARATORS_PROTOCOL.md', 'comparators_protocol.json'}, 'Seal must name exactly both protocol files')
    for filename, digest in pairs:
        require(sha(root / filename) == digest, f'Protocol seal broken: {filename}')
    protocol = read_json(protocol_path)
    require(protocol['status'] == 'SEALED_STAGE1_COMPLETE', 'Stage 1 is incomplete')
    require(protocol['seeds'] == [17070, 17071, 17072], 'Seed drift')
    require(protocol['methods']['all_applicable'] == list(ALL_METHODS), 'Method roster drift')
    require(protocol['methods']['primary'] == list(PRIMARY_METHODS), 'Primary roster drift')
    for relative, digest in protocol['bundle_files_sha256'].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f'Bundle path escaped: {relative}')
        require(sha(path) == digest, f'Bundle artifact drift: {relative}')
    validation = read_json(root / 'development_validation.json')
    require(validation['status'] == 'PASS', 'Development gate not passed')
    for alg in ('fedavg', 'fedprox'):
        selected = [r for r in validation['comparators'] if r['algorithm'] == alg]
        require([r['seed'] for r in selected] == protocol['seeds'] and
                [r['metrics']['strict_successes'] for r in selected] == [46, 44, 43] and
                all(r['pass'] and r['metrics']['scored'] == 49 for r in selected), f'{alg} reproduction drift')
        for seed in protocol['seeds']:
            stem = f'{alg}_h12_seed{seed}'
            checkpoint = read_json(root / 'frozen/models' / (stem + '_checkpoint.json'))
            require(checkpoint['reproduction_pass'] and checkpoint['rounds'] == 12 and
                    checkpoint['mu'] == (0.01 if alg == 'fedprox' else 0.0) and
                    checkpoint['state_sha256'] == sha(root / 'frozen/models' / (stem + '.pkl')),
                    f'Invalid checkpoint {stem}')
    return protocol


def install_write_boundary(root):
    result_root = (root / 'results').resolve()
    allowed = {result_root / name for name in ('summary.json', 'per_unit.csv', 'REPORT.md')}

    def boundary(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and path not in allowed:
                raise PermissionError(f'Writes permitted only to three comparator result files: {path}')
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate'):
            paths = args[:2] if event == 'os.rename' else args[:1]
            for path in paths:
                if isinstance(path, (str, bytes, os.PathLike)):
                    resolved = Path(os.fsdecode(path)).resolve()
                    require(event == 'os.mkdir' and resolved == result_root,
                            f'Filesystem mutation prohibited: {event} {resolved}')
        elif event in ('subprocess.Popen', 'os.system'):
            raise PermissionError('Subprocesses are not part of the sealed runner')
    sys.addaudithook(boundary)
    return result_root


def indices(parts, np):
    return {key: np.asarray(value, dtype=np.int64) for key, value in parts.items()}


def build_units(v3, p, mods, np, pd):
    units = []
    registered = [dict(u, event='event1') if u['turbine'] == 5 else dict(u)
                  for u in p['snd']['units']]
    registered.append(dict(next(u for u in p['snd']['units'] if u['turbine'] == 5), event='event2'))
    for original in registered:
        turbine, event = original['turbine'], original.get('event')
        suffix = f'::Turbine{turbine}' + (f'::{event}' if event else '')
        unit_id = 'SND' + suffix
        print(f'Build fixed Arm A features and partitions: {unit_id}', flush=True)
        data = v3.snd_features(original, p, mods, np)
        try:
            pos, ti, ci, hi, fi, onset = v3.make_snd_partitions(data, original, ARM, p, mods, np, pd)
        except v3.NoProxyOnset as exc:
            if event != 'event2':
                raise
            units.append({'unit': unit_id, 'dataset': 'SND', 'turbine': turbine, 'event': event,
                          'primary': False, 'evaluation_status': 'not evaluable',
                          'reason': str(exc), 'onset': exc.onset})
            continue
        features = data['features'][pos]
        x = mods['data_cache'].physical_matrix([features], ti)
        onset_years = None
        if onset:
            onset_years = (onset['onset_years'] if onset['method'] == 'documented'
                           else float(data['times'][pos[int(onset['onset_index'])]]))
        units.append({'unit': unit_id, 'dataset': 'SND', 'turbine': turbine, 'event': event,
                      'primary': event != 'event2', 'control': turbine != 5,
                      'evaluation_status': 'evaluable', 'x': x,
                      'train_indices': ti, 'calibration_indices': ci,
                      'healthy_test_indices': hi, 'unhealthy_test_indices': fi,
                      'coordinate': data['times'][pos], 'source_rows': data['csv_rows'][pos],
                      'onset': onset, 'onset_years': onset_years,
                      'replacement_years': (2.0 if event == 'event2' else 1.2) if turbine == 5 else None})
    from scipy.io import loadmat, whosmat
    feature_rows, descriptors = [], []
    for item in p['hss']['records']:
        path = Path(item['path'])
        header = {name: shape for name, shape, _ in whosmat(path)}
        require(header.get('vibration') == tuple(item['vibration_shape']), f'HSS MAT schema drift: {path.name}')
        signal = np.asarray(loadmat(path, variable_names=['vibration'])['vibration'], dtype=np.float64)
        require(signal.ndim == 2 and min(signal.shape) == 1 and np.all(np.isfinite(signal)), 'Invalid HSS vibration')
        channel = signal.reshape(1, -1)
        feature_rows.append(mods['raw_online_federated_v12'].local_preprocess_current_acquisition(channel))
        descriptors.append(mods['frozen_v19_eda_features'].acquisition_features(channel))
    require(len(feature_rows) == 50, 'Expected all 50 HSS record-days')
    # This estimator has its original 105-acquisition guard: never rescale its constants to HSS.
    proxy = mods['frozen_v19_analytic_onset'].estimate(pd.DataFrame(descriptors))
    ti, ci = np.arange(15, dtype=np.int64), np.arange(15, 20, dtype=np.int64)
    x = mods['data_cache'].physical_matrix([np.asarray(feature_rows, dtype=np.float32)], ti)
    units.append({'unit': 'MathWorks_HSS::bearing', 'dataset': 'MathWorks_HSS', 'turbine': None,
                  'event': None, 'primary': True, 'control': False, 'evaluation_status': 'evaluable',
                  'x': x, 'train_indices': ti, 'calibration_indices': ci,
                  'healthy_test_indices': np.arange(20, 44, dtype=np.int64),
                  'unhealthy_test_indices': np.arange(44, 50, dtype=np.int64),
                  'coordinate': np.arange(1, 51), 'source_rows': np.arange(1, 51),
                  'onset': None, 'onset_years': None, 'replacement_years': None,
                  'historical_onset_sensitivity': proxy,
                  'records': [{k: item[k] for k in ('record_day_1based', 'path', 'timestamp_utc')}
                              for item in p['hss']['records']]})
    require(len(units) == 8 and sum(u['primary'] for u in units) == 7, 'Unit roster drift')
    return units


def load_saved_scores(unit, seed, np):
    path = EXTERNAL / 'results_v3/scores' / (unit['unit'].replace('::', '__') + '__' + ARM + f'_seed{seed}.csv')
    with path.open('r', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(unit['x']), f'Cached score count drift: {path.name}')
    expected_partitions = np.full(len(rows), 'test', dtype=object)
    for key, label in (('train_indices', 'training'), ('calibration_indices', 'calibration')):
        expected_partitions[unit[key]] = label
    if unit['dataset'] == 'SND':
        for key, label in (('healthy_test_indices', 'healthy_test'), ('unhealthy_test_indices', 'fault_test')):
            expected_partitions[unit[key]] = label
    require([int(r['index_0based']) for r in rows] == list(range(len(rows))), 'Cached index drift')
    require(np.array_equal([int(r['source_row_or_day_1based']) for r in rows], unit['source_rows']), 'Source membership drift')
    require(np.array_equal([float(r['coordinate']) for r in rows], unit['coordinate']), 'Coordinate drift')
    require(np.array_equal([r['partition'] for r in rows], expected_partitions), 'Partition drift')
    require(all(r['alarm'] in ('True', 'False') for r in rows), 'Invalid cached Boolean')
    scores = np.asarray([float(r['score']) for r in rows])
    require(np.all(np.isfinite(scores)) and np.all(scores >= 0), 'Invalid cached scores')
    return scores, np.asarray([r['alarm'] == 'True' for r in rows]), {'path': str(path), 'sha256': sha(path)}


def outcome(unit, seed, method, flags, np):
    base = {k: unit.get(k) for k in ('unit', 'dataset', 'turbine', 'event', 'primary', 'evaluation_status')}
    base.update(method=method, seed=seed, arm=ARM)
    if unit['evaluation_status'] != 'evaluable':
        base.update(reason=unit['reason'], strict_success=None, healthy_alarm=None, fault_detected=None,
                    first_alarm_index_0based=None, first_alarm_day=None, lead_time_days=None)
        return base
    hi, fi = unit['healthy_test_indices'], unit['unhealthy_test_indices']
    hits = np.flatnonzero(flags)
    first = int(hits[0]) if len(hits) else None
    healthy_alarm, detected = bool(np.any(flags[hi])), bool(np.any(flags[fi]))
    row_or_day = int(unit['source_rows'][first]) if first is not None else None
    coordinate = float(unit['coordinate'][first]) if first is not None else None
    base.update(strict_success=not healthy_alarm and (unit['control'] or detected),
                healthy_alarm=healthy_alarm, fault_detected=detected if not unit['control'] else None,
                healthy_alarm_acquisitions=int(np.sum(flags[hi])),
                first_alarm_index_0based=first, first_alarm_source_row_or_day_1based=row_or_day,
                first_alarm_coordinate=coordinate, training_rows=len(unit['train_indices']),
                calibration_rows=len(unit['calibration_indices']), healthy_test_rows=len(hi), fault_test_rows=len(fi),
                n_rows=len(flags), first_alarm_in_fault_period=bool(first in fi) if first is not None else False,
                alarm_indices_0based=hits.tolist(), onset_years=unit['onset_years'])
    if unit['dataset'] == 'MathWorks_HSS':
        base.update(first_alarm_day=row_or_day, precedes_failure_day50=row_or_day < 50 if row_or_day is not None else False,
                    lead_time_days=50-row_or_day if row_or_day is not None else None, lead_time_basis='record-days',
                    strict_criterion='no alarm days 21-44 AND at least one alarm days 45-50')
        proxy = unit['historical_onset_sensitivity']
        onset = proxy.get('onset_index')
        if onset is None:
            base.update(hss_proxy_status='not evaluable: ' + proxy['analysis_status'],
                        hss_proxy_strict_success=None, hss_proxy_healthy_alarm=None, hss_proxy_fault_detected=None)
        else:
            healthy = np.arange(20, min(50, onset))
            fault = np.arange(max(20, onset), 50)
            valid = len(healthy) > 0 and len(fault) > 0
            ha, fd = bool(np.any(flags[healthy])), bool(np.any(flags[fault]))
            base.update(hss_proxy_status='evaluable' if valid else 'not evaluable: empty evaluation partition',
                        hss_proxy_strict_success=not ha and fd if valid else None,
                        hss_proxy_healthy_alarm=ha if valid else None, hss_proxy_fault_detected=fd if valid else None)
    else:
        replacement = unit['replacement_years']
        lead = replacement-coordinate if replacement is not None and coordinate is not None else None
        base.update(lead_time_days=lead*365.25 if lead is not None else None,
                    lead_to_replacement_years=lead,
                    first_alarm_minus_replacement_years=-lead if lead is not None else None,
                    lead_time_basis='days (365.25 per source year)',
                    first_alarm_before_onset=coordinate < unit['onset_years'] if coordinate is not None and unit['onset_years'] is not None else None)
    return base


def compare_original_per_unit(unit, actual, reference, require_equal=True):
    keys = ['first_alarm_index_0based', 'first_alarm_source_row_or_day_1based', 'first_alarm_coordinate',
            'lead_time_days', 'n_rows', 'training_rows', 'calibration_rows']
    if 'reference' in actual:
        keys += ['reference', 'threshold', 'threshold_method']
    if unit['dataset'] == 'SND':
        keys += ['healthy_alarm', 'fault_detected', 'strict_success', 'healthy_test_rows', 'fault_test_rows']
        if not unit['control']:
            keys += ['first_alarm_in_fault_period', 'lead_to_replacement_years', 'first_alarm_minus_replacement_years']
    else:
        keys += ['first_alarm_day', 'precedes_failure_day50']
    for key in keys:
        value, original = actual.get(key), reference.get(key)
        if original in ('', 'None', 'null', None):
            expected = None
        elif isinstance(value, bool):
            require(original in ('True', 'False'), f'Invalid v3 Boolean {key}')
            expected = original == 'True'
        elif isinstance(value, (int, float)):
            expected = type(value)(original)
        else:
            expected = original
        # v3 control fault_detected is null, matching the comparator export.
        if require_equal:
            require(value == expected, f'v3 per-unit mismatch {unit["unit"]} seed {actual["seed"]} {key}: {value!r} != {expected!r}')


def baseline_gate(units, seeds, models, mods, np, protocol):
    path = EXTERNAL / 'results_v3/per_unit.csv'
    with path.open('r', encoding='utf-8', newline='') as stream:
        original = [r for r in csv.DictReader(stream) if r['arm'] == ARM]
    cached, proposed_rows, evidence = {}, [], {'per_unit_sha256': sha(path), 'units': []}
    cadence, wider = mods['cadence'], mods['wider_models']
    for unit in units:
        for seed in seeds:
            refs = [r for r in original if r['dataset'] == unit['dataset'] and int(r['seed']) == seed and
                    (unit['dataset'] == 'MathWorks_HSS' or
                     int(r['turbine']) == unit['turbine'] and (r.get('event') or None) == unit['event'])]
            require(len(refs) == 1, f'Expected one v3 Arm A row: {unit["unit"]}, seed {seed}')
            ref = refs[0]
            require(ref['evaluation_status'] == unit['evaluation_status'], 'Evaluability differs from v3')
            if unit['evaluation_status'] != 'evaluable':
                proposed_rows.append(outcome(unit, seed, 'proposed_fedadam_h12', None, np))
                continue
            scores, original_flags, audit = load_saved_scores(unit, seed, np)
            fresh = wider.score(models[seed], unit['unit'] + '::' + ARM, unit['x'])
            require(np.array_equal(fresh, scores), f'v3 reconstruction-score mismatch: {unit["unit"]} seed {seed}; stop')
            flags = cadence.dense_flags(scores, unit['train_indices'], unit['calibration_indices'], 21, 13, 2.)
            require(np.array_equal(flags, original_flags), f'v3 alarm-vector mismatch: {unit["unit"]} seed {seed}; stop')
            row = outcome(unit, seed, 'proposed_fedadam_h12', flags, np)
            hist = np.concatenate((unit['train_indices'], unit['calibration_indices']))
            statistic = cadence._rolling(scores[hist], 21, 13)
            finite = statistic[np.isfinite(statistic)]
            reference = float(np.max(finite)) if len(finite) else float(np.quantile(scores[unit['calibration_indices']], .995))
            row.update(reference=reference, threshold=2*reference,
                       threshold_method='history_max_persistent' if len(finite) else 'calibration_point_quantile_0.995')
            compare_original_per_unit(unit, row, ref)
            cached[(unit['unit'], seed)] = (scores, flags)
            proposed_rows.append(row)
            evidence['units'].append(dict(unit=unit['unit'], seed=seed, exact_score_match=True,
                                          exact_alarm_vector_match=True, per_unit_match=True, **audit))
    for seed in seeds:
        selected = [r for r in proposed_rows if r['seed'] == seed]
        snd = [r for r in selected if r['primary'] and r['dataset'] == 'SND']
        controls = [r for r in snd if r['turbine'] != 5]
        t5 = next(r for r in snd if r['turbine'] == 5)
        hss = next(r for r in selected if r['dataset'] == 'MathWorks_HSS')
        event2 = next(r for r in selected if r['event'] == 'event2')
        expected = protocol['public_v3_reference'][str(seed)]
        require(sum(r['strict_success'] is True for r in snd) == expected['snd_strict_successes'] and
                sum(r['strict_success'] is True for r in controls) == expected['controls_silent'] and
                t5['fault_detected'] is False and t5['first_alarm_coordinate'] is None and
                hss['first_alarm_day'] == expected['hss_first_alarm_day'] and hss['strict_success'] is True and
                sum(r['strict_success'] is True for r in selected if r['primary']) == 6,
                f'Public v3 primary reference mismatch seed {seed}')
        require(event2['strict_success'] is True and event2['healthy_alarm'] is False and
                event2['fault_detected'] is True and event2['first_alarm_coordinate'] == expected['event2_first_alarm_years'] and
                event2['lead_time_days'] == expected['event2_lead_days'], f'Public v3 event-2 reference mismatch seed {seed}')
    evidence['pass'] = True
    return cached, proposed_rows, evidence


def decision_rules(published, scores, unit, seed, np):
    output = {}
    for sigma in (3, 4, 6):
        output[f'wang_{sigma}sigma'] = published.wang(scores, unit, sigma, persist=False)
    for regimes in (2, 1):
        thresholds, detail = published.gev(scores, unit, regimes, seed)
        name = 'gev_regime2' if regimes == 2 else 'gev_no_clustering'
        output[name] = published.point_flags(scores, thresholds, unit, persist=False), detail
    output['javanmardi_2means'] = published.onset(scores, unit, seed)
    return output


def aggregate(rows, methods, seeds):
    cells = []
    for method in methods:
        for seed in seeds:
            selected = [r for r in rows if r['method'] == method and r['seed'] == seed]
            primary = [r for r in selected if r['primary']]
            require(len(primary) == 7 and len(selected) == 8, 'Missing registered method-unit rows')
            controls = [r for r in primary if r['dataset'] == 'SND' and r['turbine'] != 5]
            hss = next(r for r in primary if r['dataset'] == 'MathWorks_HSS')
            event2 = next(r for r in selected if r['event'] == 'event2')
            t5 = next(r for r in primary if r['dataset'] == 'SND' and r['turbine'] == 5)
            cells.append({'method':method, 'seed':seed, 'strict_successes':sum(r['strict_success'] is True for r in primary),
                          'registered_primary_units':7, 'evaluable_primary_units':sum(r['evaluation_status']=='evaluable' for r in primary),
                          'controls_silent':sum(r['strict_success'] is True for r in controls), 'registered_controls':5,
                          't5_event1': t5, 'hss':hss, 't5_event2':event2})
    return cells


def format_report(summary):
    lines = ['# Unseen-machine comparator addendum', '', f'Complete: {summary["complete"]}.', '',
             summary['disclosure'], '',
             'Arm A only. Primary denominator: five SND controls, Turbine 5 event 1, and HSS (seven machines).',
             'HSS strict success: no alarm on record-days 21-44 and at least one alarm on record-days 45-50.',
             'V3 itself reported HSS timing without a strict score; this addendum uses the author-specified paper yardstick for every method.', '',
             '## Primary methods', '',
             '| Method | 17070 strict /7 | 17071 strict /7 | 17072 strict /7 |', '|---|---:|---:|---:|']
    if not summary['complete']:
        lines += ['', 'STOP: ' + summary['stop_reason'], 'Comparator outcomes are not presented as a completed comparison.']
        return '\n'.join(lines) + '\n'
    for method in PRIMARY_METHODS:
        cells = [next(c for c in summary['cells'] if c['method']==method and c['seed']==s) for s in summary['seeds']]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{c["strict_successes"]}/7' for c in cells) + ' |')
    lines.append('| C3: PCA + personalized H12 | not applicable | not applicable | not applicable |')
    lines += ['', '## Secondary decision-rule variants', '',
              '| Method | 17070 strict /7 | 17071 strict /7 | 17072 strict /7 |', '|---|---:|---:|---:|']
    for method in ALL_METHODS:
        if method not in PRIMARY_METHODS:
            cells = [next(c for c in summary['cells'] if c['method']==method and c['seed']==s) for s in summary['seeds']]
            lines.append('| ' + method + ' | ' + ' | '.join(f'{c["strict_successes"]}/7' for c in cells) + ' |')
    lines += ['', '## Secondary outcomes for every applicable method', '',
              '| Method | Seed | Controls silent /5 | T5 event-1 detected | HSS healthy alarm | HSS detected days 45-50 | HSS first-alarm day | Event-2 healthy alarm | Event-2 detected | Event-2 strict | Event-2 lead days |',
              '|---|---:|---:|---|---|---|---:|---|---|---|---:|']
    def cell(value):
        return 'null' if value is None else str(value)
    for c in summary['cells']:
        h, e = c['hss'], c['t5_event2']
        values = [c['method'],c['seed'],f'{c["controls_silent"]}/5',c['t5_event1']['fault_detected'],
                  h['healthy_alarm'],h['fault_detected'],h.get('first_alarm_day'),e['healthy_alarm'],e['fault_detected'],
                  e['strict_success'],e.get('lead_time_days')]
        lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
    lines += ['', '## HSS historical-onset sensitivity', '',
              '| Method | Seed | Status | Strict success | Healthy alarm | Fault detected |', '|---|---:|---|---|---|---|']
    for c in summary['cells']:
        h = c['hss']
        lines.append('| ' + ' | '.join(cell(v) for v in (c['method'],c['seed'],h['hss_proxy_status'],
                     h['hss_proxy_strict_success'],h['hss_proxy_healthy_alarm'],h['hss_proxy_fault_detected'])) + ' |')
    lines += ['', 'The unchanged historical estimator requires 105 acquisitions; HSS has 50. Its insufficient-data result is retained. No onset is substituted and the detector is not refitted for this sensitivity.', '',
              '## Applicability and interpretation', '',
              'C3: not applicable. The development personalized model retains client-specific decoders across federated rounds. No frozen procedure initializes and trains a new-device decoder from commissioning rows. The unseen-ID fallback uses a global decoder and does not reproduce the declared personalized method.',
              'C4: not included. The optional frozen logES adapter hard-codes 25600 Hz and two-channel development loaders; these unseen single-channel sources have different sampling rates. No new front-end settings were introduced.', '',
              'The Wang, GEV and two-means comparators reuse fixed decision code on the frozen H12 scores. They are adapted decision rules, not full published-system replications. Two-means uses the full trajectory retrospectively. GEV fit diagnostics and fallbacks are retained in summary.json.',
              'The proposed method passed exact score, alarm-vector and per-unit checks against v3 before comparator evaluation. HSS strict success adds the same author-specified yardstick to all methods; its v3 timing endpoint is also retained.',
              'Event 2 is secondary and shares Turbine 5 and its measurement channel with event 1. Lead days include premature alarms. No alarm has null timing. Replacement years are nominal and filename-to-turbine mapping remains assumed. Turbine 2 retains its documented electrical measurement issue.',
              'Seed repetitions and turbines from one farm are correlated. The proposed unseen results were known when this addendum was written. Comparator unseen results were unknown. Development configuration selection and retrospective onset labels remain limitations.', '',
              'Protocol, source, model, cached-input hashes, environment, all per-unit outcomes and exact baseline checks are in summary.json. Per-unit outcomes are also in per_unit.csv.']
    return '\n'.join(lines) + '\n'


def write_results(result_root, summary, rows):
    (result_root / 'summary.json').write_text(json.dumps(safe(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    export = [{k:json.dumps(safe(v), separators=(',', ':')) if isinstance(v, (dict,list,tuple)) else v
               for k,v in row.items()} for row in rows]
    fields = list(dict.fromkeys(k for r in export for k in r)) or ['method','seed','unit','strict_success']
    with (result_root / 'per_unit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(export)
    (result_root / 'REPORT.md').write_text(format_report(summary), encoding='utf-8')


def execute(root, protocol):
    result_root = install_write_boundary(root)
    require(not result_root.exists(), 'results already exists; no automatic rerun, overwrite or resume')
    result_root.mkdir()
    summary = {'complete':False, 'protocol_id':protocol['protocol_id'], 'seeds':protocol['seeds'],
               'disclosure':protocol['disclosure'], 'utc_started':datetime.now(timezone.utc).isoformat(),
               'protocol_sha256':sha(root/'comparators_protocol.json'), 'stop_reason':None,
               'applicability':protocol['applicability'], 'source_hashes':protocol['source_files_sha256'],
               'bundle_hashes':protocol['bundle_files_sha256'], 'cells':[], 'rows':[]}
    rows = []
    try:
        for path,digest in protocol['external_files_sha256'].items():
            require(sha(path)==digest, f'External frozen source/protocol drift: {path}')
        v3 = module('comparator_v3_runner', EXTERNAL/'run_unseen.py')
        _, p = v3.verify_protocol(EXTERNAL/'protocol.json')
        require(p['seeds']==protocol['seeds'], 'v3 seeds differ')
        metadata = v3.time_metadata(EXTERNAL,p)
        require(not metadata['reasons'], 'v3 identity/chronology gate failed')
        summary['identity_gate'] = metadata
        v3.check_data_integrity(p)
        mods = v3.frozen_imports(EXTERNAL,p)
        import numpy as np
        import pandas as pd
        import importlib.metadata
        from threadpoolctl import threadpool_limits
        summary['environment'] = {'python':sys.version,
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','scikit-learn','threadpoolctl')},
            'threads':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}}
        require(summary['environment']['packages']==protocol['stage1_package_versions'], 'Package versions differ from stage 1')
        with threadpool_limits(limits=1):
            units = build_units(v3,p,mods,np,pd)
            models = v3.load_models(EXTERNAL,p,np)
            cached, rows, gate = baseline_gate(units,protocol['seeds'],models,mods,np,protocol)
            summary['baseline_gate'] = gate
            summary['unit_audit'] = [{k:safe(v) for k,v in u.items() if k!='x'} for u in units]
            print('PASS: exact v3 scores, alarms, per-unit outcomes and public reference. Begin comparators.',flush=True)
            published = module('frozen_published_comparison',root/'frozen/pipeline/research/published_methods_20261021/run_comparison.py')
            c2 = {}
            for algorithm in ('fedavg','fedprox'):
                for seed in protocol['seeds']:
                    model = root/'frozen/models'/f'{algorithm}_h12_seed{seed}.pkl'
                    with model.open('rb') as stream:
                        state = pickle.load(stream)
                    require(state['algorithm']==algorithm and state['seed']==seed and state['rounds']==12 and
                            state['mu']==(.01 if algorithm=='fedprox' else 0) and tuple(state['clients'])==tuple(models[seed]['clients']),
                            'Comparator checkpoint identity/cohort mismatch')
                    c2[(algorithm,seed)] = state
            for unit in units:
                for seed in protocol['seeds']:
                    if unit['evaluation_status'] != 'evaluable':
                        rows.extend(outcome(unit,seed,m,None,np) for m in ALL_METHODS if m!='proposed_fedadam_h12')
                        continue
                    scores,_ = cached[(unit['unit'],seed)]
                    outputs = decision_rules(published,scores,unit,seed,np)
                    for algorithm in ('fedavg','fedprox'):
                        c2scores = mods['wider_models'].score(c2[(algorithm,seed)],unit['unit'],unit['x'])
                        require(np.all(np.isfinite(c2scores)), 'Nonfinite comparator reconstruction')
                        flags = mods['cadence'].dense_flags(c2scores,unit['train_indices'],unit['calibration_indices'],21,13,2.)
                        hist = np.concatenate((unit['train_indices'],unit['calibration_indices']))
                        statistic = mods['cadence']._rolling(c2scores[hist],21,13)
                        finite = statistic[np.isfinite(statistic)]
                        reference = float(np.max(finite)) if len(finite) else float(np.quantile(c2scores[unit['calibration_indices']],.995))
                        outputs[algorithm+'_h12'] = flags, {'reference':reference, 'threshold':2*reference,
                            'threshold_method':'history_max_persistent' if len(finite) else 'calibration_point_quantile_0.995',
                            'scores':c2scores.tolist()}
                    for method,(flags,fit) in outputs.items():
                        row = outcome(unit,seed,method,flags,np)
                        row['fit'] = safe(fit)
                        rows.append(row)
                    print(f'Compared {unit["unit"]} seed {seed}',flush=True)
            summary['cells'] = aggregate(rows,ALL_METHODS,protocol['seeds'])
            require(len(rows)==8*3*len(ALL_METHODS), 'Output row count differs from fixed design')
            summary['complete'] = True
            summary['rows'] = rows
            summary['utc_completed'] = datetime.now(timezone.utc).isoformat()
            write_results(result_root,summary,rows)
    except Exception as exc:
        summary.update(complete=False,stop_reason=f'{type(exc).__name__}: {exc}',rows=rows)
        write_results(result_root,summary,rows)
        raise
    print('Complete:',result_root,flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print('DRY RUN: arguments parsed; no protocol, source, score or waveform file opened. Analysis requires ' + str(DESTINATION))
        return
    root = Path(__file__).resolve().parent
    if args.run:
        require(root == DESTINATION.resolve(), 'Analysis permitted only from D:\\external_unseen\\comparators')
        require(Path.cwd().resolve()==root, 'Run analysis with D:\\external_unseen\\comparators as working directory')
    protocol = verify_bundle(root,args.protocol)
    if args.verify_only:
        print('PASS: local protocol seals, source copies, development evidence and model hashes; no target data opened.')
        return
    execute(root,protocol)


if __name__ == '__main__':
    main()
