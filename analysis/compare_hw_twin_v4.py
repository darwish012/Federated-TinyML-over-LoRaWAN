"""Replay protocol-v4 hardware delivery through the unchanged int16 hardware twin.

    python compare_hw_twin_v4.py RUN_DIR --output NEW_DIR [--allow-absent-positions]

Reads a v4 server run directory (PROTOCOL_V4.md section 4 "Logs"), re-runs the twin with the
logged per-phase uplink / downlink masks and base slots, and applies the predeclared
hardware_vs_twin criteria of research/hardware_twin_scaling_20260929/protocol.json PER PHASE:

  global_codes      after every round >= 99% of the 562 int8 codes identical, every difference <= +-1
  strict_outcomes   per-machine strict outcome identical, at most 1 mismatch per phase
  first_alarm       first-alarm index within +-1 for >= 95% of machines that alarm in both
  round_completion  all rounds of the phase complete

plus the v4 integrity checks (slot_integrity: GLOBAL slots unique / monotone and every logged
base_slot is a GLOBAL of the same phase; membership: the masks agree with the delivery log and
every base is a model the position actually held; planned_coverage: planned mode only).

Phases replace v3 levels; slots are run-wide u16 (base_slot is resolved to a round through the
phase's GLOBAL records); membership (joins, leaves, re-admissions) is whatever the masks show:
a position is simply absent (uplink False, downlink False) before it joins and after it departs.
  * planned: scope of a phase = its planned positions (masks "planned_positions"); a planned
    position that never delivered an UPDATE is reported and fails planned_coverage unless
    --allow-absent-positions.
  * elastic: scope of a phase = the positions that participated (>= 1 delivered UPDATE).
A position is scored for a phase only if it holds the phase's FINAL model (downlink True in the
last round): PROTOCOL_V4 3.4 replays only "every board that holds the final model". Positions of
the scope without it are listed as no_final_model (not a mismatch), and a REPLAY_DONE claiming
the phase for such a position is an inconsistency (strict_outcomes FAIL).

The field list this tool expects from the server is FIELDS_EXPECTED below (python compare_hw_twin_v4.py --fields).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'autoencoder_cwru_release/full_pipeline_simulation/research'
TWIN = RESEARCH / 'hardware_twin_scaling_20260929'
for directory in (TWIN, RESEARCH / 'healthy_only_suite_20260907', RESEARCH / 'fl_algorithm_comparison_20260929',
                  RESEARCH / 'healthy_only_causal_20260908'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
import numpy as np
import twin
import twin_server
import machine_order_v2
import raw_int16_v2
import fl
import cadence

SCHEMA = 'tinyfl-replay-masks-v4'
HARDWARE_ELIGIBLE = 41                       # machine_order_v2 positions 0..40
MAX_PHASE_INDEX = 254
N_PARAMS = twin.N_PARAMS                     # 562

FIELDS_EXPECTED = {
    'replay_masks.json': {
        'top': ['schema == "tinyfl-replay-masks-v4"', 'run_id (u32)', 'seed (int, fl.init seed of every phase unless the phase has its own)',
                'mode ("planned" | "elastic")', 'phases (list of objects with phase_index per PROTOCOL_V4 4.1; an object keyed by phase_index is also accepted)'],
        'per phase': ['phase_index (0..254)', 'rounds_in_phase R', 'rounds_completed (== len(uplink))', 'complete (bool)',
                      'positions (every position seen in the phase, unique; matrix columns follow this order)',
                      'uplink[row][col] bool: the UPDATE of that position for training round row+1 was delivered',
                      'downlink[row][col] bool: the position held the complete GLOBAL of round row+1 when delivery ended',
                      'base_slot[row][col] u16 | null: slot of the GLOBAL actually trained from (null where uplink is false)',
                      'planned_positions (planned mode: list)', 'machines (optional, parallel to positions)',
                      'seed (optional per-phase override)'],
    },
    'global_models.jsonl': ['run_id', 'phase_index', 'round (0..R)', 'slot (u16; "round_id" accepted)', 'codes (562 ints in [-127,127])',
                            'scales_f32 / crc32 / codes_sha256 (not used by the criteria)'],
    'delivery_log.jsonl': ['records with record == "round": phase_index, run_id (optional), counters update_fragments_received, '
                           'update_repair_requests, update_repair_downlinks_sent, global_fragments_sent, global_repair_rounds, global_bursts',
                           'optional cross-checks per round record: participants, arrived_positions, stale_positions, '
                           'joined_positions, left_positions (the k-th round record of a phase is mask row k)'],
    'alarms.csv': ['run_id', 'position', 'phase_index', 'acquisition_index', 'score', 'threshold (score/threshold not used)',
                   'source ("alarm", or "replay_done" when only REPLAY_DONE carried it: score/threshold empty; optional)'],
    'replay_done.csv': ['run_id', 'position', 'phase_index', 'acquisitions_scored', 'alarm_latched (0/1)',
                        'first_alarm_index (rev 4.3; 4294967295 or empty = none; optional)  -- one row per (position, phase) scored; repeats are merged'],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------------------
# Data: lazily loaded hardware-eligible machines (positions 0..40 of machine_order_v2)
# --------------------------------------------------------------------------------------
class LazyData:
    """position -> (meta, arrays, x); loaded on first use, int16 feature cache verified."""

    def __init__(self):
        info = machine_order_v2.build_all_v2()
        self.order = info['order']
        self.noldm = info['noldm']
        self._items = {}

    def __contains__(self, position):
        return isinstance(position, (int, np.integer)) and 0 <= position < HARDWARE_ELIGIBLE

    def __getitem__(self, position):
        if position not in self:
            raise KeyError(position)
        position = int(position)
        if position not in self._items:
            meta = self.order[position]
            with np.load(meta['path'], allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            cache = raw_int16_v2.load_machine_cache(meta['machine_id'])
            x = cache['x_int16'].astype(np.float32)
            reproduced, _ = twin.standardize(cache['raw22_int16'], arrays['train_indices'].astype(int))
            if not np.array_equal(x, reproduced):
                raise ValueError('int16 feature cache mismatch: ' + meta['machine_id'])
            self._items[position] = (meta, arrays, x)
        return self._items[position]


def load_data():
    data = LazyData()
    return data, data.noldm


# --------------------------------------------------------------------------------------
# Predeclared criteria (protocol.json hardware_vs_twin) -- the text is checked, not trusted
# --------------------------------------------------------------------------------------
def predeclared_criteria():
    spec = read(TWIN / 'protocol.json')['criteria_fixed_now']['hardware_vs_twin (applied later, same file)']
    needs = {'global_model': ('99%', '+-1'), 'per_machine_outcome': ('at most 1',), 'alarm_timing': ('+-1', '95%'),
             'completion': ('12 rounds',)}
    for key, words in needs.items():
        if any(w not in spec.get(key, '') for w in words):
            raise ValueError('protocol.json hardware_vs_twin.%s no longer says %s; update this tool deliberately' % (key, words))
    return {'identical_fraction': 0.99, 'max_code_delta': 1, 'max_strict_mismatches': 1, 'first_alarm_window': 1,
            'first_alarm_fraction': 0.95, 'thesis_rounds': 12}


# --------------------------------------------------------------------------------------
# Twin pieces
# --------------------------------------------------------------------------------------
def wire_scales(scales):
    return [float(np.float32(s)) for s in scales]


def quantize_global(server_params):
    """Server GLOBAL: twin downlink codec; the scales travel as float32."""
    codes_list, scales = twin.downlink_message(server_params)
    scales = wire_scales(scales)
    return (twin.all_codes_flat(codes_list).astype(int).tolist(),
            twin.decode_params(codes_list, scales, dtype=np.float64))


def quantize_update(trained, source):
    """Board UPDATE: twin per-machine uplink codec (delta vs the model trained from)."""
    codes_list, scales, _ = twin.per_machine_uplink_message(trained, source)
    return twin.decode_params(codes_list, wire_scales(scales), dtype=np.float64)


def flags_for(params, item):
    _, arrays, x = item
    scores = twin.score(params, x)
    if not np.all(np.isfinite(scores)):
        raise ValueError('nonfinite replay scores')
    return cadence.dense_flags(scores.astype(np.float64), arrays['train_indices'].astype(int),
                               arrays['calibration_indices'].astype(int), 21, 13, 2.0)


def _tsum(a, b, c):
    return tuple(x + y - z for x, y, z in zip(a, b, c))


# --------------------------------------------------------------------------------------
# Normalisation / validation of the logs
# --------------------------------------------------------------------------------------
def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def normalise_phases(masks):
    raw = masks.get('phases')
    if raw is None:
        raise ValueError('replay_masks.json has no "phases"')
    items = list(raw.items()) if isinstance(raw, dict) else [(str(p.get('phase_index')), p) for p in raw]
    phases = {}
    for key, ph in items:
        index = int(ph.get('phase_index', key))
        if int(key) != index:
            raise ValueError('phase key %s != phase_index %s' % (key, index))
        if not 0 <= index <= MAX_PHASE_INDEX or index in phases:
            raise ValueError('invalid or duplicate phase_index %s' % index)
        phases[index] = ph
    if not phases:
        raise ValueError('no phases in replay_masks.json')
    return dict(sorted(phases.items()))


def check_phase_shape(index, ph):
    positions = ph.get('positions')
    if not isinstance(positions, list) or not positions or not all(_is_int(p) for p in positions) or len(set(positions)) != len(positions):
        raise ValueError('phase %d: positions must be a non-empty list of unique integers' % index)
    for p in positions:
        if not 0 <= p < HARDWARE_ELIGIBLE:
            raise ValueError('phase %d: position %d is not hardware-eligible (0..%d)' % (index, p, HARDWARE_ELIGIBLE - 1))
    rounds = ph.get('rounds_in_phase')
    if not _is_int(rounds) or not 1 <= rounds <= 255:
        raise ValueError('phase %d: invalid rounds_in_phase' % index)
    for key in ('uplink', 'downlink', 'base_slot'):
        if key not in ph:
            raise ValueError('phase %d: missing %s' % (index, key))
    n = len(ph['uplink'])
    if n > rounds or len(ph['downlink']) != n or len(ph['base_slot']) != n:
        raise ValueError('phase %d: invalid round count' % index)
    if 'rounds_completed' in ph and ph['rounds_completed'] != n:
        raise ValueError('phase %d: rounds_completed %s != %d mask rows' % (index, ph['rounds_completed'], n))
    for key in ('uplink', 'downlink', 'base_slot'):
        if any(not isinstance(row, list) or len(row) != len(positions) for row in ph[key]):
            raise ValueError('phase %d: invalid mask shape: %s' % (index, key))
    for r in range(n):
        for i in range(len(positions)):
            up, dn, slot = ph['uplink'][r][i], ph['downlink'][r][i], ph['base_slot'][r][i]
            if up not in (True, False, 0, 1) or dn not in (True, False, 0, 1):
                raise ValueError('phase %d: masks must be boolean' % index)
            if bool(up) != (slot is not None):
                raise ValueError('phase %d row %d position %d: base_slot must be null exactly where uplink is false' % (index, r, positions[i]))
            if slot is not None and not (_is_int(slot) and 0 <= slot <= 0xFFFF):
                raise ValueError('phase %d: invalid base_slot %r' % (index, slot))
    if ph.get('machines') is not None and len(ph['machines']) != len(positions):
        raise ValueError('phase %d: machines length differs from positions' % index)
    planned = ph.get('planned_positions')
    if planned is not None and (not all(_is_int(p) for p in planned) or len(set(planned)) != len(planned)):
        raise ValueError('phase %d: invalid planned_positions' % index)
    return n


def group_models(models, run_id):
    """{phase: {round: record}} and the slot map; exactness problems are reported, not raised."""
    by_phase, problems = {}, []
    slots_seen = {}
    last_slot = None
    for m in models:
        if m.get('run_id') != run_id:
            raise ValueError('mixed run IDs')
        slot = m.get('slot', m.get('round_id'))
        if not _is_int(slot) or not 0 <= slot <= 0xFFFF:
            raise ValueError('global_models.jsonl record without a valid u16 slot (field "slot")')
        phase, rnd = m.get('phase_index'), m.get('round')
        if not _is_int(phase) or not _is_int(rnd):
            raise ValueError('global_models.jsonl record without phase_index/round')
        if slot in slots_seen:
            problems.append('slot %d used by GLOBAL (phase %s round %s) and (phase %s round %s)' % (slot, *slots_seen[slot], phase, rnd))
        slots_seen[slot] = (phase, rnd)
        if last_slot is not None and slot <= last_slot:
            problems.append('slot %d does not increase (previous %d)' % (slot, last_slot))
        last_slot = slot
        if rnd in by_phase.setdefault(phase, {}):
            by_phase[phase][rnd].setdefault('_duplicates', []).append(m)
        else:
            by_phase[phase][rnd] = dict(m, slot=slot)
    for phase, rounds in by_phase.items():
        ordered = [rounds[r]['slot'] for r in sorted(rounds)]
        if ordered != sorted(ordered):
            problems.append('phase %d: slots do not increase with the round' % phase)
    return by_phase, problems


def slot_round_map(records):
    return {rec['slot']: r for r, rec in records.items()}


# --------------------------------------------------------------------------------------
# The twin replay of one phase
# --------------------------------------------------------------------------------------
def replay_phase(seed, ph, slot_round, data, evaluator):
    positions = ph['positions']
    n = len(ph['uplink'])
    R = ph['rounds_in_phase']
    server = twin_server.Server(fl.init(list(twin.LAYERS), seed))
    code0, g0 = quantize_global(server.p)
    globals_, codes = [g0], [code0]
    trained_cache = {}
    for r in range(n):
        pairs = []
        for i, p in enumerate(positions):
            if not ph['uplink'][r][i]:
                continue
            slot = ph['base_slot'][r][i]
            if slot not in slot_round:
                raise ValueError('phase %s row %d position %d: base_slot %d is not a GLOBAL of this phase in global_models.jsonl' % (ph['phase_index'], r, p, slot))
            base = slot_round[slot]
            if not 0 <= base <= r:
                raise ValueError('phase %s row %d position %d: base round %d outside 0..%d' % (ph['phase_index'], r, p, base, r))
            key = (p, base)
            if key not in trained_cache:
                source = twin.to_float32(globals_[base])
                _, arrays, x = data[p]
                trained = twin.local_train(x[arrays['train_indices'].astype(int)], source)
                trained_cache[key] = quantize_update(trained, source)
            decoded = trained_cache[key]
            if base != r:                       # server stale-base semantics: aggregate client - current_global
                decoded = _tsum(globals_[base], decoded, globals_[r])
            pairs.append((decoded, 1.0))
        server.round(pairs)                     # empty round: model and FedAdam step unchanged
        code, glob = quantize_global(server.p)
        globals_.append(glob)
        codes.append(code)
    final = {}
    if n == R:
        for i, p in enumerate(positions):
            if ph['downlink'][n - 1][i]:
                final[p] = twin.to_float32(globals_[n])
    return codes, final


def twin_rows(final, data, evaluator):
    return {p: evaluator.evaluate(data[p][0], data[p][1], flags_for(params, data[p])) for p, params in final.items()}


# --------------------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------------------
def membership_report(ph):
    positions = ph['positions']
    n = len(ph['uplink'])
    out = {}
    for i, p in enumerate(positions):
        up = [r for r in range(n) if ph['uplink'][r][i]]
        dn = [r for r in range(n) if ph['downlink'][r][i]]
        entry = {'uplink_rounds': len(up), 'downlink_rounds': len(dn),
                 'first_uplink_row': up[0] if up else None, 'last_uplink_row': up[-1] if up else None,
                 'holds_final_model': bool(n == ph['rounds_in_phase'] and n and ph['downlink'][n - 1][i])}
        if up:
            entry['gap_rows'] = [r for r in range(up[0], up[-1] + 1) if not ph['uplink'][r][i]]
            entry['joined_after_start'] = up[0] > 0
            entry['left_before_end'] = bool(up[-1] < n - 1 and not any(ph['uplink'][r][i] for r in range(up[-1] + 1, n)))
        out[str(p)] = entry
    return out


def participation(ph, rounds_records, mode):
    """participants[row] = set(positions) -- delivery log if it carries them, else planned
    (every planned position every round) or the first..last uplink window of each position."""
    positions = ph['positions']
    n = len(ph['uplink'])
    if rounds_records and len(rounds_records) == n and all('participants' in rec for rec in rounds_records):
        return [set(rec['participants']) for rec in rounds_records], 'delivery_log'
    if mode == 'planned' and ph.get('planned_positions') is not None:
        return [set(ph['planned_positions']) for _ in range(n)], 'planned_positions'
    rows = [set() for _ in range(n)]
    for i, p in enumerate(positions):
        up = [r for r in range(n) if ph['uplink'][r][i]]
        for r in range(up[0], up[-1] + 1) if up else []:
            rows[r].add(p)
    return rows, 'uplink_window'


def membership_problems(ph, base_rounds, rounds_records, parts, mode):
    problems = []
    positions = ph['positions']
    n = len(ph['uplink'])
    # (a) a position can only train from a GLOBAL it held: round 0 (phase start) or one it received
    for i, p in enumerate(positions):
        for r in range(n):
            if ph['uplink'][r][i]:
                b = base_rounds[r][i]
                if b > 0 and not ph['downlink'][b - 1][i]:
                    problems.append('position %d row %d trained from GLOBAL %d that its downlink mask says it never held' % (p, r, b))
    # (b) planned mode: nobody outside the plan delivers
    planned = ph.get('planned_positions')
    if mode == 'planned' and planned is not None:
        for i, p in enumerate(positions):
            if p not in planned and any(ph['uplink'][r][i] for r in range(n)):
                problems.append('position %d delivered UPDATEs but is not planned for this phase' % p)
    # (c) delivery log cross-checks, when the log carries the fields
    if rounds_records:
        if len(rounds_records) != n:
            problems.append('%d delivery round records for %d mask rows' % (len(rounds_records), n))
        else:
            for r, rec in enumerate(rounds_records):
                arrived = {p for i, p in enumerate(positions) if ph['uplink'][r][i]}
                if 'arrived_positions' in rec and set(rec['arrived_positions']) != arrived:
                    problems.append('row %d: delivery arrived_positions %s != uplink mask %s' % (r, sorted(rec['arrived_positions']), sorted(arrived)))
                if 'participants' in rec and not arrived <= set(rec['participants']):
                    problems.append('row %d: UPDATE delivered by non-participant(s) %s' % (r, sorted(arrived - set(rec['participants']))))
                if 'stale_positions' in rec:
                    stale = {p for i, p in enumerate(positions) if ph['uplink'][r][i] and base_rounds[r][i] < r}
                    if set(rec['stale_positions']) != stale:
                        problems.append('row %d: delivery stale_positions %s != masks %s' % (r, sorted(rec['stale_positions']), sorted(stale)))
                if 'joined_positions' in rec and 'participants' in rec:
                    before = set(rounds_records[r - 1]['participants']) if r and 'participants' in rounds_records[r - 1] else set()
                    for p in rec['joined_positions']:
                        if p not in rec['participants'] or p in before:
                            problems.append('row %d: position %d logged as joined but was not newly participating' % (r, p))
                        elif p in positions and ph['uplink'][r][positions.index(p)] and base_rounds[r][positions.index(p)] != r:
                            # PROTOCOL_V4 4 Mid-phase join: ACTIVE from the first round whose seed GLOBAL it holds
                            problems.append('row %d: joined position %d trained from GLOBAL %d, not from that round\'s seed' % (r, p, base_rounds[r][positions.index(p)]))
                if 'left_positions' in rec and 'participants' in rec:
                    before = set(rounds_records[r - 1]['participants']) if r and 'participants' in rounds_records[r - 1] else set()
                    for p in rec['left_positions']:
                        if p in rec['participants'] or p not in before:
                            problems.append('row %d: position %d logged as left but was not a participant that just stopped' % (r, p))
    return problems


# --------------------------------------------------------------------------------------
# Hardware-side facts
# --------------------------------------------------------------------------------------
def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


NO_ALARM_INDEX = 0xFFFFFFFF


def done_table(dones):
    """{(position, phase_index): {'alarm_latched': bool, 'acquisitions_scored': int, 'first_alarm_index': int|None}}
    -- REPLAY_DONE is per (position, phase), rev 4.1; rev 4.3 adds first_alarm_index (0xFFFFFFFF / empty = none)."""
    out = {}
    for d in dones:
        key = (_int(d.get('position'), -1), _int(d.get('phase_index'), -1))
        e = out.setdefault(key, {'alarm_latched': False, 'acquisitions_scored': 0, 'first_alarm_index': None})
        e['alarm_latched'] = e['alarm_latched'] or bool(_int(d.get('alarm_latched')))
        e['acquisitions_scored'] = max(e['acquisitions_scored'], _int(d.get('acquisitions_scored')))
        index = _int(d.get('first_alarm_index'), NO_ALARM_INDEX)
        if index != NO_ALARM_INDEX:
            e['first_alarm_index'] = index if e['first_alarm_index'] is None else min(e['first_alarm_index'], index)
    return out


# --------------------------------------------------------------------------------------
# One phase
# --------------------------------------------------------------------------------------
def compare_phase(index, ph, seed, mode, records, slot_problems, delivery_rounds, alarms, done, data, evaluator, crit,
                  allow_absent):
    n = check_phase_shape(index, ph)
    positions = ph['positions']
    R = ph['rounds_in_phase']
    if ph.get('machines') is not None and ph['machines'] != [data[p][0]['machine_id'] for p in positions]:
        raise ValueError('phase %d: logged machine order differs from amendment 2' % index)
    slot_round = slot_round_map(records)
    base_rounds = [[None if s is None else slot_round.get(s) for s in row] for row in ph['base_slot']]
    for r, row in enumerate(ph['base_slot']):
        for i, s in enumerate(row):
            if s is not None and base_rounds[r][i] is None:
                raise ValueError('phase %d row %d position %d: base_slot %d is not a GLOBAL of this phase in global_models.jsonl' % (index, r, positions[i], s))
            if s is not None and base_rounds[r][i] > r:
                raise ValueError('phase %d row %d position %d: invalid delivered base slot %d (GLOBAL round %d is newer than the round)' % (index, r, positions[i], s, base_rounds[r][i]))
    base_rounds = [[-1 if b is None else b for b in row] for row in base_rounds]
    ph = dict(ph, phase_index=index)
    phase_seed = ph.get('seed', seed)
    codes, final = replay_phase(phase_seed, ph, slot_round, data, evaluator)
    rows = twin_rows(final, data, evaluator)

    rounds = []
    for r, expected in enumerate(codes):
        rec = records.get(r)
        entry = {'round': r, 'pass': False, 'differences': []}
        if rec is None or '_duplicates' in rec or 'codes' not in rec:
            entry['error'] = 'Missing/duplicate GLOBAL or exact codes unavailable; criterion cannot be established.'
        else:
            actual = np.asarray(rec['codes'], dtype=np.int64)
            if actual.shape != (N_PARAMS,) or np.any((actual < -127) | (actual > 127)):
                entry['error'] = 'invalid int8 code vector'
            else:
                delta = actual - np.asarray(expected)
                entry['identical_fraction'] = float(np.mean(delta == 0))
                entry['differences'] = [{'index': int(i), 'hardware': int(actual[i]), 'twin': expected[i], 'delta': int(delta[i])}
                                        for i in np.flatnonzero(delta)]
                entry['pass'] = entry['identical_fraction'] >= crit['identical_fraction'] and bool(np.all(np.abs(delta) <= crit['max_code_delta']))
        rounds.append(entry)

    participated = [p for i, p in enumerate(positions) if any(ph['uplink'][r][i] for r in range(n))]
    planned = ph.get('planned_positions')
    if mode == 'planned' and planned is not None:
        scope = list(planned)
        absent = [p for p in planned if p not in participated]
    else:
        scope = participated
        absent = []
    unknown = [p for p in scope if p not in data]
    if unknown:
        raise ValueError('phase %d: positions %s are not hardware-eligible' % (index, unknown))

    machines, mismatches, unfinished, ambiguous, no_final, inconsistent = [], [], [], [], [], []
    both = close = 0
    for p in scope:
        mid = data[p][0]['machine_id']
        d = done.get((p, index))
        bit = d is not None
        if p not in final:
            no_final.append(mid)
            if bit:
                inconsistent.append(mid)
            machines.append({'position': p, 'machine': mid, 'scored_by_twin': False, 'reason': 'no_final_model' if p in participated else 'never_participated',
                             'hardware_replay_done': bit})
            continue
        item, row = data[p], rows[p]
        finished = bit
        idxs = [a['acquisition_index'] for a in alarms if a['position'] == p and a['phase_index'] == index]
        first = min(idxs) if idxs else None
        # rev 4.3: REPLAY_DONE carries the first alarm's index, so a lost ALARM message no longer loses
        # the alarm time (the server also writes such an alarm to alarms.csv with source=replay_done)
        done_index = d['first_alarm_index'] if d is not None else None
        if first is None and done_index is not None:
            first = done_index
        elif first is not None and done_index is not None and done_index != first:
            inconsistent.append(mid)                                   # ALARM index != REPLAY_DONE index (3.3)
        lost = bool(d and d['alarm_latched'] and first is None)       # the board says it latched an alarm, no ALARM message arrived
        if d is not None and not d['alarm_latched'] and first is not None:
            inconsistent.append(mid)                                   # an ALARM message although REPLAY_DONE says none latched
        hw_strict = None
        if finished:
            if first is None:
                if lost:
                    ambiguous.append(mid)
                else:
                    hw_strict = row['expected'] == 'healthy_control'
            elif first in item[1]['healthy_test_indices']:
                hw_strict = False
            elif first in item[1]['unhealthy_test_indices']:
                hw_strict = row['expected'] == 'fault'
            elif row['expected'] == 'unknown':
                hw_strict = False
            else:
                ambiguous.append(mid)
        else:
            unfinished.append(mid)
        mismatch = hw_strict is None or hw_strict != row['strict_success']
        if mismatch and row['scored']:
            mismatches.append(mid)
        tf = row['first_alarm_index']
        alarm_close = None
        if first is not None and tf is not None:
            both += 1
            alarm_close = abs(first - tf) <= crit['first_alarm_window']
            close += int(alarm_close)
        machines.append({'position': p, 'machine': mid, 'scored_by_twin': True, 'expected': row['expected'], 'finished': finished,
                         'hardware_strict': hw_strict, 'twin_strict': row['strict_success'], 'strict_differs': mismatch,
                         'hardware_first_alarm': first, 'twin_first_alarm': tf, 'first_alarm_within_one': alarm_close})

    membership = membership_report(ph)
    parts, basis = participation(ph, delivery_rounds, mode)
    m_problems = membership_problems(ph, base_rounds, delivery_rounds, parts, mode)
    stats = {key: sum(d.get(key, 0) for d in delivery_rounds)
             for key in ('update_fragments_received', 'update_repair_requests', 'update_repair_downlinks_sent',
                         'global_fragments_sent', 'global_repair_rounds', 'global_bursts')}
    stats.update({
        'participation_basis': basis,
        'missing_uplinks': sum(1 for r in range(n) for i, p in enumerate(positions) if p in parts[r] and not ph['uplink'][r][i]),
        'missing_downlinks': sum(1 for r in range(n) for i, p in enumerate(positions) if p in parts[r] and not ph['downlink'][r][i]),
        'stale_bases': sum(1 for r in range(n) for i in range(len(positions)) if ph['uplink'][r][i] and base_rounds[r][i] < r),
        'empty_rounds': sum(1 for r in range(n) if not any(ph['uplink'][r])),
        'exact_fragment_losses': None})

    completed = bool(ph.get('complete')) and n == R and len(delivery_rounds) == R
    criteria = {
        'global_codes': all(r['pass'] for r in rounds) and len(rounds) == R + 1 and len(records) == R + 1,
        'strict_outcomes': len(mismatches) <= crit['max_strict_mismatches'] and not unfinished and not ambiguous and not inconsistent,
        'first_alarm': (both == 0 or close / both >= crit['first_alarm_fraction']) and not unfinished and not ambiguous and not inconsistent,
        'round_completion': completed,
        'slot_integrity': not slot_problems,
        'membership': not m_problems,
    }
    if mode == 'planned':
        criteria['planned_coverage'] = (not absent) or allow_absent
    return {
        'criteria': {k: 'PASS' if v else 'FAIL' for k, v in criteria.items()},
        'mode': mode, 'rounds_in_phase': R, 'rounds_completed': n, 'positions_seen': positions, 'scope': scope,
        'participated': participated, 'absent_planned_positions': absent, 'no_final_model': no_final,
        'rounds': rounds, 'machines': machines, 'strict_mismatches': mismatches, 'unfinished': unfinished,
        'ambiguous': ambiguous, 'inconsistent_replay_done': inconsistent,
        'first_alarm': {'both_alarming': both, 'within_one': close, 'fraction': close / both if both else None, 'empty_cohort': both == 0},
        'membership': membership, 'membership_problems': m_problems, 'slot_problems': slot_problems,
        'loss_statistics': stats,
        'joins_leaves_logged': {'joined': sorted({p for rec in delivery_rounds for p in rec.get('joined_positions', [])}),
                                'left': sorted({p for rec in delivery_rounds for p in rec.get('left_positions', [])})},
    }


# --------------------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------------------
def compare(run, data=None, evaluator=None, allow_absent=False):
    run = Path(run)
    crit = predeclared_criteria()
    masks = read(run / 'replay_masks.json')
    if masks.get('schema') != SCHEMA:
        raise ValueError('schema %s required' % SCHEMA)
    mode = masks.get('mode')
    if mode not in ('planned', 'elastic'):
        raise ValueError('replay_masks.json mode must be "planned" or "elastic"')
    models = jsonl(run / 'global_models.jsonl')
    delivery = jsonl(run / 'delivery_log.jsonl')
    alarms_raw = csv_rows(run / 'alarms.csv')
    dones_raw = csv_rows(run / 'replay_done.csv')
    run_id = masks['run_id']
    for rec in delivery:
        if 'run_id' in rec and rec['run_id'] != run_id:
            raise ValueError('mixed run IDs')
    for row in alarms_raw + dones_raw:
        if row.get('run_id') not in (None, '') and _int(row['run_id'], -1) != run_id:
            raise ValueError('mixed run IDs')
    if data is None:
        data, evaluator = load_data()
    phases = normalise_phases(masks)
    by_phase, slot_problems_all = group_models(models, run_id)
    for phase in by_phase:
        if phase not in phases:
            slot_problems_all.append('GLOBAL records for phase %d which has no mask entry' % phase)
    alarms = [{'position': _int(a.get('position'), -1), 'phase_index': _int(a.get('phase_index'), -1),
               'acquisition_index': _int(a.get('acquisition_index'), -1)} for a in alarms_raw]
    done = done_table(dones_raw)
    report = {'run_id': run_id, 'seed': masks['seed'], 'mode': mode, 'run_directory': str(run.resolve()),
              'criteria_applied': crit,
              'notes': ['Wire float32 scales and server stale-base reconstruction are reproduced; hardware masks and base slots drive the replay.',
                        'Fragment losses before receipt cannot be counted exactly from these logs; received fragments and repair traffic are reported.',
                        'ALARM logs contain latched first alarms, not all decision flags. Strict outcomes can be unidentifiable when first alarm is '
                        'outside healthy/fault test sets; these fail closed.',
                        'A position is scored for a phase only if it holds the final model (downlink True in the last round); REPLAY_DONE is per '
                        '(position, phase) (rev 4.1), so every phase 0..254 can be confirmed.',
                        'Elastic runs: criteria are applied per phase over the positions that participated; positions that left before the end '
                        'hold no final model and are listed as no_final_model.'],
              'spec_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (TWIN / 'protocol.json', TWIN / 'amendment_hardware_scope.json', TWIN / 'amendment2_int16_raw_order.json',
                                        ROOT / 'lorawan_two_node_experiment/PROTOCOL_V4.md') if p.exists()},
              'phases': {}}
    run_wide = [p for p in slot_problems_all if not p.startswith('phase ')]
    for index, ph in phases.items():
        records = by_phase.get(index, {})
        own_problems = [p for p in slot_problems_all if p.startswith('phase %d:' % index)] + run_wide
        rounds_records = [d for d in delivery if d.get('record') == 'round' and d.get('phase_index') == index]
        result = compare_phase(index, ph, masks['seed'], mode, records, own_problems, rounds_records, alarms, done,
                               data, evaluator, crit, allow_absent)
        if ph['rounds_in_phase'] != crit['thesis_rounds']:
            result['note'] = 'rounds_in_phase %d differs from the predeclared 12; criteria applied with R=%d' % (ph['rounds_in_phase'], ph['rounds_in_phase'])
        report['phases'][str(index)] = result
    report['pass'] = bool(report['phases']) and all(v == 'PASS' for ph in report['phases'].values() for v in ph['criteria'].values())
    return report


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    text = ['# Hardware vs twin comparison (protocol v4)', '',
            'Run %s; seed %s; mode %s; overall %s' % (report['run_id'], report['seed'], report['mode'], 'PASS' if report['pass'] else 'FAIL'),
            '', *report['notes']]
    for phase, result in report['phases'].items():
        text += ['', '## Phase %s' % phase, '', *['- %s: **%s**' % (k, v) for k, v in result['criteria'].items()], '',
                 'Scope: %s; no final model: %s; absent planned: %s' % (result['scope'], result['no_final_model'], result['absent_planned_positions']),
                 '', 'Loss statistics: ' + json.dumps(result['loss_statistics']),
                 '', 'Machine differences: ' + json.dumps([m for m in result['machines'] if m.get('strict_differs') or m.get('first_alarm_within_one') is False])]
        if result['membership_problems'] or result['slot_problems']:
            text += ['', 'Problems: ' + json.dumps(result['membership_problems'] + result['slot_problems'])]
        for r in result['rounds']:
            if not r['pass'] or r['differences']:
                text += ['', 'Round %d: ' % r['round'] + json.dumps(r)]
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (directory / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_directory', type=Path, nargs='?')
    parser.add_argument('--output', type=Path, help='New directory; existing directories are refused')
    parser.add_argument('--fields', action='store_true', help='print the fields expected from the server and exit')
    parser.add_argument('--allow-absent-positions', action='store_true', help='planned mode: do not fail planned positions that never delivered')
    args = parser.parse_args()
    if args.fields:
        print(json.dumps(FIELDS_EXPECTED, indent=2))
        return 0
    if args.run_directory is None or args.output is None:
        parser.error('run_directory and --output are required')
    try:
        report = compare(args.run_directory, allow_absent=args.allow_absent_positions)
        write_report(report, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print('PASS' if report['pass'] else 'FAIL')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
