"""Synthetic PROTOCOL_V4 run-directory generator (test fixtures for compare_hw_twin_v4.py).

The v4 server is written in parallel, so this generator builds the logs from PROTOCOL_V4.md
section 4 ("Logs") and sections 3.1 / 3.2 / 3.4 directly. It emulates the HARDWARE side with its
own arithmetic, deliberately different from the twin the comparison tool replays with:

  * boards quantise UPDATE deltas in float32 (`scale = max(max|x|/127, 1e-7)` and
    `q = roundf(x / scale)` in float32, PROTOCOL_V4 3.1) and dequantise GLOBALs with a float32 multiply;
  * the server quantises the float64 FedAdam state directly (twin_server.Server is the shared module,
    as the contract mandates), reconstructs stale UPDATEs as `global[base_slot] + delta - global[round]`
    (3.1) and numbers every GLOBAL with the next run-wide u16 slot (section 1).

So an agreement between this generator and the tool is a real test of the +-1 / 99% tolerance, not an
identity. A scenario fixes who is in each phase and which messages are lost; everything else is
derived. Scenarios: planned_scenario() (the 10-board 4..28 sweep with losses, an empty round and
stale bases) and elastic_scenario() (joins, leaves, a re-admission, a lost catch-up downlink).

Fields written (everything the comparison tool reads is listed in compare_hw_twin_v4.FIELDS_EXPECTED):
  replay_masks.json  global_models.jsonl  delivery_log.jsonl  alarms.csv  replay_done.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

import compare_hw_twin_v4 as cmp

twin, twin_server, fl = cmp.twin, cmp.twin_server, cmp.fl
F32 = np.float32
SLICES = twin.TENSOR_SLICES
RUN_ID = 0x5EED0001
SEED = 17070
ROUNDS_THESIS = 12
ALARM_FIELDS = ('run_id', 'position', 'phase_index', 'acquisition_index', 'score', 'threshold', 'source')       # PROTOCOL_V4 4.1 / 4.3
DONE_FIELDS = ('run_id', 'position', 'phase_index', 'acquisitions_scored', 'alarm_latched', 'first_alarm_index')   # rev 4.3: + first alarm index


# ---------------------------------------------------------------------------
# Hardware-side arithmetic (independent of twin.encode_tensor)
# ---------------------------------------------------------------------------
def fw_roundf(x):
    """C roundf in float32 (ties away from zero)."""
    x = np.asarray(x, dtype=F32)
    t = np.trunc(x)
    return (t + np.where(np.abs(x - t) >= F32(0.5), np.sign(x), F32(0))).astype(F32)


def fw_quantize_update(delta_tensors):
    """Board side: float32 per-tensor int8 codes + float32 scales of a 562-vector."""
    flat = np.concatenate([np.asarray(t, dtype=F32).ravel() for t in delta_tensors])
    codes, scales = np.empty(562, dtype=np.int8), []
    for sl in SLICES:
        x = flat[sl]
        scale = F32(np.max(np.abs(x))) / F32(127)
        scale = max(scale, F32(1e-7))
        codes[sl] = np.clip(fw_roundf(x / scale), -127, 127).astype(np.int8)
        scales.append(F32(scale))
    return codes, scales


def srv_quantize_global(params64):
    """Server side: float64 FedAdam state -> int8 codes + float32 scales (as v3's protocol_v3)."""
    flat = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in params64])
    codes, scales = np.empty(562, dtype=np.int8), []
    for sl in SLICES:
        x = flat[sl]
        scale = max(float(np.max(np.abs(x))) / 127.0, 1e-7)
        y = x / scale
        codes[sl] = np.clip(np.trunc(y + np.copysign(0.5, y)), -127, 127).astype(np.int8)
        scales.append(F32(scale))
    return codes, scales


def dequantize(codes, scales, dtype):
    out = np.empty(562, dtype=dtype)
    for k, sl in enumerate(SLICES):
        if dtype is F32 or dtype == np.float32:
            out[sl] = codes[sl].astype(F32) * F32(scales[k])                 # firmware: float32 multiply
        else:
            out[sl] = codes[sl].astype(np.float64) * float(F32(scales[k]))   # server: float64 of the wire scale
    return out


def unflat(vec):
    return tuple(vec[sl].reshape(shape) for sl, shape in zip(SLICES, twin.SHAPES))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def member(join=0, leave_after=None, absent=()):
    return {'join': join, 'leave_after': leave_after, 'absent': tuple(absent)}


def planned_scenario(n_boards=10, phase_sizes=(4, 8, 12, 16, 20, 24, 28), rounds=ROUNDS_THESIS, loss=0.07, rng_seed=99,
                     empty_round=(0, 5)):
    """The thesis sweep as one planned configuration: phase i activates the first N positions.
    Random uplink / downlink losses (GLOBAL 0 is always delivered), plus one fully empty round."""
    rng = np.random.default_rng(rng_seed)
    phases = []
    for i, size in enumerate(phase_sizes):
        up = {(r, p) for r in range(rounds) for p in range(size) if rng.random() < loss}
        dn = {(r, p) for r in range(rounds) for p in range(size) if rng.random() < loss}
        if empty_round and empty_round[0] == i:
            up |= {(empty_round[1], p) for p in range(size)}
        phases.append({'phase_index': i, 'rounds': rounds, 'planned_positions': list(range(size)),
                       'members': {p: member() for p in range(size)}, 'uplink_loss': up, 'downlink_loss': dn})
    return {'mode': 'planned', 'n_boards': n_boards, 'phases': phases}


def elastic_scenario(n_boards=4):
    """Two phases, boards joining, leaving and being re-admitted at round boundaries.

    phase 0 (8 rounds): 0,1,2,3 from the start; 4 joins at row 3, 5 at row 5; 2 leaves after row 5;
                        1 is absent in rows 3-4 and re-admitted at row 5.
    phase 1 (6 rounds): 0,1,3,4,5 from the start; 3 leaves after row 3; 6 and 7 join at row 2, but the
                        catch-up downlink of 6 is lost so it can only join at row 3. Position 2 is
                        not in this phase at all."""
    p0 = {'phase_index': 0, 'rounds': 8, 'planned_positions': None,
          'members': {0: member(), 1: member(absent=(3, 4)), 2: member(leave_after=5), 3: member(), 4: member(join=3), 5: member(join=5)},
          'uplink_loss': {(1, 3), (2, 0), (6, 4)}, 'downlink_loss': {(2, 0), (4, 3), (6, 5)}}
    p1 = {'phase_index': 1, 'rounds': 6, 'planned_positions': None,
          'members': {0: member(), 1: member(), 3: member(leave_after=3), 4: member(), 5: member(), 6: member(join=2), 7: member(join=2)},
          'uplink_loss': {(0, 5), (4, 4)}, 'downlink_loss': {(1, 6), (3, 1)}}
    return {'mode': 'elastic', 'n_boards': n_boards, 'phases': [p0, p1]}


def active_at(spec, r, rounds):
    last = rounds - 1 if spec['leave_after'] is None else min(spec['leave_after'], rounds - 1)
    return spec['join'] <= r <= last and r not in spec['absent']


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def generate_run(out_dir, scenario, data, evaluator, run_id=RUN_ID, seed=SEED, slot_start=0):
    """Writes the v4 run directory and returns the ground truth (codes, held rounds, flags...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode, n_boards = scenario['mode'], scenario['n_boards']
    models, deliveries, masks_phases, alarm_rows, finals = [], [], {}, [], {}
    truth = {'phases': {}}
    slot = slot_start
    for ph in scenario['phases']:
        idx, R = ph['phase_index'], ph['rounds']
        members = ph['members']
        positions = sorted(members)
        col = {p: i for i, p in enumerate(positions)}
        server = twin_server.Server(fl.init(list(twin.LAYERS), seed))
        g_codes, g_scales, g64, g32, g_slot = [], [], [], [], []

        def publish(round_index):
            nonlocal slot
            codes, scales = srv_quantize_global(server.p)
            g_codes.append(codes)
            g_scales.append(scales)
            g64.append(unflat(dequantize(codes, scales, np.float64)))
            g32.append(unflat(dequantize(codes, scales, F32)))
            g_slot.append(slot)
            blob = codes.tobytes() + b''.join(np.float32(s).tobytes() for s in scales)
            models.append({'run_id': run_id, 'phase_index': idx, 'round': round_index, 'slot': slot,
                           'rounds_in_phase': R, 'codes': [int(c) for c in codes], 'scales_f32': [float(s) for s in scales],
                           'crc32': '%08X' % (zlib.crc32(blob) & 0xFFFFFFFF), 'codes_sha256': hashlib.sha256(codes.tobytes()).hexdigest()})
            slot += 1

        publish(0)
        held = {p: (0 if members[p]['join'] == 0 else None) for p in positions}
        up = [[False] * len(positions) for _ in range(R)]
        dn = [[False] * len(positions) for _ in range(R)]
        base = [[None] * len(positions) for _ in range(R)]
        deliveries.append({'record': 'phase_start', 'run_id': run_id, 'phase_index': idx, 'rounds_in_phase': R, 'mode': mode,
                           'seed': seed, 'initial_global_slot': g_slot[0]})
        prev_active = set()
        for r in range(R):
            attempted = [p for p in positions if active_at(members[p], r, R) and held[p] is not None]
            pairs, arrived, stale, bases = [], [], [], {}
            for p in attempted:
                b = held[p]
                _, arrays, x = data[p]
                source = g32[b]
                trained = twin.local_train(x[arrays['train_indices'].astype(int)], source)
                delta = tuple(np.asarray(t, F32) - np.asarray(s, F32) for t, s in zip(trained, source))
                codes, scales = fw_quantize_update(delta)
                if (r, p) in ph['uplink_loss']:
                    continue
                decoded = unflat(dequantize(codes, scales, np.float64))
                client = tuple(a + d for a, d in zip(g64[b], decoded))                       # global[base_slot] + delta
                pairs.append((tuple(c - g for c, g in zip(client, g64[r])), 1.0))            # aggregated: client - current_global
                up[r][col[p]] = True
                base[r][col[p]] = g_slot[b]
                arrived.append(p)
                bases[str(p)] = g_slot[b]
                if b < r:
                    stale.append(p)
            server.round(pairs)
            publish(r + 1)
            lost_down = 0
            for p in positions:
                # a participant hears the result; a joiner / re-admitted board catches up with the GLOBAL that opens its first round
                listening = (p in attempted) or (r + 1 < R and active_at(members[p], r + 1, R))
                if listening:
                    if (r, p) in ph['downlink_loss']:
                        lost_down += 1
                    else:
                        held[p] = r + 1
                        dn[r][col[p]] = True
            now_active = set(attempted)
            deliveries.append({
                'record': 'round', 'run_id': run_id, 'phase_index': idx, 'round': r + 1, 'seed_slot': g_slot[r], 'result_slot': g_slot[r + 1],
                'participants': sorted(now_active), 'arrived_positions': sorted(arrived), 'stale_positions': sorted(stale),
                'base_slot_by_position': bases,
                'joined_positions': sorted(now_active - prev_active) if r else [], 'left_positions': sorted(prev_active - now_active) if r else [],
                'update_fragments_received': 3 * len(arrived) + len(attempted) - len(arrived), 'update_repair_requests': len(attempted) - len(arrived),
                'update_repair_downlinks_sent': len(attempted) - len(arrived), 'global_fragments_sent': 3 + lost_down,
                'global_repair_rounds': int(lost_down > 0), 'global_bursts': 1 + int(lost_down > 0), 'close_reason': 'complete'})
            prev_active = now_active
        deliveries.append({'record': 'phase_end', 'run_id': run_id, 'phase_index': idx, 'status': 'complete', 'final_slot': g_slot[R]})
        masks_phases[str(idx)] = {
            'phase_index': idx, 'rounds_in_phase': R, 'rounds_completed': R, 'complete': True, 'positions': positions,
            'machines': [data[p][0]['machine_id'] for p in positions],
            'planned_positions': ph['planned_positions'], 'uplink': up, 'downlink': dn, 'base_slot': base}
        if ph['planned_positions'] is None:
            del masks_phases[str(idx)]['planned_positions']
        # board replay on the final model (RUN_CONTROL REPLAY): only boards that hold it
        finals[idx] = {p: g32[R] for p in positions if held[p] == R}
        truth['phases'][idx] = {'codes': [c.copy() for c in g_codes], 'slots': list(g_slot), 'held_final': sorted(finals[idx]),
                                'uplink': up, 'downlink': dn, 'base_slot': base}
    # replay results: one REPLAY_DONE per (position, phase) the board scored, ALARM once per (position, phase) at its first flag
    done_rows = []
    for idx, by_pos in finals.items():
        for p, params in sorted(by_pos.items()):
            flags = cmp.flags_for(params, data[p])
            hit = np.flatnonzero(flags)
            if len(hit):
                alarm_rows.append({'run_id': run_id, 'position': p, 'phase_index': idx, 'acquisition_index': int(hit[0]), 'score': 1.0, 'threshold': 0.5,
                                   'source': 'alarm'})
            done_rows.append({'run_id': run_id, 'position': p, 'phase_index': idx, 'acquisitions_scored': len(flags), 'alarm_latched': int(len(hit) > 0),
                              'first_alarm_index': int(hit[0]) if len(hit) else 0xFFFFFFFF})
    (out / 'replay_masks.json').write_text(json.dumps({'schema': cmp.SCHEMA, 'run_id': run_id, 'seed': seed, 'mode': mode,
                                                       'phases': [masks_phases[k] for k in sorted(masks_phases, key=int)]}, indent=2) + '\n')
    (out / 'global_models.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in models))
    (out / 'delivery_log.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in deliveries))
    for name, fields, rows in (('alarms.csv', ALARM_FIELDS, alarm_rows), ('replay_done.csv', DONE_FIELDS, done_rows)):
        with (out / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    truth['alarm_rows'], truth['done_rows'] = alarm_rows, done_rows
    return truth
