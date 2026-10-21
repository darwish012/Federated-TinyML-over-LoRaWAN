"""Tests of compare_hw_twin_v4 on synthetic PROTOCOL_V4 run directories (synth_v4_run.py).

    python -B -m unittest discover -s analysis -p "test_compare_hw_twin_v4.py"

The fixtures are built from PROTOCOL_V4.md section 4, not from the (parallel) server: a 10-board
planned sweep with losses, an elastic run with joins / leaves / a re-admission, and corrupted
copies that must FAIL.
"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_hw_twin_v4 as cmp
import synth_v4_run as synth


def clone(src, dst, **edits):
    """Copy a run directory; edits = {file name: callable(old text) -> new text}."""
    dst.mkdir(parents=True)
    for p in src.iterdir():
        text = p.read_text(encoding='utf-8')
        if p.name in edits:
            text = edits[p.name](text)
        (dst / p.name).write_text(text, encoding='utf-8')
    return dst


def edit_json(fn):
    def apply(text):
        obj = json.loads(text)
        fn(obj)                                   # mutates in place; the return value is ignored (pop() etc.)
        return json.dumps(obj, indent=2) + '\n'
    return apply


def edit_jsonl(fn):
    def apply(text):
        rows = [json.loads(s) for s in text.splitlines() if s.strip()]
        out = fn(rows)
        rows = rows if out is None else out
        return ''.join(json.dumps(r) + '\n' for r in rows)
    return apply


def edit_csv(fn):
    def apply(text):
        rows = list(csv.DictReader(io.StringIO(text)))
        names = list(rows[0].keys()) if rows else None
        out = fn(rows)
        rows = rows if out is None else out
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=names or [], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
    return apply


def PH(o, index):
    """phase object index of a parsed replay_masks.json (rev 4.1: "phases" is a list)."""
    return next(p for p in o['phases'] if p['phase_index'] == index)


def bump(code, delta):
    return code + delta if code + delta <= 127 else code - delta


class V4ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data, cls.evaluator = cmp.load_data()
        cls.planned_dir = cls.root / 'planned'
        cls.planned_truth = synth.generate_run(cls.planned_dir, synth.planned_scenario(), cls.data, cls.evaluator)
        cls.elastic_dir = cls.root / 'elastic'
        cls.elastic_truth = synth.generate_run(cls.elastic_dir, synth.elastic_scenario(), cls.data, cls.evaluator)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_dir(self):
        type(self).n += 1
        return self.root / ('case%d' % self.n)

    def compare(self, path, **kw):
        return cmp.compare(path, self.data, self.evaluator, **kw)

    def failed(self, report, phase, criterion):
        self.assertFalse(report['pass'])
        self.assertEqual(report['phases'][str(phase)]['criteria'][criterion], 'FAIL', json.dumps(report['phases'][str(phase)]['criteria']))

    # ------------------------------------------------------------------ planned sweep
    def test_planned_ten_board_sweep_with_losses_passes(self):
        report = self.compare(self.planned_dir)
        self.assertTrue(report['pass'], json.dumps({k: v['criteria'] for k, v in report['phases'].items()}))
        self.assertEqual(report['mode'], 'planned')
        self.assertEqual(sorted(report['phases']), [str(i) for i in range(7)])
        stats = [ph['loss_statistics'] for ph in report['phases'].values()]
        self.assertTrue(all(s['missing_uplinks'] > 0 for s in stats[1:]))
        self.assertGreater(sum(s['stale_bases'] for s in stats), 10)
        self.assertEqual(stats[0]['empty_rounds'], 1)
        self.assertGreater(sum(s['update_fragments_received'] for s in stats), 0)
        for i, ph in enumerate(report['phases'].values()):
            self.assertEqual(ph['scope'], list(range(4 * (i + 1))))
            self.assertEqual(ph['rounds_completed'], 12)
            self.assertEqual(len(ph['rounds']), 13)
            self.assertEqual(ph['absent_planned_positions'], [])
        self.assertGreater(sum(ph['first_alarm']['both_alarming'] for ph in report['phases'].values()), 0)
        self.assertGreater(sum(len(ph['no_final_model']) for ph in report['phases'].values()), 0)   # a lost last-round downlink exists
        self.assertTrue(all(r['identical_fraction'] >= .99 for ph in report['phases'].values() for r in ph['rounds']))

    def test_empty_round_leaves_the_global_unchanged(self):
        codes = self.planned_truth['phases'][0]['codes']
        self.assertTrue(np.array_equal(codes[5], codes[6]))          # round 5 delivered nothing: GLOBAL 6 == GLOBAL 5
        self.assertFalse(np.array_equal(codes[4], codes[5]))
        report = self.compare(self.planned_dir)
        self.assertEqual(report['phases']['0']['rounds'][6]['identical_fraction'], 1.0)

    def test_slots_are_run_wide_and_consecutive(self):
        slots = [json.loads(s)['slot'] for s in (self.planned_dir / 'global_models.jsonl').read_text().splitlines()]
        self.assertEqual(slots, list(range(7 * 13)))
        report = self.compare(self.planned_dir)
        self.assertTrue(all(ph['criteria']['slot_integrity'] == 'PASS' for ph in report['phases'].values()))

    def test_report_written_once(self):
        report = self.compare(self.planned_dir)
        target = self.run_dir()
        cmp.write_report(report, target)
        self.assertTrue((target / 'report.json').exists() and (target / 'report.md').exists())
        self.assertIn('Phase 6', (target / 'report.md').read_text())
        with self.assertRaises(FileExistsError):
            cmp.write_report(report, target)

    # ------------------------------------------------------------------ elastic run
    def test_elastic_joins_leaves_and_readmission(self):
        report = self.compare(self.elastic_dir)
        self.assertTrue(report['pass'], json.dumps({k: v['criteria'] for k, v in report['phases'].items()}))
        self.assertEqual(report['mode'], 'elastic')
        p0, p1 = report['phases']['0'], report['phases']['1']
        self.assertEqual(p0['scope'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(p1['scope'], [0, 1, 3, 4, 5, 6, 7])                  # position 2 is not in phase 1
        m0, m1 = p0['membership'], p1['membership']
        self.assertEqual((m0['4']['first_uplink_row'], m0['4']['joined_after_start']), (3, True))
        self.assertEqual(m0['5']['first_uplink_row'], 5)
        self.assertEqual((m0['2']['last_uplink_row'], m0['2']['left_before_end']), (5, True))
        self.assertEqual(m0['1']['gap_rows'], [3, 4])                         # absent, then re-admitted at row 5
        self.assertEqual(m0['1']['uplink_rounds'], 6)
        self.assertEqual(m1['7']['first_uplink_row'], 2)
        self.assertEqual(m1['6']['first_uplink_row'], 3)                      # catch-up downlink lost: joins one round later
        self.assertTrue(m1['3']['left_before_end'])
        self.assertEqual(p0['joins_leaves_logged'], {'joined': [1, 4, 5], 'left': [1, 2]})
        # a position that left holds no final model: listed, not a mismatch
        self.assertEqual(p0['no_final_model'], ['hit::data2'])
        self.assertEqual(p1['no_final_model'], ['ims::R2'])
        self.assertFalse(p0['strict_mismatches'] or p1['strict_mismatches'])
        self.assertEqual(p0['loss_statistics']['participation_basis'], 'delivery_log')
        self.assertGreater(p0['loss_statistics']['stale_bases'], 0)

    def test_elastic_phase_masks_have_absent_positions_before_join_and_after_leave(self):
        masks = PH(json.loads((self.elastic_dir / 'replay_masks.json').read_text()), 0)
        col = {p: i for i, p in enumerate(masks['positions'])}
        up, dn = masks['uplink'], masks['downlink']
        self.assertEqual([r for r in range(8) if up[r][col[4]]], [3, 4, 5, 7])                  # row 6 is a lost uplink
        self.assertTrue(all(not up[r][col[4]] for r in range(3)) and all(not dn[r][col[4]] for r in range(2)) and dn[2][col[4]])
        self.assertTrue(all(not up[r][col[2]] and not dn[r][col[2]] for r in range(6, 8)))
        self.assertTrue(all(masks['base_slot'][r][col[4]] is None for r in range(3)))

    def test_elastic_from_high_slot_start(self):
        run = self.run_dir()
        synth.generate_run(run, synth.elastic_scenario(), self.data, self.evaluator, slot_start=65500)
        report = self.compare(run)
        self.assertTrue(report['pass'])

    def test_phase_index_above_31_is_confirmable_per_phase_replay_done(self):
        scenario = synth.planned_scenario(phase_sizes=(4,), rounds=3)
        scenario['phases'][0]['phase_index'] = 40
        run = self.run_dir()
        synth.generate_run(run, scenario, self.data, self.evaluator)
        report = self.compare(run)
        self.assertTrue(report['pass'], json.dumps(report['phases']['40']['criteria']))      # rev 4.1: REPLAY_DONE carries a u8 phase_index
        self.assertFalse(report['phases']['40']['unfinished'])
        self.assertTrue(any(d['phase_index'] == '40' for d in csv.DictReader((run / 'replay_done.csv').open())))
        gone = clone(run, self.run_dir(), **{'replay_done.csv': edit_csv(lambda rows: rows[1:])})          # drop one (position, phase) row
        self.failed(self.compare(gone), 40, 'strict_outcomes')

    # ------------------------------------------------------------------ corrupted globals
    def last_record_edit(self, phase, rnd, fn):
        def apply(rows):
            for r in rows:
                if r['phase_index'] == phase and r['round'] == rnd:
                    fn(r)
        return edit_jsonl(apply)

    def test_corrupted_final_global_fails_planned(self):
        def corrupt(r):
            r['codes'][0] = bump(r['codes'][0], 2)
        bad = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': self.last_record_edit(5, 12, corrupt)})
        report = self.compare(bad)
        self.failed(report, 5, 'global_codes')
        self.assertEqual(report['phases']['5']['rounds'][12]['differences'][0]['index'], 0)
        self.assertEqual(report['phases']['4']['criteria']['global_codes'], 'PASS')     # other phases untouched
        self.assertEqual(report['phases']['6']['criteria']['global_codes'], 'PASS')

    def test_one_code_off_by_one_is_within_tolerance(self):
        def corrupt(r):
            r['codes'][7] = bump(r['codes'][7], 1)
        ok = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': self.last_record_edit(5, 12, corrupt)})
        report = self.compare(ok)
        self.assertTrue(report['pass'])
        self.assertEqual(len(report['phases']['5']['rounds'][12]['differences']), 1)

    def test_two_percent_of_codes_off_by_one_fails(self):
        def corrupt(r):
            for i in range(20):
                r['codes'][i] = bump(r['codes'][i], 1)
        bad = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': self.last_record_edit(2, 7, corrupt)})
        report = self.compare(bad)
        self.failed(report, 2, 'global_codes')
        self.assertLess(report['phases']['2']['rounds'][7]['identical_fraction'], .99)

    def test_corrupted_global_fails_elastic(self):
        def corrupt(r):
            r['codes'][100] = bump(r['codes'][100], 3)
        bad = clone(self.elastic_dir, self.run_dir(), **{'global_models.jsonl': self.last_record_edit(1, 4, corrupt)})
        report = self.compare(bad)
        self.failed(report, 1, 'global_codes')
        self.assertEqual(report['phases']['0']['criteria']['global_codes'], 'PASS')

    def test_missing_and_duplicate_global_records(self):
        missing = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': edit_jsonl(
            lambda rows: [r for r in rows if not (r['phase_index'] == 3 and r['round'] == 12)])})
        self.failed(self.compare(missing), 3, 'global_codes')
        referenced = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': edit_jsonl(
            lambda rows: [r for r in rows if not (r['phase_index'] == 3 and r['round'] == 4)])})
        with self.assertRaisesRegex(ValueError, 'not a GLOBAL of this phase'):        # a delivered base the log cannot resolve
            self.compare(referenced)
        dup = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': edit_jsonl(
            lambda rows: rows + [dict(rows[20])])})
        report = self.compare(dup)
        self.assertFalse(report['pass'])
        self.assertTrue(any(ph['criteria']['slot_integrity'] == 'FAIL' for ph in report['phases'].values()))

    def test_slot_reuse_across_phases_fails_slot_integrity(self):
        def reuse(rows):
            for r in rows:
                if r['phase_index'] == 1 and r['round'] == 0:
                    r['slot'] = 5                                           # a slot already used in phase 0
        bad = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': edit_jsonl(reuse)})
        with self.assertRaisesRegex(ValueError, 'not a GLOBAL of this phase'):      # the phase-1 base slots no longer resolve
            self.compare(bad)

    # ------------------------------------------------------------------ malformed input (exit 2)
    def test_mixed_run_ids_and_schema(self):
        a = clone(self.planned_dir, self.run_dir(), **{'global_models.jsonl': edit_jsonl(lambda rows: rows[-1].update(run_id=1))})
        with self.assertRaisesRegex(ValueError, 'mixed run IDs'):
            self.compare(a)
        b = clone(self.planned_dir, self.run_dir(), **{'alarms.csv': edit_csv(lambda rows: rows[0].update(run_id='7'))})
        with self.assertRaisesRegex(ValueError, 'mixed run IDs'):
            self.compare(b)
        c = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(lambda o: o.update(schema='tinyfl-replay-masks-v2'))})
        with self.assertRaisesRegex(ValueError, 'schema'):
            self.compare(c)
        d = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(lambda o: o.pop('mode'))})
        with self.assertRaisesRegex(ValueError, 'mode'):
            self.compare(d)

    def test_invalid_base_slots_raise(self):
        def with_slot(value):
            def apply(o):
                ph = PH(o, 2)
                r, i = next((r, i) for r in range(12) for i in range(len(ph['positions'])) if ph['uplink'][r][i])
                ph['base_slot'][r][i] = value
            return edit_json(apply)
        slot_of_other_phase = 3                                             # a GLOBAL of phase 0
        for value, text in ((slot_of_other_phase, 'not a GLOBAL of this phase'), (60000, 'not a GLOBAL of this phase')):
            run = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': with_slot(value)})
            with self.assertRaisesRegex(ValueError, text):
                self.compare(run)
        # a GLOBAL of this phase that does not exist yet at that row (round 12 slot at row 0)
        late = 2 * 13 + 12

        def at_row0(o):
            ph = PH(o, 2)
            ph['base_slot'][0][next(i for i, v in enumerate(ph['uplink'][0]) if v)] = late
        run = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(at_row0)})
        with self.assertRaisesRegex(ValueError, 'invalid delivered base slot'):
            self.compare(run)

    def test_invalid_mask_shapes_and_positions(self):
        cases = {
            'null slot where uplink': lambda o: PH(o, 1)['base_slot'][0].__setitem__(next(i for i, v in enumerate(PH(o, 1)['uplink'][0]) if v), None),
            'shape': lambda o: PH(o, 1)['uplink'][3].pop(),
            'rows': lambda o: PH(o, 1)['downlink'].pop(),
            'position': lambda o: PH(o, 1)['positions'].__setitem__(0, 41),
            'duplicate position': lambda o: PH(o, 1)['positions'].__setitem__(1, 0),
            'machine order': lambda o: PH(o, 1).__setitem__('machines', list(reversed(PH(o, 1)['machines']))),
            'rounds_completed': lambda o: PH(o, 1).__setitem__('rounds_completed', 11),
        }
        for label, fn in cases.items():
            run = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(fn)})
            with self.assertRaises(ValueError, msg=label):
                self.compare(run)

    # ------------------------------------------------------------------ hardware replay results
    def test_unfinished_replay_fails_strict_outcomes(self):
        def drop_row(rows):
            return [r for r in rows if not (int(r['position']) == 1 and int(r['phase_index']) == 3)]      # one (position, phase)
        run = clone(self.planned_dir, self.run_dir(), **{'replay_done.csv': edit_csv(drop_row)})
        report = self.compare(run)
        self.failed(report, 3, 'strict_outcomes')
        self.assertTrue(report['phases']['3']['unfinished'])
        self.assertEqual(report['phases']['2']['criteria']['strict_outcomes'], 'PASS')

    def test_replay_done_for_a_position_without_the_final_model_is_inconsistent(self):
        def claim(rows):
            rows.append(dict(rows[0], position='2', phase_index='0', alarm_latched='0'))     # position 2 left: no final model
        run = clone(self.elastic_dir, self.run_dir(), **{'replay_done.csv': edit_csv(claim)})
        report = self.compare(run)
        self.failed(report, 0, 'strict_outcomes')
        self.assertEqual(report['phases']['0']['inconsistent_replay_done'], ['hit::data2'])

    def test_shifted_first_alarms_fail_timing(self):
        base = self.compare(self.planned_dir)
        self.assertGreater(sum(ph['first_alarm']['both_alarming'] for ph in base['phases'].values()), 3)
        # ALARM and REPLAY_DONE agree (3.3) but both are 5 acquisitions late
        run = clone(self.planned_dir, self.run_dir(), **{
            'alarms.csv': edit_csv(lambda rows: [dict(r, acquisition_index=str(int(r['acquisition_index']) + 5)) for r in rows]),
            'replay_done.csv': edit_csv(lambda rows: [dict(r, first_alarm_index=str(int(r['first_alarm_index']) + 5))
                                                      if int(r['alarm_latched']) else r for r in rows])})
        report = self.compare(run)
        self.assertFalse(report['pass'])
        self.assertTrue(any(ph['criteria']['first_alarm'] == 'FAIL' for ph in report['phases'].values()))
        self.assertFalse(any(ph['inconsistent_replay_done'] for ph in report['phases'].values()))

    def test_one_alarm_within_one_acquisition_is_accepted(self):
        victim = list(csv.DictReader((self.planned_dir / 'alarms.csv').open()))[0]
        key = (victim['position'], victim['phase_index'])
        run = clone(self.planned_dir, self.run_dir(), **{
            'alarms.csv': edit_csv(lambda rows: [dict(r, acquisition_index=str(int(r['acquisition_index']) + (1 if k == 0 else 0))) for k, r in enumerate(rows)]),
            'replay_done.csv': edit_csv(lambda rows: [dict(r, first_alarm_index=str(int(r['first_alarm_index']) + 1))
                                                      if (r['position'], r['phase_index']) == key else r for r in rows])})
        report = self.compare(run)
        self.assertTrue(report['pass'], json.dumps({k: v['criteria'] for k, v in report['phases'].items()}))

    def test_alarm_index_that_differs_from_the_replay_done_index_is_inconsistent(self):
        # rev 4.3, 3.3: the ALARM's acquisition_index must equal REPLAY_DONE.first_alarm_index
        run = clone(self.planned_dir, self.run_dir(), **{'alarms.csv': edit_csv(
            lambda rows: [dict(r, acquisition_index=str(int(r['acquisition_index']) + (1 if k == 0 else 0))) for k, r in enumerate(rows)])})
        victim = list(csv.DictReader((self.planned_dir / 'alarms.csv').open()))[0]
        report = self.compare(run)
        self.failed(report, int(victim['phase_index']), 'strict_outcomes')
        self.assertTrue(report['phases'][victim['phase_index']]['inconsistent_replay_done'])

    def test_a_replay_done_sourced_alarm_counts_for_first_alarm_timing(self):
        # the ALARM message was lost: the server wrote the alarm from REPLAY_DONE (source=replay_done, no score)
        def from_done(rows):
            for r in rows:
                r['source'] = 'replay_done'
                r['score'] = r['threshold'] = ''
        run = clone(self.planned_dir, self.run_dir(), **{'alarms.csv': edit_csv(from_done)})
        report = self.compare(run)
        self.assertTrue(report['pass'], json.dumps({k: v['criteria'] for k, v in report['phases'].items()}))
        self.assertGreater(sum(ph['first_alarm']['both_alarming'] for ph in report['phases'].values()), 3)
        # shifted by 5 it fails timing exactly as an ALARM-sourced alarm would
        both = clone(self.planned_dir, self.run_dir(), **{
            'alarms.csv': edit_csv(lambda rows: [dict(r, source='replay_done', score='', threshold='',
                                                      acquisition_index=str(int(r['acquisition_index']) + 5)) for r in rows]),
            'replay_done.csv': edit_csv(lambda rows: [dict(r, first_alarm_index=str(int(r['first_alarm_index']) + 5))
                                                      if int(r['alarm_latched']) else r for r in rows])})
        self.assertTrue(any(ph['criteria']['first_alarm'] == 'FAIL' for ph in self.compare(both)['phases'].values()))

    def test_replay_done_is_per_phase_not_per_position(self):
        # a position's other phases stay confirmed when one phase row disappears
        run = clone(self.planned_dir, self.run_dir(), **{'replay_done.csv': edit_csv(
            lambda rows: [r for r in rows if not (int(r['position']) == 0 and int(r['phase_index']) == 6)])})
        report = self.compare(run)
        self.failed(report, 6, 'strict_outcomes')
        self.assertTrue(all(report['phases'][str(i)]['criteria']['strict_outcomes'] == 'PASS' for i in range(6)))
        self.assertEqual(report['phases']['6']['unfinished'], [report['phases']['6']['machines'][0]['machine']])

    def test_latched_alarm_without_message_and_message_without_latch(self):
        base = list(csv.DictReader((self.planned_dir / 'alarms.csv').open()))
        victim = base[0]
        no_msg = clone(self.planned_dir, self.run_dir(), **{'alarms.csv': edit_csv(lambda rows: rows[1:])})
        report = self.compare(no_msg)                                      # rev 4.3: REPLAY_DONE carries the index, nothing is lost
        self.assertTrue(report['pass'], json.dumps({k: v['criteria'] for k, v in report['phases'].items()}))
        # a rev 4.2 style REPLAY_DONE (no index) with the ALARM lost is still ambiguous and fails closed
        no_index = clone(self.planned_dir, self.run_dir(), **{
            'alarms.csv': edit_csv(lambda rows: rows[1:]),
            'replay_done.csv': edit_csv(lambda rows: [dict(r, first_alarm_index='') for r in rows])})
        report = self.compare(no_index)
        self.failed(report, int(victim['phase_index']), 'strict_outcomes')
        self.assertIn(True, [m.get('finished') and m.get('hardware_strict') is None for m in report['phases'][victim['phase_index']]['machines']])
        no_latch = clone(self.planned_dir, self.run_dir(), **{'replay_done.csv': edit_csv(
            lambda rows: [dict(r, alarm_latched='0') if (r['position'], r['phase_index']) == (victim['position'], victim['phase_index']) else r for r in rows])})
        report = self.compare(no_latch)
        self.failed(report, int(victim['phase_index']), 'strict_outcomes')
        self.assertTrue(report['phases'][victim['phase_index']]['inconsistent_replay_done'])

    def test_joiner_must_start_from_its_seed(self):
        # PROTOCOL_V4 4 Mid-phase join: position 4 joins at row 3 and trains from that round's seed GLOBAL.
        def stale(o):
            ph = PH(o, 0)
            col = ph['positions'].index(4)
            ph['base_slot'][3][col] = ph['base_slot'][3][col] - 1     # an older GLOBAL of the phase
        run = clone(self.elastic_dir, self.run_dir(), **{'replay_masks.json': edit_json(stale)})
        report = self.compare(run)
        self.failed(report, 0, 'membership')
        self.assertTrue(any('not from that round' in m for m in report['phases']['0']['membership_problems']))

    def test_phases_as_keyed_object_is_also_accepted(self):
        run = clone(self.elastic_dir, self.run_dir(), **{'replay_masks.json': edit_json(
            lambda o: o.__setitem__('phases', {str(p['phase_index']): p for p in o['phases']}))})
        self.assertTrue(self.compare(run)['pass'])

    def test_missing_alarms_fail_outcomes_closed(self):
        run = clone(self.planned_dir, self.run_dir(), **{
            'alarms.csv': edit_csv(lambda rows: []),
            'replay_done.csv': edit_csv(lambda rows: [dict(r, first_alarm_index='') for r in rows])})
        report = self.compare(run)
        self.assertFalse(report['pass'])
        self.assertTrue(any(ph['criteria']['strict_outcomes'] == 'FAIL' for ph in report['phases'].values()))

    # ------------------------------------------------------------------ planned coverage, completion, membership
    def test_planned_position_without_a_board_is_reported(self):
        scenario = synth.planned_scenario(phase_sizes=(4,), rounds=12)
        scenario['phases'][0]['members'][3] = synth.member(join=99)       # planned, never delivers
        run = self.run_dir()
        synth.generate_run(run, scenario, self.data, self.evaluator)
        report = self.compare(run)
        self.failed(report, 0, 'planned_coverage')
        ph = report['phases']['0']
        self.assertEqual(ph['absent_planned_positions'], [3])
        self.assertEqual([m['reason'] for m in ph['machines'] if m['position'] == 3], ['never_participated'])
        allowed = self.compare(run, allow_absent=True)
        self.assertTrue(allowed['pass'])

    def test_unplanned_position_delivering_fails_membership(self):
        run = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(
            lambda o: PH(o, 1).__setitem__('planned_positions', list(range(7))))})
        report = self.compare(run)
        self.failed(report, 1, 'membership')

    def test_incomplete_phase_fails_completion(self):
        def truncate(o):
            ph = PH(o, 6)
            for key in ('uplink', 'downlink', 'base_slot'):
                ph[key] = ph[key][:8]
            ph['rounds_completed'], ph['complete'] = 8, False
        run = clone(self.planned_dir, self.run_dir(), **{'replay_masks.json': edit_json(truncate)})
        report = self.compare(run)
        self.failed(report, 6, 'round_completion')
        self.assertEqual(report['phases']['5']['criteria']['round_completion'], 'PASS')

    def test_delivery_log_cross_checks(self):
        def edit_round(phase, rnd, fn):
            def apply(rows):
                for r in rows:
                    if r.get('record') == 'round' and r['phase_index'] == phase and r['round'] == rnd:
                        fn(r)
            return edit_jsonl(apply)
        cases = {
            'arrived': lambda r: r['arrived_positions'].pop(),
            'participants': lambda r: r.update(participants=[p for p in r['participants'] if p != r['arrived_positions'][0]]),
            'stale': lambda r: r.update(stale_positions=sorted(set(r['stale_positions']) | {r['arrived_positions'][0]})),
            'joined': lambda r: r.update(joined_positions=[r['participants'][0]]),
            'left': lambda r: r.update(left_positions=[r['participants'][0]]),
        }
        for label, fn in cases.items():
            run = clone(self.elastic_dir, self.run_dir(), **{'delivery_log.jsonl': edit_round(0, 6, fn)})
            report = self.compare(run)
            self.failed(report, 0, 'membership')
            self.assertTrue(report['phases']['0']['membership_problems'], label)
        dropped = clone(self.elastic_dir, self.run_dir(), **{'delivery_log.jsonl': edit_jsonl(
            lambda rows: [r for r in rows if not (r.get('record') == 'round' and r['phase_index'] == 1 and r['round'] == 3)])})
        report = self.compare(dropped)
        self.failed(report, 1, 'round_completion')
        self.failed(report, 1, 'membership')

    def test_base_never_held_fails_membership(self):
        def drop_downlink(o):
            ph = PH(o, 0)
            col = ph['positions'].index(4)
            ph['downlink'][2][col] = False                    # position 4 trains from GLOBAL 3 at row 3 but 'never received' it
        run = clone(self.elastic_dir, self.run_dir(), **{'replay_masks.json': edit_json(drop_downlink)})
        report = self.compare(run)
        self.failed(report, 0, 'membership')

    # ------------------------------------------------------------------ module facts
    def test_firmware_arithmetic_vs_twin_codec(self):
        rng = np.random.default_rng(5)
        delta = tuple(rng.normal(size=s).astype(np.float32) * np.float32(10.0 ** -k) for s, k in zip(cmp.twin.SHAPES, (2, 3, 2, 3)))
        codes, scales = synth.fw_quantize_update(delta)
        codes_list, scales_t, _ = cmp.twin.per_machine_uplink_message(delta, tuple(np.zeros_like(d) for d in delta))
        theirs = cmp.twin.all_codes_flat(codes_list)
        self.assertGreaterEqual(np.mean(codes == theirs), .99)
        self.assertLessEqual(np.max(np.abs(codes.astype(int) - theirs)), 1)
        for a, b in zip(scales, scales_t):
            self.assertAlmostEqual(float(a) / b, 1.0, places=6)
        # the half-way cases the two codecs may legitimately disagree on do exist only at float32 resolution
        self.assertTrue(np.array_equal(synth.fw_roundf(np.array([0.5, -0.5, 1.5, 2.5, -2.5, 0.49999997], dtype=np.float32)), np.array([1, -1, 2, 3, -3, 0], dtype=np.float32)))

    def test_predeclared_criteria_text_is_checked(self):
        crit = cmp.predeclared_criteria()
        self.assertEqual((crit['identical_fraction'], crit['max_code_delta'], crit['max_strict_mismatches'], crit['first_alarm_fraction']), (.99, 1, 1, .95))
        spec = json.loads((cmp.TWIN / 'protocol.json').read_text())
        spec['criteria_fixed_now']['hardware_vs_twin (applied later, same file)']['global_model'] = 'changed'
        with mock.patch.object(cmp, 'read', return_value=spec):
            with self.assertRaisesRegex(ValueError, 'no longer says'):
                cmp.predeclared_criteria()

    def test_fields_expected_lists_every_file_read(self):
        text = json.dumps(cmp.FIELDS_EXPECTED)
        for name in ('replay_masks.json', 'global_models.jsonl', 'delivery_log.jsonl', 'alarms.csv', 'replay_done.csv', 'base_slot', 'alarm_latched', 'phase_index'):
            self.assertIn(name, text)

    def test_lazy_data_only_serves_hardware_eligible_positions(self):
        self.assertIn(0, self.data)
        self.assertIn(40, self.data)
        self.assertNotIn(41, self.data)
        self.assertNotIn(-1, self.data)
        with self.assertRaises(KeyError):
            self.data[41]

    # ------------------------------------------------------------------ CLI
    def test_cli_exit_codes(self):
        def run(args):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(sys, 'argv', ['compare_hw_twin_v4.py'] + [str(a) for a in args]), \
                    mock.patch.object(cmp, 'load_data', return_value=(self.data, self.evaluator)):
                try:
                    code = cmp.main()
                except SystemExit as exit_:
                    code = exit_.code
            return code, out.getvalue()
        code, out = run([self.elastic_dir, '--output', self.run_dir()])
        self.assertEqual((code, out.strip()), (0, 'PASS'))
        bad = clone(self.elastic_dir, self.run_dir(), **{'global_models.jsonl': self.last_record_edit(0, 8, lambda r: r['codes'].__setitem__(0, bump(r['codes'][0], 4)))})
        code, out = run([bad, '--output', self.run_dir()])
        self.assertEqual((code, out.strip()), (1, 'FAIL'))
        code, _ = run([self.root / 'does_not_exist', '--output', self.run_dir()])
        self.assertEqual(code, 2)
        code, out = run(['--fields'])
        self.assertEqual(code, 0)
        self.assertIn('replay_masks.json', out)


if __name__ == '__main__':
    unittest.main()
