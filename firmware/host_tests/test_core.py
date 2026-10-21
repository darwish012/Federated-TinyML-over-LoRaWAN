"""Read-only reference comparisons. Generated cards live in TemporaryDirectory.
Run: python test_core.py --report REPORT.json
Never touches the server, twin sources, or exported card files.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import ctypes as C
import json
from pathlib import Path
import shutil
import struct
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / 'autoencoder_cwru_release/full_pipeline_simulation'
sys.path[:0] = [str(ROOT / 'lorawan_two_node_experiment/server'),
               str(SIM / 'research/hardware_twin_scaling_20260929'),
               str(SIM / 'research/healthy_only_causal_20260908')]
import protocol_v3 as wire
import raw_features_f32 as rf
import twin
from cadence import dense_flags

lib = C.CDLL(str(HERE / 'tfcore.dll'))
for name in ('tfc_encode_update', 'tfc_update_fragment', 'tfc_encode_global',
             'tfc_encode_hello', 'tfc_encode_alarm', 'tfc_encode_self_check',
             'tfc_encode_replay_done', 'tfc_encode_ready', 'tfc_encode_status',
             'tfc_heap_peak', 'tfc_heap_current', 'tfc_encode_repair', 'tfc_global_fragment'):
    getattr(lib, name).restype = C.c_size_t
lib.tfc_last_error.restype = C.c_char_p
lib.tfc_mean_loss.restype = C.c_float
lib.tfc_init()
metrics = {}
def ptr(a): return C.c_void_p(a.ctypes.data)
def buf(n): return np.zeros(n, dtype=np.uint8)
def encoded(name, *args):
    out = buf(594)
    n = getattr(lib, name)(ptr(out), *args)
    return bytes(out[:n])
def check(ok):
    if not ok: raise AssertionError(lib.tfc_last_error().decode())
def asbuf(data): return np.frombuffer(data, np.uint8).copy()
def raw_header(p):
    with p.open('rb') as f: h = f.read(66)
    vals = struct.unpack('<4sHI4H12I', h)
    return dict(hbytes=vals[2], pos=vals[3], S=vals[5], C=vals[6], n=vals[7],
                fixed=vals[8], parts=vals[9:])
def acquisitions(p, limit=None):
    h = raw_header(p)
    with p.open('rb') as f:
        f.seek(66)
        lengths = [h['fixed']] * h['n'] if h['fixed'] else np.frombuffer(f.read(4*h['n']), '<u4')
        f.seek(h['hbytes'])
        for i, n in enumerate(lengths):
            if limit is not None and i >= limit: break
            q = np.empty((h['C'], int(n)), dtype=np.int16)
            scales = np.empty(h['C'], dtype=np.float32)
            for c in range(h['C']):
                scales[c] = struct.unpack('<f', f.read(4))[0]
                q[c] = np.frombuffer(f.read(2*int(n)), '<i2')
            yield q, scales
def extract(q, scales):
    q = np.ascontiguousarray(q, np.int16)
    scales = np.ascontiguousarray(scales, np.float32)
    out = np.empty(22, np.float32)
    check(lib.tfc_extract(ptr(q), ptr(scales), len(scales), q.shape[1], ptr(out)))
    return out
def crc(data): return data + struct.pack('<I', zlib.crc32(data))
def make_card(root, variable=False):
    rng = np.random.default_rng(5)
    lengths = [512, 514] if variable else [512, 512]
    head = struct.pack('<4sHI4H12I', b'TFR1', 1, 70 + (8 if variable else 0),
                       0, 0, 1, 1, 2, 0 if variable else 512,
                       0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    if variable: head += struct.pack('<2I', *lengths)
    data = b''; rows = []
    for n in lengths:
        q = rng.integers(-30000, 30000, n, dtype=np.int16)
        s = np.float32(0.001)
        data += struct.pack('<f', s) + q.astype('<i2').tobytes()
        rows.append(rf.acquisition_22_f32((q*s)[None, :]))
    (root/'m00s0.raw').write_bytes(crc(head)+crc(data))
    ch = struct.pack('<4sHHIHH', b'TFC1', 1, 0, 2, 1, 0)
    (root/'m00.chk').write_bytes(crc(ch+np.asarray(rows, '<f4').tobytes()))

class CoreTests(unittest.TestCase):
    def tearDown(self): lib.tfc_extractor_release()

    def test_01_synthetic_features(self):
        rng = np.random.default_rng(123)
        errors = []
        for n in (512, 514, 1000, 1024, 1601, 16001, 16006, 16129, 16384, 49999):
            lib.tfc_extractor_release(); lib.tfc_heap_reset_peak()
            t = np.arange(n)
            signals = [np.zeros(n), np.ones(n)*700,
                       15000*np.sin(2*np.pi*17*t/n), rng.normal(0, 3000, n)]
            q = np.clip(np.asarray(signals), -32767, 32767).astype(np.int16)
            s = np.array([1, .001, .0002, .001], np.float32)
            actual = extract(q, s)
            expected = rf.acquisition_22_f32(q.astype(np.float32)*s[:, None])
            err = float(np.max(np.abs(actual-expected)))
            per_channel_errors=[]
            for c in range(len(s)):
                out=np.empty(11,np.float32)
                check(lib.tfc_channel_descriptors(ptr(q[c]),n,C.c_float(s[c]),ptr(out)))
                expected_channel=rf.channel_descriptors_11_f32(q[c].astype(np.float32)*s[c])
                cerr=float(np.max(abs(out-expected_channel))); per_channel_errors.append(cerr)
                self.assertLessEqual(cerr,1e-3)
            errors.append({'samples': n, 'max_abs': err, 'channel_max_abs': max(per_channel_errors),
                           'core_peak_bytes': int(lib.tfc_heap_peak())})
            self.assertLessEqual(err, 1e-3, str(n))
        metrics['synthetic_features'] = errors
        metrics['fft_core_peak_bytes'] = max(v['core_peak_bytes'] for v in errors)

    def test_02_real_features_and_training(self):
        card = SIM/'results/hardware_twin_scaling_20260929/sd_cards_v2/board_03'
        raw = card/'m02s0.raw'; h = raw_header(raw)
        ref = np.frombuffer((card/'m02.chk').read_bytes()[16:-4], '<f4').reshape(-1, 22)
        actual = np.stack([extract(q, s) for q, s in acquisitions(raw, 50)])
        expected = np.stack([rf.acquisition_22_f32(q.astype(np.float32)*s[:, None])
                             for q, s in acquisitions(raw, 50)])
        chkerr = float(np.max(np.abs(actual-ref[:50])))
        pyerr = float(np.max(np.abs(actual-expected)))
        self.assertLessEqual(chkerr, 1e-3); self.assertLessEqual(pyerr, 1e-3)
        metrics['real_first50'] = {'position': 2, 'acquisitions': 50,
                                  'channels_per_acquisition': h['C'],
                                  'chk_max_abs': chkerr, 'python_max_abs': pyerr}
        ntrain, ncal = h['parts'][1], h['parts'][3]
        train = np.ascontiguousarray(ref[:ntrain], np.float32)
        centre = np.empty(22, np.float32); scale = np.empty(22, np.float32)
        check(lib.tfc_std_params(ptr(train), ntrain, ptr(centre), ptr(scale)))
        pc, ps = twin.stream_standardisation_params(train)
        np.testing.assert_allclose(centre, pc, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(scale, ps, rtol=2e-6, atol=2e-6)
        X = np.stack([self.std_row(r, centre, scale) for r in train])
        px = twin.apply_standardisation(train, pc, ps)
        np.testing.assert_allclose(X, px, rtol=3e-6, atol=3e-6)
        metrics['standardisation'] = {'centre_max_abs': float(np.max(abs(centre-pc))),
                                     'scale_max_abs': float(np.max(abs(scale-ps))),
                                     'rows_max_abs': float(np.max(abs(X-px)))}
        rng = np.random.default_rng(77)
        w = rng.normal(0, .1, 562).astype(np.float32)
        p = twin.unflatten(w.copy()); maxerr = 0; agreement = []
        for r in range(12):
            before = w.copy(); pbefore = twin.flatten(p)
            lib.tfc_local_train(ptr(X), ntrain, ptr(w))
            p = twin.local_train(px, p)
            err = float(np.max(abs(w-twin.flatten(p)))); maxerr = max(maxerr, err)
            np.testing.assert_allclose(w, twin.flatten(p), rtol=1e-4, atol=3e-6)
            codes = np.empty(562, np.int8); scales = np.empty(4, np.float32)
            delta = w-before
            lib.tfc_quantise(ptr(delta), ptr(codes), ptr(scales))
            qref = wire.quantize_model((twin.flatten(p)-pbefore).astype(np.float32))
            agreement.append(float(np.mean(codes == qref.codes)))
        self.assertGreaterEqual(min(agreement), .99)
        metrics['training_12_rounds'] = {'n_train': ntrain, 'epochs_each': 2,
                                       'max_weight_abs_error': maxerr,
                                       'min_delta_code_agreement': min(agreement)}
        # Real score vector: all acquisitions of this real machine, features
        # recomputed from the exported raw data (not synthetic alarm inputs).
        allraw = np.stack([rf.acquisition_22_f32(q.astype(np.float32)*s[:, None])
                           for q, s in acquisitions(raw)])
        allx = twin.apply_standardisation(allraw, pc, ps)
        real_scores = twin.score(p, allx).astype(np.float32)
        self.compare_alarm(real_scores, ntrain, ncal)
        with tempfile.TemporaryDirectory(prefix='tinyfl-test-') as d:
            root = Path(d)
            shutil.copyfile(raw, root/raw.name); shutil.copyfile(card/'m02.chk', root/'m02.chk')
            cs = np.empty(4*22, np.float32); ss = np.empty_like(cs); check3 = np.empty(3, np.float32)
            ns = C.c_int(); nt = C.c_int()
            check(lib.tfc_prepare(str(root).encode(), 2, C.byref(ns), C.byref(nt), ptr(check3), ptr(cs), ptr(ss)))
            self.assertLessEqual(check3[1], 1e-3)
            xload = np.empty((nt.value,22),np.float32)
            check(lib.tfc_load_x(str(root).encode(), 2, ptr(xload), nt.value, C.byref(nt)))
            np.testing.assert_allclose(xload, px, rtol=1e-4, atol=1e-4)
            metrics['complete_prepare'] = {'streams': ns.value, 'train_rows': nt.value,
                                          'self_check_rows': int(check3[0]), 'max_abs': float(check3[1])}
            modelset=np.stack([w,w*np.float32(.8),w*np.float32(1.2)]).astype(np.float32)
            levels=np.array([0,2,6],np.int32); n=h['n']; nm=len(levels)
            scores=np.empty((n,nm),np.float32); flags=np.empty((n,nm),np.uint8); stats=np.empty((n,nm),np.float32)
            thrs=np.empty(nm,np.float64); firsts=np.empty(nm,np.int64)
            checked=C.c_int(); bad=C.c_int(); nout=C.c_int()
            check(lib.tfc_replay(str(root).encode(),2,nm,ptr(modelset),ptr(levels),n,ptr(scores),ptr(flags),ptr(stats),
                                 ptr(thrs),ptr(firsts),C.byref(checked),C.byref(bad),C.byref(nout)))
            maxscore=0
            for i in range(nm):
                expected_scores=twin.score(twin.unflatten(modelset[i]),allx)
                maxscore=max(maxscore,float(np.max(abs(scores[:,i]-expected_scores))))
                np.testing.assert_allclose(scores[:,i],expected_scores,rtol=3e-5,atol=3e-5)
                expected_flags=dense_flags(scores[:,i],np.arange(ntrain),np.arange(ntrain,ntrain+ncal))
                self.assertTrue(np.array_equal(flags[:,i].astype(bool),expected_flags))
            self.assertEqual((checked.value,bad.value,nout.value),(1,0,n))
            metrics['full_real_replay']={'acquisitions':n,'models':nm,'identical_flags':n*nm,
                                         'score_max_abs':maxscore,'crc_checked':checked.value,'crc_bad':bad.value}

    def std_row(self, row, c, s):
        out = np.empty(22, np.float32); lib.tfc_std_apply(ptr(row),ptr(c),ptr(s),ptr(out)); return out

    def compare_alarm(self, scores, nt, nc):
        n = len(scores); flags = buf(n); stats = np.empty(n, np.float32)
        thr = C.c_double(); first = C.c_int64(); events = C.c_int()
        lib.tfc_alarm(ptr(scores), n, 0, nt, nt, nc, ptr(flags), ptr(stats),
                      C.byref(thr), C.byref(first), C.byref(events))
        expected = dense_flags(scores, np.arange(nt), np.arange(nt,nt+nc))
        self.assertTrue(np.array_equal(flags.astype(bool), expected))
        self.assertEqual(events.value, int(np.any(expected)))
        metrics['real_alarm'] = {'scores': n, 'identical_flags': n,
                                 'threshold': thr.value, 'first_alarm': first.value,
                                 'latched_events': events.value}

    def test_03_codec(self):
        rng = np.random.default_rng(31); w = rng.normal(0,.3,562).astype(np.float32)
        q = wire.quantize_model(w); codes = np.empty(562, np.int8); scales = np.empty(4,np.float32)
        lib.tfc_quantise(ptr(w),ptr(codes),ptr(scales))
        self.assertTrue(np.array_equal(codes,q.codes))
        np.testing.assert_array_equal(scales,np.array(q.wire_scales(),np.float32))
        u = encoded('tfc_encode_update',2,6,11,321,88,997,ptr(scales),ptr(codes),C.c_float(.125))
        g = encoded('tfc_encode_global',6,12,321,ptr(scales),ptr(codes))
        self.assertEqual(u,wire.encode_update(2,6,11,997,q,.125,run_id=321,base_slot=88))
        self.assertEqual(g,wire.encode_global(6,12,q,run_id=321))
        self.assertEqual(wire.decode_update(u).base_slot,88)
        self.assertEqual(wire.decode_global(g).run_id,321)
        ints = [C.c_int() for _ in range(6)]; loss=C.c_float(); ub=asbuf(u); gb=asbuf(g)
        check(lib.tfc_decode_update(ptr(ub),*[C.byref(i) for i in ints],ptr(scales),ptr(codes),C.byref(loss)))
        self.assertEqual([i.value for i in ints],[2,6,11,321,88,997])
        check(lib.tfc_decode_global(ptr(gb),*[C.byref(i) for i in ints[:3]],ptr(scales),ptr(codes)))
        self.assertEqual([i.value for i in ints[:3]],[6,12,321])
        for i, frame in enumerate(wire.fragment_update(u,90)):
            self.assertEqual(frame,encoded('tfc_update_fragment',ptr(ub),90,i))
        self.assertEqual(encoded('tfc_update_fragment',ptr(ub),90,3),b'')
        globals = wire.fragment_payload(g,wire.MSG_GLOBAL_FRAGMENT,90)
        for i,f in enumerate(globals): self.assertEqual(f,encoded('tfc_global_fragment',ptr(gb),90,i))
        frames = asbuf(b''.join([globals[2],globals[0],globals[0],globals[1]]))
        lens=np.array([len(globals[i]) for i in (2,0,0,1)],np.int32); out=buf(586); rs=np.empty(4,np.int32)
        self.assertEqual(lib.tfc_reassemble(ptr(frames),ptr(lens),4,ptr(out),ptr(rs)),7)
        self.assertEqual(bytes(out),g)
        corrupt=bytearray(globals[0]); corrupt[80]^=1
        frames=asbuf(globals[0]+bytes(corrupt)+globals[2]+globals[1])
        lens=np.array([len(globals[i]) for i in (0,0,2,1)],np.int32)
        self.assertEqual(lib.tfc_reassemble(ptr(frames),ptr(lens),4,ptr(out),ptr(rs)),7)
        self.assertEqual(rs[1],0); self.assertEqual(bytes(out),g)
        controls = [
            ('tfc_encode_hello',(3,3,32,3,321),wire.encode_board_hello(3,3,32,3,321),wire.decode_board_hello),
            ('tfc_encode_alarm',(2,6,600,C.c_float(9),C.c_float(7)),wire.encode_alarm(2,6,600,9,7),wire.decode_alarm),
            ('tfc_encode_self_check',(2,360,C.c_float(.0001),C.c_float(.00001)),wire.encode_self_check(2,360,np.float32(.0001),np.float32(.00001)),wire.decode_self_check),
            ('tfc_encode_replay_done',(2,127,450,1),wire.encode_replay_done(2,127,450,1),wire.decode_replay_done),
            ('tfc_encode_ready',(90,5,5),wire.encode_class_c_ready(90,5,5),None),
            ('tfc_encode_status',(90,5),wire.encode_global_status(90,5),wire.decode_global_status)]
        for name,args,expected,decode in controls:
            got=encoded(name,*args); self.assertEqual(got,expected)
            f=wire.decode_frame(got)
            if decode: decode(f)
            hbuf=buf(5); body=buf(214); source=asbuf(expected)
            check(lib.tfc_parse_frame(ptr(source),len(source),ptr(hbuf),ptr(body)))
            self.assertEqual(bytes(body[:len(f.data)]),f.data)
            self.assertEqual(list(hbuf),[f.message_type,f.round_id,f.fragment_index,f.fragment_count,len(f.data)])
        repair=wire.encode_repair_request(90,2,5); rb=asbuf(repair)
        self.assertEqual(repair,encoded('tfc_encode_repair',90,2,5))
        check(lib.tfc_parse_repair(ptr(rb),len(rb),*[C.byref(i) for i in ints[:3]]))
        self.assertEqual([i.value for i in ints[:3]],[90,5,2])
        self.assertEqual(wire.decode_repair_request(wire.decode_frame(repair)),(5,2))
        metrics['codec']={'message_types':9,'quant_codes_identical':562,
                          'update_bytes':len(u),'global_bytes':len(g),
                          'update_frame_bytes':[len(f) for f in wire.fragment_update(u,90)],
                          'global_frame_bytes':[len(f) for f in globals]}

    def test_04_validation(self):
        count=0
        with tempfile.TemporaryDirectory(prefix='tinyfl-invalid-') as d:
            root=Path(d)
            for variable in (False,True):
                make_card(root,variable)
                path=root/'m00s0.raw'; good=path.read_bytes()
                check(lib.tfc_raw_validate(str(root).encode(),path.name.encode()))
                hb=struct.unpack_from('<I',good,6)[0]
                mutations=[]
                for off,value,fmt in ((6,hb+4,'I'),(10,54,'H'),(12,1,'H'),
                                      (26,1,'I'),(hb,float('nan'),'f'),(hb,0,'f'),(hb,-1,'f')):
                    b=bytearray(good); struct.pack_into('<'+fmt,b,off,value)
                    if off<hb-4: struct.pack_into('<I',b,hb-4,zlib.crc32(b[:hb-4]))
                    else: struct.pack_into('<I',b,len(b)-4,zlib.crc32(b[hb:-4]))
                    mutations.append(b)
                b=bytearray(good); b[-10]^=1; mutations += [b,good[:-1],good+b'\0']
                if variable:
                    b=bytearray(good); struct.pack_into('<I',b,66,1)
                    struct.pack_into('<I',b,hb-4,zlib.crc32(b[:hb-4])); mutations.append(b)
                for bad in mutations:
                    path.write_bytes(bad); lib.tfc_heap_reset_peak()
                    self.assertEqual(lib.tfc_raw_validate(str(root).encode(),path.name.encode()),0)
                    # A rejected file frees everything; the only allocation allowed while
                    # validating is the variable-length table (<= 4096 entries, 16 KB).
                    self.assertEqual(lib.tfc_heap_current(),0)
                    self.assertLessEqual(lib.tfc_heap_peak()-lib.tfc_heap_current(),16384)
                    count+=1
                path.write_bytes(good)
            for f in (b'\xf1\x03\x03\x00\x03\x03\x00',
                      b'\xf1\x03\x09\xff\x00\x03\x00',
                      b'\xf1\x03\x0a\xff\x00\x01\x00'):
                source=asbuf(f); self.assertEqual(lib.tfc_parse_frame(ptr(source),len(f),ptr(buf(5)),ptr(buf(214))),0)
                count+=1
        metrics['malformed_rejections']=count

    def test_05_alarm_latch_and_small_history(self):
        real_metric=metrics.get('real_alarm')
        scores=np.r_[np.ones(80),np.full(50,4),np.ones(30)].astype(np.float32)
        self.compare_alarm(scores,30,20)
        self.assertEqual(metrics['real_alarm']['latched_events'],1)
        metrics['synthetic_alarm_latch']=metrics.pop('real_alarm')
        self.compare_alarm(scores,5,5)
        metrics['small_history_alarm']=metrics.pop('real_alarm')
        if real_metric: metrics['real_alarm']=real_metric

    def test_06_exact_median_and_single_row_standardisation(self):
        with tempfile.TemporaryDirectory(prefix='tinyfl-median-') as d:
            root=Path(d); make_card(root,variable=True)
            refs=np.frombuffer((root/'m00.chk').read_bytes()[16:-4],'<f4').reshape(2,22)
            refs=refs.copy()+np.linspace(1e-5,1e-3,44,dtype=np.float32).reshape(2,22)
            chkhead=(root/'m00.chk').read_bytes()[:16]
            (root/'m00.chk').write_bytes(crc(chkhead+refs.astype('<f4').tobytes()))
            actual=np.stack([extract(q,s) for q,s in acquisitions(root/'m00s0.raw')])
            diff=abs(actual-refs).ravel()
            ns=C.c_int(); nt=C.c_int(); check3=np.empty(3,np.float32)
            c=np.empty(88,np.float32); s=np.empty_like(c)
            check(lib.tfc_prepare(str(root).encode(),0,C.byref(ns),C.byref(nt),ptr(check3),ptr(c),ptr(s)))
            self.assertEqual(check3[1],np.max(diff)); self.assertEqual(check3[2],np.median(diff))
            self.assertTrue(np.array_equal(s[:22],np.ones(22,np.float32)))
            metrics['exact_self_check_median']={'differences':int(len(diff)),
                                               'median':float(check3[2]),'exact':True}
        bad=np.full((2,22),np.nan,np.float32); c=np.empty(22,np.float32); s=np.empty_like(c)
        self.assertEqual(lib.tfc_std_params(ptr(bad),2,ptr(c),ptr(s)),0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    metrics['tests_run']=result.testsRun; metrics['failures']=len(result.failures); metrics['errors']=len(result.errors)
    print(json.dumps(metrics,indent=2))
    if args.report: args.report.write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
