"""Host interoperability: production C++ v4 code, read-only Python codec and wire oracle.

No real serial ports, keys, roster, protected-folder writes or RF. Run with -B.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import ctypes as C
from collections import deque
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "server"))
import protocol_v4 as p
import provision_board as pc
from keys_v4 import derive_app_key, provisioning_crc
from roster_v4 import Roster
SIM = HERE.parents[2] / "autoencoder_cwru_release/full_pipeline_simulation"
sys.path[:0] = [str(SIM/"research/hardware_twin_scaling_20260929"),
                str(SIM/"research/healthy_only_causal_20260908")]
import raw_features_f32 as rf
from cadence import dense_flags

U8, U16, U32, F32, I8 = C.c_uint8, C.c_uint16, C.c_uint32, C.c_float, C.c_int8
P8 = C.POINTER(U8)
DLL = C.CDLL(str(HERE / "tfcore_v4.dll"))
NODE = C.CDLL(str(HERE / "tfnode_v4.dll"))
for name, args, result in [
    ("tfn_reset",[C.c_int],None),("tfn_next",[P8],C.c_size_t),("tfn_finish",[],None),
    ("tfn_deliver",[P8,C.c_size_t],None),("tfn_update",[C.c_int,P8],C.c_int),
    ("tfn_state",[C.POINTER(U32)],None),("tfn_reboot",[],C.c_int),
    ("tfn_final",[C.c_int],C.c_int),("tfn_requested",[C.c_int],C.c_int),
    ("tfn_alarm_state_bytes",[],C.c_size_t),
    ("tfn_advance",[U32],None),("tfn_active",[],U32),
    ("tfn_participants",[C.c_int],U32),("tfn_heard",[],U32),
    ("tfn_model_active",[C.c_int,C.c_int],C.c_int),("tfn_csv",[P8,C.c_size_t],C.c_size_t),
    ("tfn_corrupt_latest",[],None),("tfn_load_card",[C.c_char_p],C.c_int),
    ("tfn_duty",[C.c_int],None),("tfn_uplink_ready",[],C.c_int),
    ("tfn_receiving",[],C.c_int),("tfn_classc_bad_calls",[],C.c_int),
    ("tfn_txlog_count",[],C.c_size_t),("tfn_txlog",[C.c_size_t,C.POINTER(C.c_uint64)],None),
]:
    fn=getattr(NODE,name); fn.argtypes=args; fn.restype=result
for name, args, result in [
    ("tfv4_model", [P8,U8,C.POINTER(U32),C.POINTER(F32),C.POINTER(I8),F32], C.c_size_t),
    ("tfv4_model_roundtrip", [P8,P8,C.c_size_t,U8], C.c_size_t),
    ("tfv4_fragment", [P8,P8,U8,U16,U8], C.c_size_t),
    ("tfv4_control", [P8,U8,U16,C.POINTER(U32),C.POINTER(F32),C.POINTER(U16),U8], C.c_size_t),
    ("tfv4_control_roundtrip", [P8,P8,C.c_size_t], C.c_size_t),
    ("tfv4_ready", [P8,U32,U16], C.c_size_t),
    ("tfv4_provision", [C.c_char_p,P8,P8,C.POINTER(U32),C.POINTER(C.c_int)], U32),
    ("tfv4_lmic_eui", [P8,P8], None),
    ("tfv4_nvs_reset", [C.c_int], None),
    ("tfv4_nvs_counter_untouched", [], C.c_int),
    ("tfv4_nvs_get", [C.c_int,P8], C.c_int),
    ("tfv4_assembly", [P8,C.POINTER(C.c_int),C.c_int,P8,C.POINTER(C.c_int)], C.c_int),
    ("tfc_prepare", [C.c_char_p,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(F32),C.POINTER(F32),C.POINTER(F32)], C.c_int),
    ("tfc_load_x", [C.c_char_p,C.c_int,C.POINTER(F32),C.c_int,C.POINTER(C.c_int)], C.c_int),
    ("tfc_score", [C.POINTER(F32),C.POINTER(F32),C.c_int,C.POINTER(F32)], None),
    ("tfv4_replay_features", [C.c_char_p,C.c_int,C.c_int,C.POINTER(F32),C.c_int,C.c_int,C.POINTER(F32),P8,C.POINTER(C.c_int)], C.c_int),
]:
    fn = getattr(DLL, name); fn.argtypes = args; fn.restype = result

def buf(data: bytes):
    return (U8 * len(data)).from_buffer_copy(data)

def cpp_control(type_, slot, values, floats=(), items=()):
    out = (U8 * 221)()
    n = DLL.tfv4_control(out,type_,slot,(U32*len(values))(*values),
                        (F32*len(floats))(*floats),(U16*len(items))(*items),len(items))
    return bytes(out[:n])

class FakeSerialPair:
    """Two in-memory endpoints; PC write fragments reach the C++ board parser.

    The board response uses real store/readback adapter with fake Preferences.
    This is not server.provision_board.FakeBoardSerial talking to itself.
    """
    def __init__(self, *, fault=0, badcrc=False):
        DLL.tfv4_nvs_reset(fault)
        self.to_pc = deque([b"boot noise\n", b"TFPROV HELLO 42 0123456789ABCDEF\n"])
        self.to_board = bytearray(); self.raw = None; self.join_lmic = None; self.address = None
        self.badcrc = badcrc; self.received_count = 0

    def readline(self):
        return self.to_pc.popleft() if self.to_pc else b""

    def write(self, data):
        # Exercise stream chunk boundaries rather than passing token objects.
        for start in range(0,len(data),7):
            self.to_board.extend(data[start:start+7])
            while b"\n" in self.to_board:
                line, _, rest = self.to_board.partition(b"\n"); self.to_board = bytearray(rest)
                raw, join, addr, ok = (U8*60)(), (U8*8)(), U32(), C.c_int()
                crc = DLL.tfv4_provision(bytes(line),raw,join,C.byref(addr),C.byref(ok))
                self.received_count += 1
                if ok.value:
                    self.raw, self.join_lmic, self.address = bytes(raw), bytes(join), addr.value
                    self.to_pc.append(f"TFPROV OK {crc ^ int(self.badcrc):08X}\n".encode())
                else:
                    self.to_pc.append(b"TFPROV ERR nvs_or_input_failed\n")
        return len(data)

    def flush(self):
        pass

class Interop(unittest.TestCase):
    def setUp(self):
        NODE.tfn_reset(1)   # also clears the Class C misuse counter checked in tearDown
        self.run = 0xFEDCBA98
        self.codes = np.resize(np.arange(-127,128,dtype=np.int16),562).astype(np.int8)
        self.scales = np.array([1e-7,.000123,.125,2.5],dtype=np.float32)
        self.q = p.QuantisedModel(self.codes,tuple(map(float,self.scales)))

    def tearDown(self):
        # MCCI LMIC 6.0.1 mis-handles LMIC_enableClassC (on->"1" switches OFF, "0" is a
        # no-op while on); the mock reproduces that. The firmware must never call (0),
        # nor (1) while Class C is already on, in any test.
        self.assertEqual(NODE.tfn_classc_bad_calls(),0,"LMIC_enableClassC(0), or (1) while already on")

    def model(self, type_, values, loss=0):
        out=(U8*624)()
        n=DLL.tfv4_model(out,type_,(U32*7)(*values),self.scales.ctypes.data_as(C.POINTER(F32)),
                         self.codes.ctypes.data_as(C.POINTER(I8)),loss)
        self.assertGreater(n,0); return bytes(out[:n])

    def node_state(self):
        values=(U32*13)(); NODE.tfn_state(values); self.assertFalse(values[10],"firmware entered fatal state"); return list(values)

    def node_next(self,type_):
        out=(U8*221)(); n=NODE.tfn_next(out); self.assertGreater(n,0)
        frame=p.decode_frame(bytes(out[:n])); self.assertEqual(frame.message_type,type_)
        NODE.tfn_finish(); self.node_state(); return frame

    def node_boot_messages(self,machines):
        hello=p.decode_board_hello(self.node_next(6)); self.assertEqual(hello.positions,tuple(range(300,300+machines)))
        for m in range(machines): self.assertEqual(p.decode_self_check(self.node_next(8)).position,300+m)
        return hello

    def node_global(self,run,phase,round_,rounds,slot):
        payload=p.encode_global(run,phase,round_,rounds,self.q)
        for f in reversed(p.fragment_global(payload,slot)): NODE.tfn_deliver(buf(f),len(f))
        self.node_state()

    def node_cached_updates(self,machines,phase,round_,base):
        updates=[]
        for m in range(machines):
            update=p.encode_update(300+m,self.run,phase,round_,base,30,self.q,.375)
            self.assertTrue(NODE.tfn_update(m,buf(update))); updates.append(update)
        return updates

    def masked_global(self,phase,round_,rounds,slot,positions=None,*,run=None,zero=False,base=None):
        # Independent contract oracle: no server participation encoder involved.
        selected=tuple(positions or ())
        mask_base=min(selected) if selected and base is None else (base or 0)
        mask=bytearray(32)
        for pos in selected:
            i=pos-mask_base; self.assertTrue(0<=i<256); mask[i//8]|=1<<(i%8)
        flags=int(round_==rounds)|(2 if positions is None else 0)
        codes=bytes(562) if zero else self.codes.tobytes()
        scales=(1.,)*4 if zero else tuple(self.scales)
        payload=struct.pack("<IBBBB4f",self.run if run is None else run,phase,round_,rounds,flags,*scales)+codes+struct.pack("<H32s",mask_base,mask)
        payload+=struct.pack("<I",zlib.crc32(payload)&0xffffffff)
        for i in (2,0,1):
            body=payload[213*i:213*(i+1)]
            frame=struct.pack("<BBHBBBB",0xf1,4,slot,i,3,len(body),3)+body
            NODE.tfn_deliver(buf(frame),len(frame))
        self.node_state(); return payload

    def command(self,command,phases=(),run=None):
        frame=p.encode_run_control(self.run if run is None else run,command,phases)
        NODE.tfn_deliver(buf(frame),len(frame)); self.node_state()

    def load_replay_card(self,root,machines=1,*,fault=True):
        # Synthetic raw inputs run through the REAL preparation/extraction and
        # node replay path, including CRC files, ring checkpoints and outboxes.
        rng=np.random.default_rng(20261003); n=90; train=30; cal=20
        for m in range(machines):
            pos=300+m; rows=[]; data=bytearray()
            for a in range(n):
                q=np.clip(rng.normal(0,18000 if fault and a>=60 else 2500,512),-32767,32767).astype("<i2")
                scale=np.float32(.001); data.extend(struct.pack("<f",scale)+q.tobytes())
                rows.append(rf.acquisition_22_f32((q.astype(np.float32)*scale)[None,:]))
            def crc(b): return b+struct.pack("<I",zlib.crc32(b)&0xffffffff)
            header=struct.pack("<4sHI4H12I",b"TFR1",1,70,pos,0,1,1,n,512,0,train,train,cal,50,10,60,30,0,0)
            (root/f"m{pos:04x}s0.raw").write_bytes(crc(header)+crc(data))
            chk=struct.pack("<4sHHIHH",b"TFC1",1,pos,50,1,0)
            (root/f"m{pos:04x}.chk").write_bytes(crc(chk+np.asarray(rows[:50],np.float32).astype("<f4").tobytes()))
        self.assertTrue(NODE.tfn_load_card(str(root).encode()))

    def collect_replay(self,expected):
        out=(U8*221)(); frames=[]
        for step in range(2000):
            n=NODE.tfn_next(out); self.node_state()
            if n:
                f=p.decode_frame(bytes(out[:n])); frames.append(f); NODE.tfn_finish()
                if f.message_type in (2,6) and sum(f.message_type==9 for f in frames)>=expected: return frames
        self.fail("real node replay did not finish")

    def test_rev43_mask_codec_window_boundaries_and_crc(self):
        NODE.tfn_reset(8)
        for selected,base in [((300,301,307),300),((65534,),65300),((0,255),0),((),0),(None,0)]:
            payload=self.masked_global(0,1,1,1,selected,base=base)
            out=(U8*624)(); self.assertEqual(DLL.tfv4_model_roundtrip(out,buf(payload),624,3),624)
            self.assertEqual(bytes(out),payload)
            self.assertEqual(len(payload),624)
            self.assertEqual([len(p.fragment_global(payload,1)[i]) for i in range(3)],[221,221,206])
            for at in (586,588,619,620):
                bad=bytearray(payload); bad[at]^=1
                self.assertEqual(DLL.tfv4_model_roundtrip(out,buf(bad),624,3),0)
        bad=bytearray(payload); bad[7]|=4; bad[-4:]=struct.pack("<I",zlib.crc32(bad[:-4])&0xffffffff)
        self.assertEqual(DLL.tfv4_model_roundtrip(out,buf(bad),624,3),0)

    def test_rev43_participation_updates_resume_and_crc_fallback(self):
        NODE.tfn_reset(8); self.node_boot_messages(8); self.node_next(6)
        self.masked_global(0,0,2,1,(300,303,307)); self.assertEqual(NODE.tfn_active(),0x89)
        self.node_next(4); self.node_cached_updates(8,0,1,1)
        for _ in range(3): self.assertEqual(p.split_update_fragment(self.node_next(1))[0],300)
        # Both checkpoint generations carry the held mask; corrupt the newer
        # one and recover the preceding generation without activating extras.
        NODE.tfn_corrupt_latest(); self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(8)
        self.assertEqual(NODE.tfn_active(),0x89)
        remaining=[]
        for _ in range(10):
            out=(U8*221)(); n=NODE.tfn_next(out); self.assertGreater(n,0); f=p.decode_frame(bytes(out[:n])); NODE.tfn_finish()
            if f.message_type==6: break
            self.assertEqual(f.message_type,1); remaining.append(p.split_update_fragment(f)[0])
        self.assertEqual(set(remaining),{300,303,307})
        self.assertEqual(self.node_state()[11],1)

    def test_rev43_inactive_board_listens_and_ignores_repairs(self):
        NODE.tfn_reset(2); self.node_boot_messages(2); self.node_next(6)
        self.masked_global(0,0,12,1,(600,)); self.assertEqual(NODE.tfn_active(),0)
        self.node_next(4); self.node_next(2); self.assertEqual(self.node_state()[11],1)
        repair=p.encode_update_repair(1,self.run,300,7); NODE.tfn_deliver(buf(repair),len(repair))
        out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)
        NODE.tfn_advance(180000); self.assertEqual(NODE.tfn_next(out),0); self.node_next(2)
        self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(2); self.node_next(6)
        self.assertEqual(NODE.tfn_active(),0); self.assertEqual(self.node_state()[11],1)

    def test_rev43_midphase_join_planned_and_elastic(self):
        for selected,wanted in [((301,),2),(None,3)]:
            NODE.tfn_reset(2); self.node_boot_messages(2); self.node_next(6)
            self.command(3); self.node_next(6); self.masked_global(2,5,12,37,selected)
            self.assertEqual(NODE.tfn_active(),wanted); self.node_next(4)
            self.node_cached_updates(2,2,6,37)
            for m in range(2):
                if wanted&(1<<m):
                    for _ in range(3):
                        f=self.node_next(1); self.assertEqual(f.round_id,37); self.assertEqual(p.split_update_fragment(f)[0],300+m)
            self.node_next(2)

    def test_rev43_replay_only_each_phases_persisted_participants(self):
        NODE.tfn_reset(2)
        with tempfile.TemporaryDirectory(prefix="tinyfl-node-replay-") as temp:
            self.load_replay_card(Path(temp),2); self.node_boot_messages(2); self.node_next(6)
            for phase,selected in [(0,(300,)),(1,(301,)),(2,())]:
                self.masked_global(phase,1,1,phase+1,selected,zero=True); self.node_next(4); self.node_next(2)
            self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(2); self.node_next(6)
            self.assertEqual([NODE.tfn_participants(i) for i in range(3)],[1,2,0])
            self.assertEqual([[NODE.tfn_model_active(p,m) for m in range(2)] for p in range(3)],[[1,0],[0,1],[0,0]])
            self.command(1,(0,1,2))
            frames=self.collect_replay(2)
            dones=[p.decode_replay_done(f) for f in frames if f.message_type==9]
            self.assertEqual({(d.position,d.phase_index) for d in dones},{(300,0),(301,1)})
            self.assertEqual(self.node_state()[11],1)
            csv=(U8*65536)(); size=NODE.tfn_csv(csv,len(csv))
            rows=[line.split(",") for line in bytes(csv[:size]).decode().splitlines()[1:]]
            self.assertEqual(len(rows),180)
            self.assertEqual({(int(row[0]),int(row[2])) for row in rows},{(300,0),(301,1)})
            for f in frames:
                if f.message_type==7:
                    a=p.decode_alarm(f); self.assertIn((a.position,a.phase_index),{(300,0),(301,1)})

    def test_rev43_done_alarm_index_survives_reboot_and_retry(self):
        NODE.tfn_reset(1)
        with tempfile.TemporaryDirectory(prefix="tinyfl-node-done-") as temp:
            self.load_replay_card(Path(temp)); self.node_boot_messages(1); self.node_next(6)
            self.masked_global(0,1,1,1,(300,),zero=True); self.node_next(4); self.node_next(2); self.command(1,(0,))
            out=(U8*221)(); alarm=None; done=None; rebooted=False
            for _ in range(1000):
                n=NODE.tfn_next(out); self.node_state()
                if not n: continue
                f=p.decode_frame(bytes(out[:n]))
                if f.message_type==7:
                    alarm=p.decode_alarm(f); NODE.tfn_finish()
                    self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(1); rebooted=True
                elif f.message_type==9:
                    done=bytes(out[:n]); NODE.tfn_finish(); break
                else: NODE.tfn_finish()
            self.assertTrue(rebooted); self.assertIsNotNone(alarm); self.assertIsNotNone(done)
            d=p.decode_replay_done(p.decode_frame(done)); self.assertEqual(len(done),24)
            self.assertEqual(d.first_alarm_index,alarm.acquisition_index); self.assertEqual(d.alarm_latched,1)
            self.assertEqual(self.node_state()[11],1); self.node_next(6) # no heard-run since reboot
            self.command(1,(0,)); self.assertEqual(NODE.tfn_next(out),0)
            n=NODE.tfn_next(out); self.assertEqual(bytes(out[:n]),done); NODE.tfn_finish(); self.node_next(2)
            self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(1); self.command(1,(0,))
            self.assertEqual(NODE.tfn_next(out),0); n=NODE.tfn_next(out); self.assertEqual(bytes(out[:n]),done); NODE.tfn_finish()

    def test_rev43_done_no_alarm_index_and_consistency_guard(self):
        frame=cpp_control(9,65535,[self.run,300,0,90,0,0xffffffff])
        self.assertEqual(len(frame),24); self.assertEqual(struct.unpack_from("<I",frame,20)[0],0xffffffff)
        self.assertEqual(cpp_control(9,65535,[self.run,300,0,90,1,0xffffffff]),b"")
        self.assertEqual(cpp_control(9,65535,[self.run,300,0,90,0,42]),b"")
        NODE.tfn_reset(1)
        with tempfile.TemporaryDirectory(prefix="tinyfl-node-noalarm-") as temp:
            self.load_replay_card(Path(temp),fault=False); self.node_boot_messages(1); self.node_next(6)
            self.masked_global(0,1,1,1,(300,),zero=True); self.node_next(4); self.node_next(2); self.command(1,(0,))
            frames=self.collect_replay(1); d=p.decode_replay_done(next(f for f in frames if f.message_type==9))
            self.assertEqual((d.alarm_latched,d.first_alarm_index),(0,0xffffffff))

    def test_rev43_hello_repeats_until_valid_held_run_since_boot(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); first=self.node_next(6)
        out=(U8*221)(); NODE.tfn_advance(180000); self.assertEqual(NODE.tfn_next(out),0)
        self.assertEqual(self.node_next(6),first)
        self.command(1,(0,),run=42); self.assertEqual(NODE.tfn_heard(),0)
        payload=self.masked_global(0,1,1,1,(300,)); self.node_next(4); self.node_next(2)
        self.assertEqual(NODE.tfn_heard(),self.run)
        self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(1); self.node_next(6)
        bad=bytearray(payload); bad[-1]^=1
        for f in p.fragment_global(bytes(bad),1): NODE.tfn_deliver(buf(f),len(f))
        self.assertEqual(NODE.tfn_heard(),0)
        for f in p.fragment_global(payload,1): NODE.tfn_deliver(buf(f),len(f))
        # Rev 4.3 radio scheduling: the STATUS answering the re-heard GLOBAL is the
        # board's listening report, so no extra READY follows it; the periodic READY
        # (now a READY, the run having been heard) comes with the idle timer.
        self.assertEqual(NODE.tfn_heard(),self.run); self.node_next(4); self.assertEqual(NODE.tfn_next(out),0)
        NODE.tfn_advance(180000); self.assertEqual(NODE.tfn_next(out),0); self.node_next(2)
        self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(1); self.command(3)
        self.assertEqual(NODE.tfn_heard(),0); self.node_next(6)
        NODE.tfn_advance(180000); self.assertEqual(NODE.tfn_next(out),0); self.node_next(6)

    def test_rev43_end_durable_silence_and_new_run_adoption(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        self.masked_global(0,1,1,1,(300,)); self.node_next(4); self.node_next(2)
        self.command(2); out=(U8*221)()
        for reboot in (False,True):
            if reboot: self.assertTrue(NODE.tfn_reboot())
            self.command(1,(0,)); self.command(3)
            self.masked_global(1,0,12,99,(300,)) # same ended run cannot restart
            for _ in range(4): NODE.tfn_advance(180001); self.assertEqual(NODE.tfn_next(out),0)
            self.assertEqual((self.node_state()[9],self.node_state()[11]),(1,1))
        self.command(3,run=42); self.assertEqual(self.node_state()[9],0)
        self.node_boot_messages(1); self.assertEqual(NODE.tfn_heard(),0)

    def test_rev43_adopt_requests_hello_same_and_new_run(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        self.command(3)
        hello=p.decode_board_hello(self.node_next(6))
        self.assertEqual(hello.stored_run_id,self.run); self.assertFalse(hello.run_complete)
        self.assertEqual(NODE.tfn_heard(),0)
        self.masked_global(2,1,1,9,(300,)); self.node_next(4); self.node_next(2)
        state=self.node_state(); self.command(3)
        hello=p.decode_board_hello(self.node_next(6))
        self.assertEqual(hello.stored_run_id,self.run)
        self.assertEqual(self.node_state()[:7],state[:7]); self.assertTrue(NODE.tfn_final(2))
        # ADOPT does not erase a valid earlier heard-run observation either.
        self.assertEqual(NODE.tfn_heard(),self.run)
        self.assertTrue(NODE.tfn_reboot()); self.node_boot_messages(1); self.node_next(6)
        # ADOPTs received before the next TX coalesce into one HELLO.
        self.command(3); self.command(3)
        self.node_next(6); self.assertEqual(NODE.tfn_heard(),0)
        out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)
        NODE.tfn_advance(180000); self.assertEqual(NODE.tfn_next(out),0); self.node_next(6)
        self.command(3,run=42); self.command(1,(0,),run=42)
        # Even a later REPLAY cannot cancel the requested next-opportunity HELLO.
        self.assertEqual(p.decode_board_hello(self.node_next(6)).stored_run_id,42)

    def test_rev43_adopt_completed_same_run_is_ignored(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        self.masked_global(0,1,1,1,(300,)); self.node_next(4); self.node_next(2)
        self.command(2); before=self.node_state(); self.command(3)
        out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)
        NODE.tfn_advance(180001); self.assertEqual(NODE.tfn_next(out),0)
        self.assertEqual(self.node_state()[:7],before[:7]); self.assertEqual(self.node_state()[9:12],[1,0,1])
        self.assertTrue(NODE.tfn_reboot()); hello=self.node_boot_messages(1); self.assertTrue(hello.run_complete)
        self.command(3); self.assertEqual(NODE.tfn_heard(),0)
        NODE.tfn_advance(360001); self.assertEqual(NODE.tfn_next(out),0)
        self.assertEqual((self.node_state()[0],self.node_state()[9]),(self.run,1))

    def test_rev43_completed_reboot_hello_repeats_until_any_control(self):
        for command,run in [(1,self.run),(2,self.run),(1,42),(2,42),(3,self.run)]:
            with self.subTest(command=command,run=run):
                NODE.tfn_reset(2); self.node_boot_messages(2); self.node_next(6)
                self.masked_global(0,1,1,1,(300,301)); self.node_next(4); self.node_next(2); self.command(2)
                out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)
                self.assertTrue(NODE.tfn_reboot()); hello=self.node_boot_messages(2)
                self.assertTrue(hello.run_complete); self.assertEqual(hello.stored_run_id,self.run)
                self.assertEqual(self.node_state()[11],1)
                NODE.tfn_advance(179999); self.assertEqual(NODE.tfn_next(out),0)
                NODE.tfn_advance(1); repeated=p.decode_board_hello(self.node_next(6)); self.assertEqual(repeated,hello)
                NODE.tfn_advance(180000); self.assertEqual(p.decode_board_hello(self.node_next(6)),hello)
                # Malformed control is not evidence that the server was heard.
                bad=p.encode_run_control(run,command,(0,) if command==1 else ())[:-1]
                NODE.tfn_deliver(buf(bad),len(bad)); NODE.tfn_advance(180000); self.node_next(6)
                self.command(command,(0,) if command==1 else (),run=run)
                NODE.tfn_advance(360001); self.assertEqual(NODE.tfn_next(out),0)
                self.assertEqual((self.node_state()[0],self.node_state()[9],self.node_state()[11]),(self.run,1,1))
                if command==3 or run!=self.run: self.assertEqual(NODE.tfn_heard(),0)

    def test_rev43_completed_reboot_only_valid_global_stops_hello(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        payload=self.masked_global(0,1,1,1,(300,)); self.node_next(4); self.node_next(2); self.command(2)
        self.assertTrue(NODE.tfn_reboot()); self.assertTrue(self.node_boot_messages(1).run_complete)
        bad=bytearray(payload); bad[-1]^=1
        for f in p.fragment_global(bytes(bad),1): NODE.tfn_deliver(buf(f),len(f))
        NODE.tfn_advance(180000); self.node_next(6)
        for f in p.fragment_global(payload,1): NODE.tfn_deliver(buf(f),len(f))
        out=(U8*221)(); NODE.tfn_advance(360001); self.assertEqual(NODE.tfn_next(out),0)
        self.assertEqual((self.node_state()[0],self.node_state()[1],self.node_state()[9]),(self.run,1,1))

    def test_rev43_completed_adopt_hello_global_slot_one_really_trains(self):
        NODE.tfn_reset(1)
        with tempfile.TemporaryDirectory(prefix="tinyfl-node-adopt-train-") as temp:
            self.load_replay_card(Path(temp),fault=False); self.node_boot_messages(1); self.node_next(6)
            self.masked_global(0,1,1,19,(300,),zero=True); self.node_next(4); self.node_next(2); self.command(2)
            self.command(3,run=42)
            hello=p.decode_board_hello(self.node_next(6))
            self.assertEqual(hello.stored_run_id,42); self.assertFalse(hello.run_complete)
            self.assertEqual(NODE.tfn_heard(),0); self.assertEqual(self.node_state()[:3],[42,0,0])
            self.masked_global(0,0,1,1,(300,),run=42,zero=True); self.node_next(4)
            partial=p.new_update_partial(); payload=None
            for _ in range(3):
                f=self.node_next(1); self.assertEqual(f.round_id,1)
                pos,f=p.split_update_fragment(f); self.assertEqual(pos,300); payload=partial.add(f)
            update=p.decode_update(payload)
            self.assertEqual((update.run_id,update.round,update.base_slot,update.n_train_rows),(42,1,1,30))
            self.assertTrue(np.any(update.model.codes)); self.assertTrue(np.isfinite(update.train_loss))
            self.assertEqual(NODE.tfn_heard(),42); self.node_next(2)

    def test_actual_node_update_repair_final_and_restart_ready(self):
        NODE.tfn_reset(8); hello=self.node_boot_messages(8); self.assertEqual(hello.stored_run_id,0)
        repeated=p.decode_board_hello(self.node_next(6)); self.assertEqual(repeated.stored_run_id,0)
        self.node_global(self.run,37,0,2,1)
        status=p.decode_global_status(self.node_next(4)); self.assertEqual((status.slot,status.missing_mask),(1,0))
        updates=self.node_cached_updates(8,37,1,1)
        for m in range(8):
            for expected in p.fragment_update(updates[m],1):
                f=self.node_next(1); self.assertEqual(p.encode_frame(f.message_type,f.round_id,f.fragment_index,f.fragment_count,f.data),expected)
        self.assertEqual(p.decode_class_c_ready(self.node_next(2)).awaited_slot,2)
        repair=p.encode_update_repair(1,self.run,300,5); NODE.tfn_deliver(buf(repair),len(repair))
        for i in (0,2):
            f=self.node_next(1); self.assertEqual(f.fragment_index,i); self.assertEqual(f.round_id,1)
            self.assertEqual(f.data,p.decode_frame(p.fragment_update(updates[0],1)[i]).data)
        self.assertEqual(p.decode_class_c_ready(self.node_next(2)).awaited_slot,2)
        self.node_global(self.run,37,1,2,2); self.node_next(4)
        self.node_cached_updates(8,37,2,2)
        for _ in range(3): self.assertEqual(p.split_update_fragment(self.node_next(1))[0],300)
        # Power loss after machine 0: actual CRC checkpoint/cache reload keeps
        # the held slot and skips its fully transmitted update on restart.
        self.assertTrue(NODE.tfn_reboot()); hello=self.node_boot_messages(8)
        self.assertEqual(hello.stored_run_id,self.run); self.assertEqual(self.node_state()[1:3],[2,2])
        for m in range(1,8):
            for _ in range(3): self.assertEqual(p.split_update_fragment(self.node_next(1))[0],300+m)
        self.assertEqual(p.decode_board_hello(self.node_next(6)).stored_run_id,self.run)
        self.node_global(self.run,37,2,2,3); self.assertTrue(NODE.tfn_final(37)); self.node_next(4)
        self.assertEqual(p.decode_class_c_ready(self.node_next(2)).awaited_slot,4)
        # A final model is not an automatic replay/end-of-run instruction.
        out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)

    def test_actual_node_stale_repair_then_late_global_keeps_bytes(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        self.node_global(self.run,0,0,3,1); self.node_next(4)
        # Seed 2 repair authorises round 2, but only GLOBAL 1 is held.
        repair=p.encode_update_repair(2,self.run,300,5); NODE.tfn_deliver(buf(repair),len(repair))
        updates=self.node_cached_updates(1,0,2,1)
        first=self.node_next(1); self.assertEqual((first.round_id,first.fragment_index),(2,0))
        self.node_global(self.run,0,1,3,2)
        last=self.node_next(1); self.assertEqual((last.round_id,last.fragment_index),(2,2))
        self.assertEqual(last.data,p.decode_frame(p.fragment_update(updates[0],2)[2]).data)
        self.node_next(4)
        self.assertEqual(p.decode_class_c_ready(self.node_next(2)).awaited_slot,3)

    def test_actual_node_adopt_completed_reboot_and_chunked_replay(self):
        NODE.tfn_reset(1); self.node_boot_messages(1); self.node_next(6)
        adopt=p.encode_run_control(self.run,3); NODE.tfn_deliver(buf(adopt),len(adopt))
        self.assertEqual(self.node_state()[:3],[self.run,0,0]); self.node_next(6); self.assertEqual(NODE.tfn_heard(),0)
        # Slot zero is never accepted; a CRC-valid mid-phase GLOBAL is allowed
        # once this run has been adopted, and positions are not limited to 54.
        self.node_global(self.run,254,128,255,0); self.assertEqual(self.node_state()[1],0)
        self.node_global(self.run,254,128,255,400); self.assertEqual(self.node_state()[1:6],[400,400,254,129,255])
        # A different active run must start at round zero (or explicit ADOPT).
        self.node_global(42,0,5,12,1); self.assertEqual(self.node_state()[0],self.run)
        adopt=p.encode_run_control(42,3); NODE.tfn_deliver(buf(adopt),len(adopt)); self.assertEqual(self.node_state()[:3],[42,0,0]); self.node_next(6)
        self.node_global(42,0,1,1,1); self.node_next(4); self.node_next(2); self.assertTrue(NODE.tfn_final(0))
        end=p.encode_run_control(42,2); NODE.tfn_deliver(buf(end),len(end)); state=self.node_state()
        self.assertEqual((state[9],state[11]),(1,1)); self.assertTrue(NODE.tfn_final(0))
        self.assertTrue(NODE.tfn_reboot()); self.assertTrue(self.node_boot_messages(1).run_complete)
        self.command(3,run=42); out=(U8*221)(); self.assertEqual(NODE.tfn_next(out),0)
        self.assertEqual(self.node_state()[11],1)
        # A completed board can adopt any valid GLOBAL of a new run, including
        # a lower slot and a final model; no unsolicited state mixing.
        self.node_global(self.run,0,1,1,1); self.node_next(4); self.node_next(2)
        for phase in range(1,255): self.node_global(self.run,phase,1,1,phase+1)
        for phases in (range(207),range(207,255)):
            command=p.encode_run_control(self.run,1,phases); NODE.tfn_deliver(buf(command),len(command))
        self.assertTrue(all(NODE.tfn_requested(phase) for phase in range(255)))
        self.assertFalse(self.node_state()[9])

    # ---- Rev 4.3 radio scheduling (duty cycle) -------------------------------
    # The mock LMIC models the EU868 1 % sub-band exactly as MCCI LMIC does
    # (band avail = tx start + airtime x 100, TX_RAMPUP, 8 h sentinel). These
    # tests run it with time advancing; the legacy tests above leave it off.

    def duty_pump(self,count,*,max_ms=600000,step=100,after_tx=2000,listening=False,stop_on=None):
        """Drive the node as its loop would: poll, finish each uplink after its RX
        windows, advance time. Returns the decoded uplinks (at least `count`)."""
        out=(U8*221)(); frames=[]; elapsed=0
        while elapsed<max_ms and len(frames)<count:
            n=NODE.tfn_next(out); state=self.node_state()
            if n:
                frames.append(p.decode_frame(bytes(out[:n]))); NODE.tfn_finish(); NODE.tfn_advance(after_tx); elapsed+=after_tx
            else:
                if listening: self.assertEqual(NODE.tfn_receiving(),1,"not receiving while only waiting for the duty cycle")
                NODE.tfn_advance(step); elapsed+=step
        self.assertGreaterEqual(len(frames),count,"node did not transmit in time"); return frames

    def duty_boot(self,machines=1):
        NODE.tfn_reset(machines); NODE.tfn_duty(1)
        frames=self.duty_pump(machines+2)
        self.assertEqual([f.message_type for f in frames],[6]+[8]*machines+[6])
        self.assertEqual(NODE.tfn_receiving(),1)

    def partial_global(self,slot=1,fragments=(0,1)):
        payload=p.encode_global(self.run,0,0,12,self.q); frames=p.fragment_global(payload,slot)
        for i in fragments: NODE.tfn_deliver(buf(frames[i]),len(frames[i]))
        self.node_state(); return frames

    def test_rev43_status_returns_to_class_c_without_extra_ready_or_hello(self):
        # The 10 %-loss stall: GLOBAL_STATUS used to set needReady, so HELLO/READY
        # was queued at once and Class C dropped for the whole duty-cycle wait.
        self.duty_boot()
        frames=self.partial_global(1,(0,1))
        NODE.tfn_advance(120000)   # idle timer with a partial GLOBAL: report what is missing
        status=self.duty_pump(1)[0]; self.assertEqual(status.message_type,4)
        decoded=p.decode_global_status(status); self.assertEqual((decoded.slot,decoded.missing_mask),(1,4))
        state=self.node_state()
        self.assertEqual((state[7],state[8],state[11]),(0,1,1)) # need_ready, waiting, class_c
        # Nothing follows the STATUS (not a READY, not a HELLO) and the board never
        # leaves Class C, throughout the duty-cycle wait and well beyond it.
        out=(U8*221)()
        for _ in range(1000):
            self.assertEqual(NODE.tfn_next(out),0); self.assertEqual(NODE.tfn_receiving(),1); NODE.tfn_advance(100)
        # The server's prompt repair (4 s after the STATUS) is received and completes the GLOBAL.
        NODE.tfn_deliver(buf(frames[2]),len(frames[2])); self.assertEqual(self.node_state()[1],1)

    def test_rev43_class_c_stays_on_and_accepts_global_while_duty_cycle_blocks(self):
        self.duty_boot()
        frames=self.partial_global(1,(0,1))
        NODE.tfn_advance(120000); self.assertEqual(self.duty_pump(1)[0].message_type,4)
        NODE.tfn_advance(4000)     # the repair multicast arrives 4 s after the STATUS
        self.assertFalse(NODE.tfn_uplink_ready()); self.assertEqual(NODE.tfn_receiving(),1)
        NODE.tfn_deliver(buf(frames[2]),len(frames[2]))      # delivered by the Class C receiver
        state=self.node_state(); self.assertEqual(state[1],1) # GLOBAL 1 accepted during the wait
        out=(U8*221)()
        # Its STATUS is wanted at once, but the duty cycle forbids it: no frame, no Class C drop.
        self.assertFalse(NODE.tfn_uplink_ready())
        self.assertEqual(NODE.tfn_next(out),0); self.assertEqual(NODE.tfn_receiving(),1)
        # A whole second GLOBAL is accepted in the same wait (waiting for the duty cycle is not deafness).
        full=p.fragment_global(p.encode_global(self.run,0,1,12,self.q),2)
        for f in full: NODE.tfn_deliver(buf(f),len(f))
        self.assertEqual(self.node_state()[1],2); self.assertEqual(NODE.tfn_receiving(),1)
        self.assertEqual(NODE.tfn_next(out),0)
        reports=self.duty_pump(1,listening=True)           # transmits only once the band is free again
        self.assertEqual(reports[0].message_type,4)
        self.assertEqual(p.decode_global_status(reports[0]).slot,2)

    def test_rev43_repair_waits_for_the_duty_cycle_in_class_c(self):
        # processDownlink(REPAIR) used to call LMIC_enableClassC(0) before the repair
        # uplink could start; the repair fragment then sat deaf in Class A.
        self.duty_boot()
        self.masked_global(0,0,12,1,(300,)); self.node_cached_updates(1,0,1,1)
        frames=self.duty_pump(5)
        self.assertEqual([f.message_type for f in frames],[4,1,1,1,2])
        self.assertEqual(NODE.tfn_receiving(),1)
        repair=p.encode_update_repair(1,self.run,300,5); NODE.tfn_deliver(buf(repair),len(repair))
        out=(U8*221)()
        self.assertFalse(NODE.tfn_uplink_ready()); self.assertEqual(NODE.tfn_receiving(),1)
        self.assertEqual(NODE.tfn_next(out),0); self.assertEqual(NODE.tfn_receiving(),1)
        repaired=self.duty_pump(3,listening=True)
        self.assertEqual([f.message_type for f in repaired],[1,1,2])
        self.assertEqual([f.fragment_index for f in repaired[:2]],[0,2])

    def test_rev43_duty_cycle_is_respected_by_every_uplink(self):
        self.duty_boot()
        self.masked_global(0,0,12,1,(300,)); self.node_cached_updates(1,0,1,1)
        self.duty_pump(5)
        repair=p.encode_update_repair(1,self.run,300,5); NODE.tfn_deliver(buf(repair),len(repair))
        self.duty_pump(3)
        NODE.tfn_advance(180000); self.duty_pump(1)          # periodic idle uplink
        count=NODE.tfn_txlog_count(); self.assertEqual(count,12)
        log=[]
        for i in range(count):
            row=(C.c_uint64*6)(); NODE.tfn_txlog(i,row); log.append(tuple(row))
        self.assertFalse(any(r[5] for r in log),"a frame was handed to LMIC that had to wait for the duty cycle")
        for earlier,later in zip(log,log[1:]):
            if earlier[3]==later[3]:  # same sub-band
                # LMIC lets a frame start up to TX_RAMPUP (10 ms) early, which is below one airtime.
                self.assertGreaterEqual(later[1]-earlier[1],99*earlier[2],(earlier,later))
        self.assertTrue(any(r[4]>=200 for r in log))  # includes full UPDATE fragments (~37 s off-time each)

    def test_rev43_class_c_enabled_once_and_not_receiving_only_during_tx(self):
        # LMIC keeps Class C enabled across uplinks but does not receive from the
        # moment a TX is requested until RX1/RX2 are over (engineUpdate/isTxRequested).
        NODE.tfn_reset(1); NODE.tfn_duty(1); out=(U8*221)()
        self.assertEqual(NODE.tfn_receiving(),0)             # not enabled before the first uplink completes
        self.assertGreater(NODE.tfn_next(out),0)             # HELLO requested
        self.assertEqual(NODE.tfn_receiving(),0); NODE.tfn_finish()
        enabled=self.node_state()[11]; self.assertEqual(enabled,1)
        self.assertEqual(NODE.tfn_receiving(),0)             # RX windows of that uplink still open
        NODE.tfn_advance(2000); self.assertEqual(NODE.tfn_receiving(),1)
        for _ in range(40):                                  # many more uplinks and waits: never switched off or toggled
            n=NODE.tfn_next(out)
            if n:
                self.assertEqual(NODE.tfn_receiving(),0); NODE.tfn_finish(); NODE.tfn_advance(2000)
            else: NODE.tfn_advance(500)
            self.assertEqual(self.node_state()[11],1); self.assertEqual(NODE.tfn_receiving(),1)

    def test_rev43_stale_band_timer_after_long_idle_does_not_stall_the_node(self):
        # LMIC's 32-bit tick arithmetic (and its now+8 h sentinel) would otherwise
        # hold an uplink for hours once a band timer is older than ~1.5 h.
        self.duty_boot()
        NODE.tfn_advance(2*3600*1000)
        frames=self.duty_pump(1,max_ms=10000)
        self.assertEqual(frames[0].message_type,6)
        row=(C.c_uint64*6)(); NODE.tfn_txlog(NODE.tfn_txlog_count()-1,row); self.assertFalse(row[5])

    def test_update_and_global_both_directions(self):
        # Explicit values beyond all v3 widths, plus boundaries and saturation.
        for pos, phase, round_, slot in [(0,0,1,1),(513,7,12,300),(65534,254,255,65534)]:
            update = p.encode_update(pos,self.run,phase,round_,slot,99999,self.q,.375)
            self.assertEqual(len(update),598)
            self.assertEqual(update,self.model(p.MSG_UPDATE_FRAGMENT,[self.run,pos,phase,round_,slot,99999,0],.375))
            out=(U8*624)()
            self.assertEqual(DLL.tfv4_model_roundtrip(out,buf(update),len(update),1),598)
            self.assertEqual(bytes(out[:598]),update)
            u=p.decode_update(bytes(out[:598])); self.assertEqual((u.position,u.run_id,u.base_slot,u.n_train_rows),(pos,self.run,slot,65535))
            for final in (False,True):
                r=round_ if final else round_-1
                global_=p.encode_global(self.run,phase,r,round_,self.q)
                self.assertEqual(global_,self.model(3,[self.run,0,phase,r,0,0,round_]))
                self.assertEqual(DLL.tfv4_model_roundtrip(out,buf(global_),len(global_),3),624)
                self.assertEqual(bytes(out),global_); self.assertEqual(p.decode_global(bytes(out)).final,final)
            for type_, payload, expected in [(1,update,p.fragment_update(update,slot)),(3,global_,p.fragment_global(global_,slot))]:
                frames=[]
                for i in range(3):
                    f=(U8*221)(); n=DLL.tfv4_fragment(f,buf(payload),type_,slot,i)
                    frames.append(bytes(f[:n])); self.assertEqual(frames[-1],expected[i]); p.decode_frame(frames[-1])
                self.assertEqual(list(map(len,frames)),[221,221,186] if type_==1 else [221,221,206])
                partial=p.new_update_partial() if type_==1 else p.new_global_partial()
                assembled=None
                for f in (frames[2],frames[0],frames[1]):
                    frame=p.decode_frame(f)
                    if type_==1: position,frame=p.split_update_fragment(frame); self.assertEqual(position,pos)
                    assembled=partial.add(frame)
                self.assertEqual(assembled,payload)

    def control_cases(self):
        r=self.run
        return [
            (2,301,[r,301],(),(),p.encode_class_c_ready(r,301),p.decode_class_c_ready),
            (2,65535,[r,65535],(),(),p.encode_class_c_ready(r,65535),p.decode_class_c_ready),
            (4,300,[r,300,0],(),(),p.encode_global_status(r,300,0),p.decode_global_status),
            (4,300,[r,300,7],(),(),p.encode_global_status(r,300,7),p.decode_global_status),
            (5,300,[r,65534,5],(),(),p.encode_update_repair(300,r,65534,5),p.decode_update_repair),
            (6,65535,[65534,42,11,r],(),(0,300,65534),p.encode_board_hello(65534,42,11,r,(0,300,65534)),p.decode_board_hello),
            (6,65535,[1000,42,3,0],(),tuple(range(8)),p.encode_board_hello(1000,42,3,0,range(8)),p.decode_board_hello),
            (7,65535,[r,65534,254,0xffffffff],(3.25,2.5),(),p.encode_alarm(r,65534,254,0xffffffff,3.25,2.5),p.decode_alarm),
            (8,65535,[65534,65535],(.001,.000125),(),p.encode_self_check(65534,65535,.001,.000125),p.decode_self_check),
            (9,65535,[r,65534,254,100000,1,999],(),(),p.encode_replay_done(r,65534,254,100000,1,999),p.decode_replay_done),
            (9,65535,[r,0,0,100000,0,0xffffffff],(),(),p.encode_replay_done(r,0,0,100000,0,0xffffffff),p.decode_replay_done),
            (10,65535,[r,1],(),(0,7,254),p.encode_run_control(r,1,(0,7,254)),p.decode_run_control),
            (10,65535,[r,1],(),tuple(range(207)),p.encode_run_control(r,1,range(207)),p.decode_run_control),
            (10,65535,[r,2],(),(),p.encode_run_control(r,2),p.decode_run_control),
            (10,65535,[r,3],(),(),p.encode_run_control(r,3),p.decode_run_control),
        ]

    def test_every_control_both_directions(self):
        for type_,slot,v,x,items,expected,decode in self.control_cases():
            with self.subTest(type=type_,slot=slot,n=len(items)):
                actual=cpp_control(type_,slot,v,x,items); self.assertEqual(actual,expected)
                self.assertEqual(decode(p.decode_frame(actual)),decode(p.decode_frame(expected)))
                out=(U8*221)(); n=DLL.tfv4_control_roundtrip(out,buf(expected),len(expected))
                self.assertEqual(bytes(out[:n]),expected)

    def test_ready_state_rule_all_transitions(self):
        # Same function used at boot, after every machine's last UPDATE,
        # repair exhaustion and final model flash storage; never seed_slot+2.
        for context in ("boot","last_update","repair_answers","final_model"):
            for held in (0,1,13,300,65534):
                out=(U8*221)(); n=DLL.tfv4_ready(out,self.run,held)
                decoded=p.decode_class_c_ready(p.decode_frame(bytes(out[:n])))
                self.assertEqual(decoded.awaited_slot,held+1 if held else 65535,(context,held))

    def test_reassembly_reorder_duplicate_conflict(self):
        payload=p.encode_global(self.run,254,255,255,self.q); f=p.fragment_global(payload,65534)
        for frames, wanted in [([f[2],f[0],f[0],f[1]],[1,1,3,2]),
                               ([f[0],f[0][:-1]+bytes([f[0][-1]^1]),f[2],f[1]],[1,0,1,2])]:
            out=(U8*624)(); results=(C.c_int*len(frames))()
            mask=DLL.tfv4_assembly(buf(b"".join(frames)),(C.c_int*len(frames))(*map(len,frames)),len(frames),out,results)
            self.assertEqual(mask,7); self.assertEqual(list(results),wanted); self.assertEqual(bytes(out),payload)

    def test_malformed_controls_and_models(self):
        bad=[]
        for *_,expected,decode in self.control_cases():
            for index in (0,1,5,6,7):
                damaged=bytearray(expected); damaged[index]=255; bad.append(bytes(damaged))
            bad += [expected[:-1],expected+b"\0"]
        # Body field guards, not only header/length errors.
        bad += [p.encode_frame(4,1,0,1,bytes.fromhex("01000000010008")),
                p.encode_frame(5,1,0,1,bytes.fromhex("01000000010000")),
                p.encode_frame(9,65535,0,1,bytes.fromhex("010000000100000100000002")),
                p.encode_frame(10,65535,0,1,bytes.fromhex("0100000003020001")),
                p.encode_frame(2,301,0,1,bytes.fromhex("010000002c01"))]
        rejected=0
        for frame in bad:
            try:
                parsed=p.decode_frame(frame)
                {2:p.decode_class_c_ready,4:p.decode_global_status,5:p.decode_update_repair,
                 6:p.decode_board_hello,7:p.decode_alarm,8:p.decode_self_check,9:p.decode_replay_done,10:p.decode_run_control}[parsed.message_type](parsed)
            except (p.ProtocolError,KeyError):
                out=(U8*221)(); self.assertEqual(DLL.tfv4_control_roundtrip(out,buf(frame),len(frame)),0); rejected+=1
        self.assertGreater(rejected,80)
        payload=p.encode_global(self.run,0,0,12,self.q)
        for at in (0,8,24,586):
            damaged=bytearray(payload); damaged[at]^=1
            self.assertEqual(DLL.tfv4_model_roundtrip((U8*624)(),buf(damaged),len(damaged),3),0)

    def test_actual_pc_provision_exchange(self):
        # Public synthetic test vectors ONLY, never environment/lab secrets.
        join=bytes.fromhex("1032547698BADCFE"); master=bytes(range(16))
        mc=pc.KeyMaterial(join,bytes.fromhex("12345678"),bytes(range(16,32)),bytes(range(32,48)))
        with tempfile.TemporaryDirectory(prefix="tinyfl-v4-") as temp:
            roster=Roster(Path(temp)/"roster.json"); pair=FakeSerialPair(); messages=[]
            result=pc.provision(pair,mc,master,roster,out=messages.append)
            app=derive_app_key(master,"0123456789ABCDEF")
            values=(join,app,mc.mc_devaddr,mc.mc_nwkskey,mc.mc_appskey)
            self.assertEqual(pair.raw,b"".join(values)); self.assertEqual(len(pair.raw),60)
            self.assertEqual(provisioning_crc(*values),zlib.crc32(pair.raw)&0xffffffff)
            self.assertEqual(pair.join_lmic,join[::-1]); self.assertEqual(pair.address,0x12345678)
            for i, value in enumerate(values):
                raw=(U8*60)(); n=DLL.tfv4_nvs_get(i,raw); self.assertEqual(bytes(raw[:n]),value)
            out=(U8*8)(); DLL.tfv4_lmic_eui(out,buf(bytes.fromhex(result.dev_eui)))
            self.assertEqual(bytes(out),bytes.fromhex(result.dev_eui)[::-1])
            self.assertEqual(pair.received_count,1); self.assertTrue(result.new_in_roster); self.assertEqual(len(roster),1)
            self.assertTrue(DLL.tfv4_nvs_counter_untouched())
            log="\n".join(messages)
            for value in (master,app,mc.mc_nwkskey,mc.mc_appskey): self.assertNotIn(value.hex().lower(),log.lower())

    def test_provision_rejects_bad_crc_and_nvs_faults(self):
        mc=pc.KeyMaterial(bytes(range(8)),bytes(range(4)),bytes(range(16)),bytes(range(16)))
        for fault,badcrc in [(0,True),(1,False),(2,False)]:
            with tempfile.TemporaryDirectory(prefix="tinyfl-v4-") as temp:
                roster=Roster(Path(temp)/"roster.json"); pair=FakeSerialPair(fault=fault,badcrc=badcrc)
                with self.assertRaises(pc.ProvisionError): pc.provision(pair,mc,bytes(range(16)),roster,out=lambda _:None)
                self.assertEqual(len(roster),0); self.assertTrue(DLL.tfv4_nvs_counter_untouched())

    def test_provision_input_rejections(self):
        DLL.tfv4_nvs_reset(0)
        good=pc.keys_line(bytes(range(8)),bytes(range(16)),pc.KeyMaterial(bytes(range(8)),bytes(range(4)),bytes(range(16)),bytes(range(16))))
        for line in (good[:-5],good.replace("TFPROV","OTHER"),good+"junk",good.replace("0001","zz01",1)):
            raw,join,addr,ok=(U8*60)(),(U8*8)(),U32(),C.c_int()
            DLL.tfv4_provision(line.encode(),raw,join,C.byref(addr),C.byref(ok)); self.assertFalse(ok.value)

    def test_v4_u16_card_prepare_and_255_model_replay(self):
        rng=np.random.default_rng(20261001); n=70; train=30; cal=20
        rows=[]; data=bytearray()
        for a in range(n):
            amplitude=2500 if a<60 else 18000
            q=np.clip(rng.normal(0,amplitude,512),-32767,32767).astype("<i2")
            scale=np.float32(.001); data.extend(struct.pack("<f",scale)+q.tobytes())
            rows.append(rf.acquisition_22_f32((q.astype(np.float32)*scale)[None,:]))
        rows=np.asarray(rows,np.float32)
        def crc(b): return b+struct.pack("<I",zlib.crc32(b)&0xffffffff)
        for pos in (513,65534):
            with tempfile.TemporaryDirectory(prefix="tinyfl-v4-card-") as temp:
                root=Path(temp)
                header=struct.pack("<4sHI4H12I",b"TFR1",1,70,pos,0,1,1,n,512,
                                   0,train,train,cal,train+cal,10,60,10,0,0)
                (root/f"m{pos:04x}s0.raw").write_bytes(crc(header)+crc(data))
                chk=struct.pack("<4sHHIHH",b"TFC1",1,pos,train+cal,1,0)
                (root/f"m{pos:04x}.chk").write_bytes(crc(chk+rows[:train+cal].astype("<f4").tobytes()))
                ns,nt=C.c_int(),C.c_int(); check=(F32*3)(); centre=(F32*88)(); scale=(F32*88)()
                self.assertTrue(DLL.tfc_prepare(str(root).encode(),pos,C.byref(ns),C.byref(nt),check,centre,scale))
                self.assertEqual((ns.value,nt.value),(1,train)); self.assertLessEqual(check[1],1e-3)
                self.assertTrue((root/f"m{pos:04x}.x").exists())
                # No v3 truncation/decimal filenames or lingering scratch files.
                self.assertFalse(list(root.glob("*.tmp"))); self.assertFalse(list(root.glob("*.dif")))
                x=np.empty((train,22),np.float32)
                self.assertTrue(DLL.tfc_load_x(str(root).encode(),pos,x.ctypes.data_as(C.POINTER(F32)),train,C.byref(nt)))
                cs=np.array(centre[:22],np.float32); ss=np.array(scale[:22],np.float32)
                allx=np.abs(np.arcsinh((rows-cs)/ss)).astype(np.float32)
                np.testing.assert_allclose(x,allx[:train],rtol=1e-4,atol=1e-4)
                # All 255 phase states, one raw pass; model 0 is deliberately untrained.
                models=rng.normal(0,.05,(255,562)).astype(np.float32); models[0]=0
                scores=np.empty((n,255),np.float32); flags=np.empty((n,255),np.uint8); count=C.c_int()
                self.assertTrue(DLL.tfv4_replay_features(str(root).encode(),pos,255,
                    models.ctypes.data_as(C.POINTER(F32)),37,n,scores.ctypes.data_as(C.POINTER(F32)),
                    flags.ctypes.data_as(P8),C.byref(count)))
                self.assertEqual(count.value,n)
                for m in (0,7,206,207,254):
                    expected=np.empty(n,np.float32)
                    DLL.tfc_score(models[m].ctypes.data_as(C.POINTER(F32)),allx.ctypes.data_as(C.POINTER(F32)),n,
                                  expected.ctypes.data_as(C.POINTER(F32)))
                    np.testing.assert_allclose(scores[:,m],expected,rtol=1e-4,atol=1e-4)
                for m in range(255):
                    self.assertTrue(np.array_equal(flags[:,m].astype(bool),
                        dense_flags(scores[:,m],np.arange(train),np.arange(train,train+cal))))

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--report",type=Path); args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Interop))
    if args.report:
        args.report.write_text(json.dumps({"protocol":"v4 rev 4.3","tests":result.testsRun,
            "failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful(),
            "hardware_tested":False,"provisioning":"actual server provision() + C++ parser/NVS adapter, fake serial pair",
            "crc_bytes":60,"messages_round_tripped":10,"positions_tested":[0,513,65534],
            "replay_phases_tested":255,"synthetic_acquisitions":70,"resume_suffix_start":37,
            "actual_node_fsm_tested":True,"fake_hardware":True,"alarm_state_bytes_host":NODE.tfn_alarm_state_bytes()},indent=2)+"\n",encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())
