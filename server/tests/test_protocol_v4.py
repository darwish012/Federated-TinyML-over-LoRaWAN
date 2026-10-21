"""Protocol v4: framing, every message with pinned offsets, validation, and the int8 codec."""
from __future__ import annotations

import importlib
import math
import struct
import sys
import zlib

import numpy as np
import pytest

import protocol_v3 as p3
import protocol_v4 as p4
from study_plan import load_fl_modules, resolve_research_root


# ---------------------------------------------------------------------------
# Helpers: an independent view of the wire, with every offset written out by hand
# ---------------------------------------------------------------------------
def model(seed: int, scale: float = 0.05) -> p4.QuantisedModel:
    return p4.quantize_model(np.random.default_rng(seed).normal(0.0, scale, p4.N_PARAMS))


def u16(b, o): return struct.unpack_from("<H", b, o)[0]
def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def f32(b, o): return struct.unpack_from("<f", b, o)[0]


def header(frame: bytes):
    """(magic, version, round_id, frag_index, frag_count, body_len, type, body)."""
    return (frame[0], frame[1], u16(frame, 2), frame[4], frame[5], frame[6], frame[7], frame[8:])


# ---------------------------------------------------------------------------
# Sizes and header
# ---------------------------------------------------------------------------
def test_contract_constants():
    assert p4.VERSION == 0x04 and p4.MAGIC == 0xF1
    assert p4.HEADER_BYTES == 8 and p4.MAX_BODY_BYTES == 213 and p4.MAX_APPLICATION_BYTES == 221
    assert p4.UPDATE_BYTES == 598 and p4.GLOBAL_BYTES == 624
    assert (p4.UPDATE_CHUNK_BYTES, p4.GLOBAL_CHUNK_BYTES) == (211, 213)
    assert p4.UPDATE_FRAGMENTS == 3 and p4.GLOBAL_FRAGMENTS == 3
    assert p4.N_PARAMS == 562 and p4.LAYERS == (22, 12, 22)


def test_frame_header_layout_is_pinned():
    frame = p4.encode_frame(p4.MSG_ALARM, 0xBEEF, 2, 3, b"xyz")
    assert frame[:8] == bytes([0xF1, 0x04, 0xEF, 0xBE, 2, 3, 3, 0x07])
    assert header(frame)[:7] == (0xF1, 4, 0xBEEF, 2, 3, 3, 0x07)
    decoded = p4.decode_frame(p4.encode_frame(p4.MSG_UPDATE_FRAGMENT, 77, 1, 3, b"abc"))
    assert (decoded.message_type, decoded.round_id, decoded.fragment_index,
            decoded.fragment_count, decoded.data) == (1, 77, 1, 3, b"abc")


def test_message_type_numbers():
    assert (p4.MSG_UPDATE_FRAGMENT, p4.MSG_CLASS_C_READY, p4.MSG_GLOBAL_FRAGMENT,
            p4.MSG_GLOBAL_STATUS, p4.MSG_UPDATE_REPAIR, p4.MSG_BOARD_HELLO, p4.MSG_ALARM,
            p4.MSG_SELF_CHECK, p4.MSG_REPLAY_DONE, p4.MSG_RUN_CONTROL) == tuple(range(1, 11))


# ---------------------------------------------------------------------------
# UPDATE: 598 B, 211/211/176, offsets pinned
# ---------------------------------------------------------------------------
def make_update(position=1234, run_id=0xA1B2C3D4, phase=3, round_=7, base_slot=0x1234, rows=70000):
    delta = model(1)
    payload = p4.encode_update(position, run_id, phase, round_, base_slot, rows, delta, 0.125)
    return delta, payload


def test_update_offsets_are_pinned():
    delta, payload = make_update()
    assert len(payload) == 598
    assert u16(payload, 0) == 1234                              # position
    assert u32(payload, 2) == 0xA1B2C3D4                        # run_id
    assert payload[6] == 3 and payload[7] == 7                  # phase_index, round
    assert u16(payload, 8) == 0x1234                            # base_slot
    assert u16(payload, 10) == 0xFFFF                           # n_train_rows saturates at 65535
    for i in range(4):
        assert f32(payload, 12 + 4 * i) == pytest.approx(np.float32(delta.scales[i]))
    assert payload[28:590] == delta.codes.tobytes()             # delta[562], W1 b1 W2 b2
    assert f32(payload, 590) == 0.125                           # train_loss
    assert u32(payload, 594) == zlib.crc32(payload[:594]) & 0xFFFFFFFF
    assert np.frombuffer(payload[28 + 0:28 + 264], np.int8).tolist() == delta.codes[:264].tolist()
    assert payload[28 + 264:28 + 276] == delta.codes[264:276].tobytes()    # b1 starts at 264
    assert payload[28 + 276:28 + 540] == delta.codes[276:540].tobytes()    # W2 at 276
    assert payload[28 + 540:28 + 562] == delta.codes[540:562].tobytes()    # b2 at 540


def test_update_round_trip():
    delta, payload = make_update(rows=321)
    u = p4.decode_update(payload)
    assert (u.position, u.run_id, u.phase_index, u.round, u.base_slot, u.n_train_rows) == (
        1234, 0xA1B2C3D4, 3, 7, 0x1234, 321)
    assert np.array_equal(u.model.codes, delta.codes)
    assert u.model.scales == tuple(float(np.float32(s)) for s in delta.scales)
    assert u.train_loss == 0.125 and u.crc32 == u32(payload, 594)


def test_update_fragments_are_211_211_176_with_a_position_prefix():
    _, payload = make_update(position=0xFFFE - 1)
    frames = p4.fragment_update(payload, header_slot=999)
    assert [len(f) for f in frames] == [221, 221, 186]
    for index, frame in enumerate(frames):
        magic, version, slot, idx, count, body_len, mtype, body = header(frame)
        assert (magic, version, slot, idx, count, mtype) == (0xF1, 4, 999, index, 3, 0x01)
        assert body_len == len(body) == [213, 213, 178][index]
        assert u16(body, 0) == 0xFFFE - 1                       # u16 position prefix on every fragment
        chunk = payload[index * 211:(index + 1) * 211]
        assert body[2:] == chunk and len(chunk) == [211, 211, 176][index]
    assert b"".join(p4.decode_frame(f).data[2:] for f in frames) == payload


def test_update_fragment_reassembly_and_split():
    _, payload = make_update()
    frames = [p4.decode_frame(f) for f in p4.fragment_update(payload, 5)]
    partial = p4.new_update_partial()
    out = None
    for frame in (frames[2], frames[0], frames[1], frames[0]):    # out of order, one duplicate
        position, chunk = p4.split_update_fragment(frame)
        assert position == 1234
        out = partial.add(chunk) or out
    assert out == payload
    bad = p4.Frame(1, 5, 0, 3, b"\x01")
    with pytest.raises(p4.ProtocolError):
        p4.split_update_fragment(bad)                             # shorter than the u16 prefix
    with pytest.raises(p4.ProtocolError):
        p4.split_update_fragment(p4.Frame(1, 5, 0, 3, b"\xff\xff" + bytes(211)))   # position 0xFFFF


# ---------------------------------------------------------------------------
# GLOBAL (rev 4.3): 624 B, 213/213/198, offsets pinned
# ---------------------------------------------------------------------------
def test_global_offsets_are_pinned():
    m = model(2)
    payload = p4.encode_global(0xDEADBEEF, 6, 12, 12, m, p4.Participation.of_positions([300, 302, 399]))
    assert len(payload) == 624
    assert u32(payload, 0) == 0xDEADBEEF
    assert payload[4] == 6 and payload[5] == 12 and payload[6] == 12
    assert payload[7] == 0x01                                    # flags bit0: final; bit1 clear (mask in use)
    for i in range(4):
        assert f32(payload, 8 + 4 * i) == pytest.approx(np.float32(m.scales[i]))
    assert payload[24:586] == m.codes.tobytes()
    assert u16(payload, 586) == 300                              # mask_base
    mask = payload[588:620]
    assert len(mask) == 32
    assert mask[0] == 0b00000101 and mask[12] == 0b00001000      # 300 -> bit 0, 302 -> bit 2; 399 = 300 + 99 -> byte 12 bit 3
    assert sum(bin(b).count("1") for b in mask) == 3
    assert u32(payload, 620) == zlib.crc32(payload[:620]) & 0xFFFFFFFF
    initial = p4.encode_global(1, 0, 0, 12, m)
    assert initial[7] == 0x02 and initial[5] == 0                # default: every position takes part (bit1)
    assert u16(initial, 586) == 0 and initial[588:620] == bytes(32)
    final_all = p4.encode_global(1, 0, 12, 12, m, p4.Participation.everyone())
    assert final_all[7] == 0x03


def test_global_participation_round_trip_and_window_limits():
    m = model(5)
    for positions in ([0, 1, 2, 3], [5], [1000, 1255], list(range(7, 40, 3)), [65534 - 255, 65534]):
        part = p4.Participation.of_positions(positions)
        g = p4.decode_global(p4.encode_global(9, 1, 2, 12, m, part))
        assert not g.participation.all and g.participation.positions == tuple(sorted(positions))
        assert g.participation.mask_base == min(positions) and g.flags & 0x02 == 0
        assert all(g.participation.covers(p) for p in positions) and not g.participation.covers(max(positions) + 1)
    everyone = p4.decode_global(p4.encode_global(9, 1, 2, 12, m, p4.Participation.everyone()))
    assert everyone.participation.all and everyone.participation.covers(60000) and everyone.flags & 0x02
    for bad in ([0, 256], [10, 400], []):
        with pytest.raises(p4.ProtocolError):
            p4.Participation.of_positions(bad)                    # span >= 256, or empty
    p4.Participation.of_positions([0, 255])                       # exactly one window: fits


def test_global_round_trip_and_fragments():
    m = model(3)
    payload = p4.encode_global(77, 2, 5, 12, m, p4.Participation.of_positions(range(4)))
    g = p4.decode_global(payload)
    assert (g.run_id, g.phase_index, g.round, g.rounds_in_phase, g.flags, g.final) == (
        77, 2, 5, 12, 0, False)
    assert g.participation.positions == (0, 1, 2, 3)
    assert np.array_equal(g.model.codes, m.codes)
    frames = p4.fragment_global(payload, slot=4242)
    assert [len(f) for f in frames] == [221, 221, 206]
    for index, frame in enumerate(frames):
        _, _, slot, idx, count, body_len, mtype, body = header(frame)
        assert (slot, idx, count, mtype) == (4242, index, 3, 0x03)
        assert body == payload[index * 213:(index + 1) * 213]
        assert body_len == [213, 213, 198][index]
    partial = p4.new_global_partial()
    got = None
    for f in reversed(frames):
        got = partial.add(p4.decode_frame(f)) or got
    assert got == payload


# ---------------------------------------------------------------------------
# Control bodies: exact sizes and offsets
# ---------------------------------------------------------------------------
def test_class_c_ready_is_6_bytes():
    frame = p4.encode_class_c_ready(0x01020304, 0x0A0B)
    _, _, slot, idx, count, body_len, mtype, body = header(frame)
    assert (mtype, idx, count, body_len, slot) == (0x02, 0, 1, 6, 0x0A0B)
    assert u32(body, 0) == 0x01020304 and u16(body, 4) == 0x0A0B
    ready = p4.decode_class_c_ready(p4.decode_frame(frame))
    assert (ready.run_id, ready.awaited_slot) == (0x01020304, 0x0A0B)
    anyslot = p4.decode_class_c_ready(p4.decode_frame(p4.encode_class_c_ready(0, p4.NO_SLOT)))
    assert anyslot.awaited_slot == 0xFFFF                          # "any phase start"


def test_global_status_is_7_bytes():
    frame = p4.encode_global_status(9, 300, 0b101)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x04, 1, 7, 300)
    assert u32(body, 0) == 9 and u16(body, 4) == 300 and body[6] == 0b101
    status = p4.decode_global_status(p4.decode_frame(frame))
    assert (status.run_id, status.slot, status.missing_mask, status.complete) == (9, 300, 5, False)
    assert p4.decode_global_status(p4.decode_frame(p4.encode_global_status(9, 300, 0))).complete


def test_update_repair_is_7_bytes():
    frame = p4.encode_update_repair(slot=55, run_id=0xCAFE, position=4321, missing_mask=0b110)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x05, 1, 7, 55)
    assert u32(body, 0) == 0xCAFE and u16(body, 4) == 4321 and body[6] == 0b110
    rep = p4.decode_update_repair(p4.decode_frame(frame))
    assert (rep.run_id, rep.position, rep.missing_mask) == (0xCAFE, 4321, 6)


@pytest.mark.parametrize("n", [0, 1, 3, 8])
def test_board_hello_is_9_plus_2n(n):
    positions = [100 * (i + 1) + 7 for i in range(n)]
    frame = p4.encode_board_hello(513, 2, 0x0B, 0xABCDEF01, positions)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x06, 1, 9 + 2 * n, 0xFFFF)
    assert u16(body, 0) == 513 and body[2] == 2 and body[3] == 0x0B
    assert u32(body, 4) == 0xABCDEF01 and body[8] == n
    assert [u16(body, 9 + 2 * i) for i in range(n)] == positions
    hello = p4.decode_board_hello(p4.decode_frame(frame))
    assert (hello.board_id, hello.fw_version, hello.flags, hello.stored_run_id) == (
        513, 2, 0x0B, 0xABCDEF01)
    assert hello.positions == tuple(positions)
    assert hello.sd_mounted and hello.self_check_passed and not hello.imu_source and hello.run_complete


def test_alarm_is_19_bytes():
    frame = p4.encode_alarm(0x11223344, 777, 5, 123456, 9.5, 3.25)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x07, 1, 19, 0xFFFF)
    assert u32(body, 0) == 0x11223344 and u16(body, 4) == 777 and body[6] == 5
    assert u32(body, 7) == 123456 and f32(body, 11) == 9.5 and f32(body, 15) == 3.25
    a = p4.decode_alarm(p4.decode_frame(frame))
    assert (a.run_id, a.position, a.phase_index, a.acquisition_index, a.score, a.threshold) == (
        0x11223344, 777, 5, 123456, 9.5, 3.25)


def test_self_check_is_12_bytes():
    frame = p4.encode_self_check(65000, 48, 1.5, 0.25)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x08, 1, 12, 0xFFFF)
    assert u16(body, 0) == 65000 and u16(body, 2) == 48 and f32(body, 4) == 1.5 and f32(body, 8) == 0.25
    c = p4.decode_self_check(p4.decode_frame(frame))
    assert (c.position, c.rows_checked, c.max_abs_diff, c.median_abs_diff) == (65000, 48, 1.5, 0.25)


def test_replay_done_is_16_bytes_with_the_first_alarm_index():
    frame = p4.encode_replay_done(0xAABBCCDD, 31, 254, 99999, 1, 4321)
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x09, 1, 16, 0xFFFF)
    assert u32(body, 0) == 0xAABBCCDD and u16(body, 4) == 31 and body[6] == 254     # phase_index u8
    assert u32(body, 7) == 99999 and body[11] == 1                                  # alarm_latched u8
    assert u32(body, 12) == 4321                                                    # first_alarm_index
    d = p4.decode_replay_done(p4.decode_frame(frame))
    assert (d.run_id, d.position, d.phase_index, d.acquisitions_scored, d.alarm_latched,
            d.first_alarm_index, d.consistent) == (0xAABBCCDD, 31, 254, 99999, 1, 4321, True)
    none = p4.decode_replay_done(p4.decode_frame(p4.encode_replay_done(1, 2, 0, 3, 0)))
    assert none.alarm_latched == 0 and none.first_alarm_index == 0xFFFFFFFF and none.consistent
    assert u32(header(p4.encode_replay_done(1, 2, 0, 3, 0))[7], 12) == 0xFFFFFFFF
    # the codec decodes a violating frame (the controller logs it); consistent says so
    assert not p4.decode_replay_done(p4.decode_frame(p4.encode_replay_done(1, 2, 0, 3, 1, 0xFFFFFFFF))).consistent
    assert not p4.decode_replay_done(p4.decode_frame(p4.encode_replay_done(1, 2, 0, 3, 0, 5))).consistent
    # the rev 4.2 12-byte form is refused
    with pytest.raises(p4.ProtocolError):
        p4.decode_replay_done(p4.decode_frame(p4.encode_frame(
            p4.MSG_REPLAY_DONE, 0xFFFF, 0, 1, struct.pack("<IHBIB", 1, 2, 0, 3, 0))))


def test_run_control_layout():
    frame = p4.encode_run_control(0x12345678, p4.CMD_REPLAY, [0, 3, 6])
    _, _, slot, _, count, body_len, mtype, body = header(frame)
    assert (mtype, count, body_len, slot) == (0x0A, 1, 9, 0xFFFF)
    assert u32(body, 0) == 0x12345678 and body[4] == 1 and body[5] == 3 and body[6:9] == bytes([0, 3, 6])
    rc = p4.decode_run_control(p4.decode_frame(frame))
    assert (rc.run_id, rc.command, rc.phases) == (0x12345678, 1, (0, 3, 6))
    end = p4.decode_run_control(p4.decode_frame(p4.encode_run_control(5, p4.CMD_END)))
    assert (end.command, end.phases) == (2, ())
    adopt = p4.encode_run_control(0x0A0B0C0D, p4.CMD_ADOPT)
    assert adopt[8:] == struct.pack("<IBB", 0x0A0B0C0D, 3, 0) and len(adopt) == 8 + 6
    a = p4.decode_run_control(p4.decode_frame(adopt))
    assert (a.run_id, a.command, a.phases) == (0x0A0B0C0D, p4.CMD_ADOPT, ())
    assert p4.MAX_PHASES_PER_RUN_CONTROL == 207
    big = p4.encode_run_control(5, p4.CMD_REPLAY, range(207))
    assert len(big) == 221


# ---------------------------------------------------------------------------
# Validation before storage
# ---------------------------------------------------------------------------
def _frame_with(**over):
    base = dict(magic=0xF1, version=4, round_id=1, idx=0, count=1, body_len=None, mtype=0x04,
                body=bytes(7))
    base.update(over)
    body = base["body"]
    blen = len(body) if base["body_len"] is None else base["body_len"]
    return (bytes([base["magic"], base["version"]]) + struct.pack("<H", base["round_id"])
            + bytes([base["idx"], base["count"], blen, base["mtype"]]) + body)


@pytest.mark.parametrize("kwargs, why", [
    (dict(magic=0xF2), "magic"),
    (dict(version=3), "version"),
    (dict(version=5), "version"),
    (dict(mtype=0x0B), "type"),
    (dict(mtype=0x00), "type"),
    (dict(idx=1, count=1), "index"),
    (dict(count=0, idx=0), "index"),
    (dict(count=3, idx=0), "control frame with 3 fragments"),
    (dict(mtype=0x01, count=1, body=bytes(5)), "fragment type with count 1"),
    (dict(mtype=0x03, count=2, body=bytes(5)), "fragment type with count 2"),
    (dict(body_len=3), "body_len mismatch"),
    (dict(body=bytes(214), body_len=214), "body over 213"),
])
def test_decode_frame_rejects_before_storing(kwargs, why):
    with pytest.raises(p4.ProtocolError):
        p4.decode_frame(_frame_with(**kwargs))


def test_decode_frame_length_bounds():
    for bad in (b"", b"\xf1\x04\x00", bytes(222)):
        with pytest.raises(p4.ProtocolError):
            p4.decode_frame(bad)
    with pytest.raises(p4.ProtocolError):
        p4.encode_frame(1, 0, 0, 3, bytes(214))
    with pytest.raises(p4.ProtocolError):
        p4.encode_frame(1, 0x10000, 0, 3, b"")


def test_a_v3_frame_is_rejected_by_the_v4_decoder_and_vice_versa():
    v3 = p3.encode_global_status(5, 0)
    with pytest.raises(p4.ProtocolError):
        p4.decode_frame(v3)
    with pytest.raises(p3.ProtocolError):
        p3.decode_frame(p4.encode_global_status(1, 1, 0))


def test_update_validation():
    delta, payload = make_update()
    flipped = bytearray(payload)
    flipped[100] ^= 0x01
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(bytes(flipped))                           # CRC
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(payload[:-1])
    codes = delta.codes.copy()
    codes[10] = -128
    bad = p4.append_crc(payload[:28] + codes.tobytes() + payload[590:594])
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(bad)                                      # int8 -128 is invalid
    for scale in (0.0, -1.0, math.inf, math.nan):
        raw = bytearray(payload[:594])
        struct.pack_into("<f", raw, 12, scale)
        with pytest.raises(p4.ProtocolError):
            p4.decode_update(p4.append_crc(bytes(raw)))
    raw = bytearray(payload[:594])
    raw[7] = 0                                                      # round 0 is not an UPDATE round
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(p4.append_crc(bytes(raw)))
    raw = bytearray(payload[:594])
    raw[6] = 255                                                    # phase_index 255 is out of range
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(p4.append_crc(bytes(raw)))
    raw = bytearray(payload[:594])
    struct.pack_into("<H", raw, 0, 0xFFFF)                          # position 0xFFFF = none
    with pytest.raises(p4.ProtocolError):
        p4.decode_update(p4.append_crc(bytes(raw)))


def test_global_validation():
    m = model(4)
    good = p4.encode_global(1, 0, 3, 12, m)
    for index, value in ((7, 1), (7, 4), (7, 0x80)):                # final flag without round == R; reserved bits
        raw = bytearray(good[:620])
        raw[index] = value
        with pytest.raises(p4.ProtocolError):
            p4.decode_global(p4.append_crc(bytes(raw)))
    p4.decode_global(p4.append_crc(bytes(bytearray(good[:620][:7]) + b"\x02" + good[8:620])))   # bit1 is fine
    with pytest.raises(p4.ProtocolError):
        p4.decode_global(p4.encode_global(1, 0, 3, 12, m)[:-4])   # a rev 4.2 length
    old = p4.append_crc(good[:586])                                  # a 590-byte rev 4.2 GLOBAL
    with pytest.raises(p4.ProtocolError):
        p4.decode_global(old)
    raw = bytearray(good[:620])
    raw[5] = 13                                                      # round > rounds_in_phase
    with pytest.raises(p4.ProtocolError):
        p4.decode_global(p4.append_crc(bytes(raw)))
    raw = bytearray(good[:620])
    raw[6] = 0                                                       # rounds_in_phase 0
    with pytest.raises(p4.ProtocolError):
        p4.decode_global(p4.append_crc(bytes(raw)))
    flipped = bytearray(good)
    flipped[50] ^= 1
    with pytest.raises(p4.ProtocolError):
        p4.decode_global(bytes(flipped))
    with pytest.raises(p4.ProtocolError):
        p4.encode_global(1, 0, 13, 12, m)


@pytest.mark.parametrize("build", [
    lambda: p4.encode_global_status(1, 1, 8),                       # bits outside 0..2
    lambda: p4.encode_update_repair(1, 1, 5, 0),                    # empty repair mask
    lambda: p4.encode_update_repair(1, 1, 5, 8),
    lambda: p4.encode_update_repair(1, 1, 0xFFFF, 1),               # position 0xFFFF
    lambda: p4.encode_board_hello(1, 1, 3, 0, range(9)),            # n > 8
    lambda: p4.encode_board_hello(1, 1, 3, 0, [0xFFFF]),
    lambda: p4.encode_run_control(1, 4),                            # unknown command
    lambda: p4.encode_run_control(1, p4.CMD_ADOPT, [0]),            # ADOPT carries no phases
    lambda: p4.encode_run_control(1, p4.CMD_REPLAY, []),            # REPLAY lists at least one
    lambda: p4.encode_replay_done(1, 2, 255, 3, 0),                 # phase_index 255
    lambda: p4.encode_replay_done(1, 2, 3, 3, 2),                   # alarm_latched is 0/1
    lambda: p4.encode_run_control(1, 1, [1, 1]),                    # duplicate phase
    lambda: p4.encode_run_control(1, 1, [255]),
    lambda: p4.encode_run_control(1, 1, range(208)),                # does not fit one frame
])
def test_encoders_refuse_invalid_bodies(build):
    with pytest.raises(p4.ProtocolError):
        build()


def _raw_control(mtype, body, round_id=0xFFFF):
    return p4.decode_frame(p4.encode_frame(mtype, round_id, 0, 1, body))


@pytest.mark.parametrize("mtype, decoder, good_size", [
    (p4.MSG_CLASS_C_READY, p4.decode_class_c_ready, 6),
    (p4.MSG_GLOBAL_STATUS, p4.decode_global_status, 7),
    (p4.MSG_UPDATE_REPAIR, p4.decode_update_repair, 7),
    (p4.MSG_ALARM, p4.decode_alarm, 19),
    (p4.MSG_SELF_CHECK, p4.decode_self_check, 12),
    (p4.MSG_REPLAY_DONE, p4.decode_replay_done, 16),
])
def test_control_bodies_must_have_their_exact_size(mtype, decoder, good_size):
    for size in (good_size - 1, good_size + 1, 0):
        with pytest.raises(p4.ProtocolError):
            decoder(_raw_control(mtype, bytes(size)))


def test_other_control_body_validation():
    with pytest.raises(p4.ProtocolError):                          # header slot != body slot
        p4.decode_class_c_ready(_raw_control(p4.MSG_CLASS_C_READY, struct.pack("<IH", 1, 9), round_id=8))
    with pytest.raises(p4.ProtocolError):
        p4.decode_global_status(_raw_control(p4.MSG_GLOBAL_STATUS, struct.pack("<IHB", 1, 9, 0), round_id=8))
    with pytest.raises(p4.ProtocolError):                          # mask bit 3 set
        p4.decode_global_status(_raw_control(p4.MSG_GLOBAL_STATUS, struct.pack("<IHB", 1, 9, 8), round_id=9))
    with pytest.raises(p4.ProtocolError):                          # repair mask empty
        p4.decode_update_repair(_raw_control(p4.MSG_UPDATE_REPAIR, struct.pack("<IHB", 1, 9, 0)))
    for body in (b"\x00" * 8, b"\x00" * 9 + b"\x09"):               # hello: short head / n=9
        with pytest.raises(p4.ProtocolError):
            p4.decode_board_hello(_raw_control(p4.MSG_BOARD_HELLO, body))
    with pytest.raises(p4.ProtocolError):                          # hello: n=2 but one position
        p4.decode_board_hello(_raw_control(p4.MSG_BOARD_HELLO, struct.pack("<HBBIB", 1, 1, 3, 0, 2) + b"\x01\x00"))
    with pytest.raises(p4.ProtocolError):                          # hello: duplicate position
        p4.decode_board_hello(_raw_control(
            p4.MSG_BOARD_HELLO, struct.pack("<HBBIB", 1, 1, 3, 0, 2) + b"\x01\x00\x01\x00"))
    with pytest.raises(p4.ProtocolError):
        p4.decode_run_control(_raw_control(p4.MSG_RUN_CONTROL, struct.pack("<IBB", 1, 1, 2) + b"\x00"))
    with pytest.raises(p4.ProtocolError):
        p4.decode_run_control(_raw_control(p4.MSG_RUN_CONTROL, struct.pack("<IBB", 1, 9, 0)))


def test_device_id_helpers():
    assert p4.dev_eui_of_device_id("eui-70b3d57ed0000001") == "70B3D57ED0000001"
    assert p4.device_id_of_dev_eui("70B3D57ED0000001") == "eui-70b3d57ed0000001"
    for bad in ("eui-123", "board-1", "eui-zzzzzzzzzzzzzzzz", ""):
        with pytest.raises(p4.ProtocolError):
            p4.dev_eui_of_device_id(bad)


# ---------------------------------------------------------------------------
# int8 codec == quant_fl.Q_DYN_TENSOR, bit for bit (the codec is v3's, re-exported)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def research():
    try:
        load_fl_modules()
    except FileNotFoundError:
        pytest.skip("research folder not available")
    root = resolve_research_root(None)
    for name in ("simulation_completion_20260929", "hardware_twin_scaling_20260929"):
        if str(root / name) not in sys.path:
            sys.path.append(str(root / name))
    return {"quant_fl": importlib.import_module("quant_fl")}


def test_codec_is_the_v3_codec():
    assert p4.quantize_model is p3.quantize_model
    assert p4.QuantisedModel is p3.QuantisedModel
    assert p4.PartialMessage is p3.PartialMessage


def _vectors():
    rng = np.random.default_rng(20261001)
    out = [rng.normal(0, 10.0 ** rng.uniform(-6, 2), p4.N_PARAMS) for _ in range(30)]
    ties = (np.arange(p4.N_PARAMS) % 255 - 127 + 0.5) * 0.01
    ties[0], ties[264], ties[276], ties[540] = 1.27, -1.27, 1.27, -1.27
    return out + [np.zeros(p4.N_PARAMS), np.full(p4.N_PARAMS, 0.3), ties]


def test_codec_matches_quant_fl_q_dyn_tensor(research):
    quant_fl = research["quant_fl"]
    for vec in _vectors():
        expected, _ = quant_fl.downlink_quantize(vec, "Q_DYN_TENSOR")
        m = p4.quantize_model(vec)
        assert m.codes.dtype == np.int8
        assert int(m.codes.min()) >= -127 and int(m.codes.max()) <= 127
        assert np.array_equal(m.dequantize(), expected)


def test_codec_survives_a_trip_through_the_v4_global_and_update_messages(research):
    quant_fl = research["quant_fl"]
    vec = np.random.default_rng(5).normal(0, 0.07, p4.N_PARAMS)
    m = p4.quantize_model(vec)
    g = p4.decode_global(p4.encode_global(1, 0, 4, 12, m))
    u = p4.decode_update(p4.encode_update(3, 1, 0, 4, 9, 100, m, 0.5))
    wire = p4.QuantisedModel(m.codes, m.wire_scales())
    assert np.array_equal(g.model.dequantize(), wire.dequantize())
    assert np.array_equal(u.model.dequantize(), wire.dequantize())
    expected, _ = quant_fl.downlink_quantize(vec, "Q_DYN_TENSOR")
    assert np.allclose(g.model.dequantize(), expected, rtol=1e-6, atol=1e-9)   # float32 scales
