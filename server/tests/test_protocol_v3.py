"""Protocol v3: framing, message codecs and the int8 codec against quant_fl."""
from __future__ import annotations

import importlib
import struct
import sys
import zlib

import numpy as np
import pytest

import protocol_v3 as p3
from protocol import ProtocolError, validate_crc
from study_plan import load_fl_modules, resolve_research_root


# ---------------------------------------------------------------------------
# References from the research folder (skipped, not failed, if it is absent)
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
    return {
        "quant_fl": importlib.import_module("quant_fl"),
        "twin": importlib.import_module("twin"),
        "fl": importlib.import_module("fl"),
    }


def random_model(seed: int, scale: float = 0.05) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, p3.N_PARAMS)


# ---------------------------------------------------------------------------
# Sizes and frames
# ---------------------------------------------------------------------------
def test_contract_sizes():
    assert p3.VERSION == 0x03
    assert (p3.UPDATE_BYTES, p3.GLOBAL_BYTES) == (594, 586)
    assert (p3.UPDATE_FRAGMENTS, p3.GLOBAL_FRAGMENTS) == (3, 3)
    assert p3.N_PARAMS == 562 and sum(p3.TENSOR_SIZES) == 562
    assert p3.HARDWARE_LEVELS == (4, 8, 12, 16, 20, 24, 28)


def test_update_fragments_carry_a_position_byte_and_213_payload_bytes():
    delta = p3.quantize_model(random_model(1))
    payload = p3.encode_update(5, 2, 7, 300, delta, 0.25)
    assert len(payload) == 594
    frames = p3.fragment_update(payload, 31)
    assert p3.UPDATE_CHUNK_BYTES == 213 and len(frames) == p3.UPDATE_FRAGMENTS == 3
    assert [len(f) for f in frames] == [221, 221, 176]                  # 213 + 213 + 168 payload bytes, each + the position byte
    assert all(f[1] == 0x03 for f in frames)
    assert all(f[7] == 5 for f in frames)                            # every fragment names its position
    assert b"".join(f[8:] for f in frames) == payload
    assert [f[4] for f in frames] == [0, 1, 2] and all(f[5] == 3 for f in frames)
    for frame in frames:
        position, chunk = p3.split_update_fragment(p3.decode_frame(frame))
        assert position == 5 and len(chunk.data) <= 213


def test_global_fragments():
    payload = p3.encode_global(1, 12, p3.quantize_model(random_model(2)))
    assert len(payload) == 586
    frames = p3.fragment_payload(payload, p3.MSG_GLOBAL_FRAGMENT, 20)
    assert [len(f) for f in frames] == [221, 221, 165]           # bodies 214 / 214 / 158


def test_v2_frames_are_refused():
    frame = bytearray(p3.encode_class_c_ready(0, 5, 5))
    frame[1] = 0x02
    with pytest.raises(ProtocolError, match="v3"):
        p3.decode_frame(bytes(frame))


def test_frame_length_and_type_validation():
    with pytest.raises(ProtocolError):
        p3.decode_frame(b"\xf1\x03\x01")
    frame = bytearray(p3.encode_class_c_ready(0, 5, 5))
    frame[2] = 0x63
    with pytest.raises(ProtocolError, match="unsupported"):
        p3.decode_frame(bytes(frame))
    with pytest.raises(ProtocolError):
        p3.decode_frame(p3.encode_class_c_ready(0, 5, 5) + b"\x00")


def test_round_id_is_level_times_thirteen_plus_g():
    assert p3.ROUND_SLOTS == 13
    assert p3.global_round_id(0, 0) == 0
    assert p3.global_round_id(0, 12) == 12 and p3.global_round_id(1, 0) == 13
    assert p3.global_round_id(6, 12) == 90
    # UPDATE of round r (payload round r-1) carries g = r
    assert p3.update_round_id(0, 0) == 1 and p3.update_round_id(6, 11) == 90
    assert p3.NO_ROUND == 0xFF
    assert p3.level_index_of(28) == 6
    assert p3.board_of_position(0) == 1 and p3.board_of_position(19) == 10
    assert p3.board_of_position(27) == 8


# ---------------------------------------------------------------------------
# UPDATE / GLOBAL round trips
# ---------------------------------------------------------------------------
def test_update_round_trip():
    delta = p3.quantize_model(random_model(3))
    payload = p3.encode_update(27, 6, 11, 65535, delta, 0.125, run_id=0xBEEF, base_slot=87)
    update = p3.decode_update(payload)
    assert (update.position, update.level_index, update.round) == (27, 6, 11)
    assert (update.run_id, update.base_slot) == (0xBEEF, 87)
    assert update.n_train_rows == 65535
    assert update.train_loss == 0.125
    assert np.array_equal(update.model.codes, delta.codes)
    assert update.model.scales == delta.wire_scales()
    assert update.crc32 == zlib.crc32(payload[:-4]) & 0xFFFFFFFF


def test_global_round_trip():
    model = p3.quantize_model(random_model(4))
    payload = p3.encode_global(3, 12, model, run_id=0x1234)
    decoded = p3.decode_global(payload)
    assert (decoded.level_index, decoded.round, decoded.run_id) == (3, 12, 0x1234)
    assert np.array_equal(decoded.model.codes, model.codes)
    assert decoded.model.scales == model.wire_scales()
    assert decoded.crc32 == validate_crc(payload, 586)


def test_fields_are_little_endian_at_the_documented_offsets():
    """PROTOCOL_V3 3.1 / 3.2 exact offsets (rev 3.2)."""
    delta = p3.quantize_model(random_model(5))
    payload = p3.encode_update(9, 4, 2, 0x0102, delta, 1.5, run_id=0xA1B2, base_slot=54)
    assert len(payload) == 594
    assert payload[0:3] == bytes((9, 4, 2))                  # position, level_index, round
    assert payload[3:5] == b"\xb2\xa1"                       # u16 run_id at 3
    assert payload[5] == 54                                  # u8 base_slot at 5
    assert payload[6:8] == b"\x02\x01"                       # u16 n_train_rows at 6
    assert struct.unpack_from("<4f", payload, 8) == delta.wire_scales()       # scales at 8
    assert np.array_equal(np.frombuffer(payload, np.int8, 562, 24), delta.codes)   # delta at 24
    assert struct.unpack_from("<f", payload, 586)[0] == 1.5  # train_loss at 586
    assert struct.unpack_from("<I", payload, 590)[0] == zlib.crc32(payload[:590]) & 0xFFFFFFFF
    global_payload = p3.encode_global(6, 12, delta, run_id=0xC0DE)
    assert len(global_payload) == 586
    assert global_payload[:2] == bytes((6, 12))
    assert global_payload[2:4] == b"\xde\xc0"                # u16 run_id at 2
    assert struct.unpack_from("<4f", global_payload, 4) == delta.wire_scales()
    assert np.array_equal(np.frombuffer(global_payload, np.int8, 562, 20), delta.codes)
    assert struct.unpack_from("<I", global_payload, 582)[0] == zlib.crc32(global_payload[:582]) & 0xFFFFFFFF


@pytest.mark.parametrize("offset", [0, 3, 5, 6, 10, 100, 586, 590])
def test_any_corrupted_update_byte_fails_the_crc(offset):
    payload = bytearray(p3.encode_update(1, 0, 0, 10, p3.quantize_model(random_model(6)), 0.1))
    payload[offset] ^= 0x10
    with pytest.raises(ProtocolError):
        p3.decode_update(bytes(payload))


def test_minus_128_code_is_refused():
    model = p3.quantize_model(random_model(7))
    codes = model.codes.copy()
    codes[10] = -128
    bad = p3.QuantisedModel(codes=codes, scales=model.scales)
    with pytest.raises(ProtocolError, match="-128"):
        p3.decode_update(p3.encode_update(1, 0, 0, 10, bad, 0.1))
    with pytest.raises(ProtocolError, match="-128"):
        p3.decode_global(p3.encode_global(0, 0, bad))


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_scale_is_refused(bad_scale):
    model = p3.quantize_model(random_model(8))
    bad = p3.QuantisedModel(codes=model.codes, scales=(bad_scale,) + tuple(model.scales[1:]))
    with pytest.raises(ProtocolError, match="scale"):
        p3.decode_global(p3.encode_global(0, 0, bad))


def test_out_of_range_fields_are_refused_on_encode():
    delta = p3.quantize_model(random_model(9))
    for args in ((54, 0, 0), (0, 7, 0), (0, 0, 12)):
        with pytest.raises(ProtocolError):
            p3.encode_update(*args, 10, delta, 0.0)
    with pytest.raises(ProtocolError):
        p3.encode_global(0, 13, delta)


# ---------------------------------------------------------------------------
# Control messages
# ---------------------------------------------------------------------------
def test_update_repair_carries_a_position_byte():
    frame_bytes = p3.encode_repair_request(37, 19, 0b101)
    assert len(frame_bytes) == 7 + 3
    frame = p3.decode_frame(frame_bytes)
    assert frame.message_type == p3.MSG_UPDATE_REPAIR and frame.round_id == 37
    assert frame.data == bytes((0b101, 0, 19))            # u16 mask (LE) + u8 position = 3 bytes
    assert p3.decode_repair_request(frame) == (0b101, 19)
    for bad_mask in (0, 0b1000, 0x8000):                   # empty or bits outside 0..2
        bad = p3.Frame(p3.MSG_UPDATE_REPAIR, 0, 0, 3, bad_mask.to_bytes(2, "little") + b"\x05")
        with pytest.raises(ProtocolError):
            p3.decode_repair_request(bad)
    with pytest.raises(ProtocolError):
        p3.decode_repair_request(p3.Frame(p3.MSG_UPDATE_REPAIR, 0, 0, 3, b"\x01\x00"))
    with pytest.raises(ProtocolError):
        p3.encode_repair_request(0, 0, 0)
    with pytest.raises(ProtocolError):
        p3.encode_repair_request(0, 0, 0b1000)


def test_control_messages_have_exact_bodies_and_no_crc():
    hello_frame = p3.encode_board_hello(7, 3, 12, 0b011, run_id=0xABCD)
    assert len(hello_frame) == 7 + 6 and hello_frame[3] == 0xFF
    hello = p3.decode_board_hello(p3.decode_frame(hello_frame))
    assert (hello.board_number, hello.n_machines, hello.fw_version, hello.run_id) == (7, 3, 12, 0xABCD)
    assert hello.sd_mounted and hello.self_check_passed and not hello.imu_source
    assert p3.decode_frame(hello_frame).data == bytes((7, 3, 12, 3, 0xCD, 0xAB))
    assert p3.decode_board_hello(
        p3.decode_frame(p3.encode_board_hello(1, 0, 0, 0b100))).imu_source
    assert p3.decode_board_hello(p3.decode_frame(p3.encode_board_hello(1, 0, 0, 3))).run_id == 0
    with pytest.raises(ProtocolError, match="6 bytes"):          # the old 4-byte body
        p3.decode_board_hello(p3.Frame(p3.MSG_BOARD_HELLO, 0xFF, 0, 1, bytes((1, 3, 1, 3))))

    alarm_frame = p3.encode_alarm(19, 4, 123456, 3.5, 2.25)
    assert len(alarm_frame) == 7 + 14
    assert p3.decode_alarm(p3.decode_frame(alarm_frame)) == p3.Alarm(19, 4, 123456, 3.5, 2.25)

    check_frame = p3.encode_self_check(4, 800, 0.5, 0.125)
    assert len(check_frame) == 7 + 11
    assert p3.decode_self_check(p3.decode_frame(check_frame)) == p3.SelfCheck(4, 800, 0.5, 0.125)

    replay_frame = p3.encode_replay_done(19, 0b1000101, 123456, 2)
    assert len(replay_frame) == 7 + 7 and p3.decode_frame(replay_frame).message_type == 0x09
    replay = p3.decode_replay_done(p3.decode_frame(replay_frame))
    assert replay == p3.ReplayDone(19, 0b1000101, 123456, 2)
    assert replay.levels() == [0, 2, 6]
    assert p3.decode_frame(replay_frame).data == bytes((19, 0b1000101)) + (123456).to_bytes(4, "little") + b"\x02"
    for decode, short in ((p3.decode_alarm, p3.MSG_ALARM), (p3.decode_self_check, p3.MSG_SELF_CHECK),
                          (p3.decode_replay_done, p3.MSG_REPLAY_DONE)):
        with pytest.raises(ProtocolError):
            decode(p3.Frame(short, 0xFF, 0, 1, b"\x00" * 3))


def test_status_and_ready_shapes():
    status_frame = p3.encode_global_status(5, 0b010)
    assert len(status_frame) == 7 + 2                              # a 2-byte bitmap, no CRC
    status = p3.decode_frame(status_frame)
    assert status.message_type == p3.MSG_GLOBAL_STATUS and status.data == bytes((0b010, 0))
    assert p3.decode_global_status(status) == 0b010
    with pytest.raises(ProtocolError):
        p3.encode_global_status(5, 0b1000)
    with pytest.raises(ProtocolError):                             # the old 6-byte body
        p3.decode_global_status(p3.Frame(p3.MSG_GLOBAL_STATUS, 5, 0, 3, bytes(6)))
    with pytest.raises(ProtocolError):                             # bits outside 0..2
        p3.decode_global_status(p3.Frame(p3.MSG_GLOBAL_STATUS, 5, 0, 3, b"\x08\x00"))
    ready = p3.decode_frame(p3.encode_class_c_ready(5, 5, 5))
    assert ready.data == bytes((5, 5)) and ready.fragment_count == 3


def test_frame_header_is_validated_before_anything_is_stored():
    """type known, index < count, count == 3 for fragments, single frames declare 1."""
    good = p3.fragment_update(p3.encode_update(0, 0, 0, 1, p3.quantize_model(random_model(11)), 0.0), 1)
    for offset, value, match in ((4, 3, "not below"), (4, 7, "not below"), (5, 2, "expected 3"),
                                 (5, 4, "expected 3")):
        frame = bytearray(good[0])
        frame[offset] = value
        with pytest.raises(ProtocolError, match=match):
            p3.decode_frame(bytes(frame))
    global_frame = bytearray(p3.fragment_payload(
        p3.encode_global(0, 0, p3.quantize_model(random_model(12))), p3.MSG_GLOBAL_FRAGMENT, 0)[0])
    global_frame[5] = 2
    with pytest.raises(ProtocolError, match="expected 3"):
        p3.decode_frame(bytes(global_frame))
    hello = bytearray(p3.encode_board_hello(1, 1, 1, 3))
    hello[5] = 3
    with pytest.raises(ProtocolError, match="single-frame"):
        p3.decode_frame(bytes(hello))
    zero_count = bytearray(p3.encode_class_c_ready(0, 5, 5))
    zero_count[5] = 0
    with pytest.raises(ProtocolError, match="not below"):
        p3.decode_frame(bytes(zero_count))


def test_partial_message_polices_lengths_and_duplicates():
    payload = p3.encode_update(0, 0, 0, 1, p3.quantize_model(random_model(10)), 0.0)
    frames = [p3.split_update_fragment(p3.decode_frame(f))[1] for f in p3.fragment_update(payload, 0)]
    partial = p3.PartialMessage(p3.UPDATE_BYTES, p3.UPDATE_FRAGMENTS, p3.UPDATE_CHUNK_BYTES)
    assert partial.missing_mask() == 0b111
    assert partial.add(frames[2]) is None
    assert partial.missing_mask() == 0b011
    assert partial.add(frames[0]) is None
    assert partial.add(frames[1]) == payload
    truncated = p3.Frame(frames[0].message_type, 0, 0, 3, frames[0].data[:-1])
    with pytest.raises(ProtocolError, match="expected"):
        p3.PartialMessage(p3.UPDATE_BYTES, 3, p3.UPDATE_CHUNK_BYTES).add(truncated)


def test_a_conflicting_duplicate_fragment_is_refused_and_not_stored():
    payload = p3.encode_update(0, 0, 0, 1, p3.quantize_model(random_model(13)), 0.0)
    frames = [p3.split_update_fragment(p3.decode_frame(f))[1] for f in p3.fragment_update(payload, 0)]
    partial = p3.PartialMessage(p3.UPDATE_BYTES, 3, p3.UPDATE_CHUNK_BYTES)
    partial.add(frames[1])
    altered = p3.Frame(frames[1].message_type, 0, 1, 3, bytes(len(frames[1].data)))
    with pytest.raises(p3.ConflictingFragment):
        partial.add(altered)
    assert partial.parts[1] == frames[1].data                   # the original is kept
    assert partial.add(frames[1]) is None                       # an identical duplicate is fine


# ---------------------------------------------------------------------------
# int8 codec == quant_fl.Q_DYN_TENSOR, bit for bit
# ---------------------------------------------------------------------------
def _vectors():
    rng = np.random.default_rng(20260930)
    out = []
    for i in range(40):
        magnitude = 10.0 ** rng.uniform(-6, 2)
        out.append(rng.normal(0, magnitude, p3.N_PARAMS))
    # Structured cases: heavy tails, sparse, a tiny tensor next to a big one,
    # all zeros (scale floor), constant, and values that land exactly on .5.
    heavy = rng.standard_t(2, p3.N_PARAMS)
    sparse = np.where(rng.random(p3.N_PARAMS) < 0.9, 0.0, rng.normal(0, 1, p3.N_PARAMS))
    mixed = rng.normal(0, 1e-4, p3.N_PARAMS)
    mixed[p3.TENSOR_SLICES[2]] *= 1e4
    zeros = np.zeros(p3.N_PARAMS)
    const = np.full(p3.N_PARAMS, -0.3)
    ties = np.empty(p3.N_PARAMS)
    ties[:] = (np.arange(p3.N_PARAMS) % 255 - 127) + 0.5
    ties *= 0.01
    ties[0], ties[264] = 1.27, -1.27                    # pin each tensor's max so scale = 0.01
    ties[276], ties[540] = 1.27, -1.27
    return out + [heavy, sparse, mixed, zeros, const, ties]


def test_codec_matches_quant_fl_q_dyn_tensor(research):
    quant_fl = research["quant_fl"]
    for vec in _vectors():
        expected_deq, _ = quant_fl.downlink_quantize(vec, "Q_DYN_TENSOR")
        expected_up, _ = quant_fl.uplink_quantize(vec, "Q_DYN_TENSOR")
        model = p3.quantize_model(vec)
        assert model.codes.dtype == np.int8
        assert int(model.codes.min()) >= -127 and int(model.codes.max()) <= 127
        assert np.array_equal(model.dequantize(), expected_deq)      # bit-identical
        assert np.array_equal(model.dequantize(), expected_up)
        for i, sl in enumerate(quant_fl.TENSOR_SLICES):
            scale = quant_fl.dynamic_scale(vec[sl])
            assert model.scales[i] == scale
            _, codes, _ = quant_fl.quantize_dequantize(vec[sl], scale, -127, 127)
            assert np.array_equal(model.codes[sl].astype(float), codes)


def test_tensor_layout_matches_quant_fl(research):
    quant_fl = research["quant_fl"]
    assert quant_fl.N_PARAMS == p3.N_PARAMS
    assert tuple(quant_fl.SIZES) == p3.TENSOR_SIZES
    assert tuple(quant_fl.SHAPES) == p3.SHAPES
    assert quant_fl.TENSOR_SLICES == list(p3.TENSOR_SLICES)
    params = research["fl"].init(list(p3.LAYERS), 17070)
    assert np.array_equal(quant_fl.flatten(params), p3.flatten(params))
    for a, b in zip(quant_fl.unflatten(quant_fl.flatten(params)), p3.unflatten(p3.flatten(params))):
        assert np.array_equal(a, b)


def test_codec_matches_the_twin_encode_and_decode(research):
    twin = research["twin"]
    for vec in _vectors()[:20]:
        params = p3.unflatten(vec)
        codes_list, scales_list = twin.encode_params(params)
        model = p3.quantize_model(params)
        assert np.array_equal(np.concatenate([c.ravel() for c in codes_list]), model.codes)
        assert tuple(scales_list) == model.scales
        deq_twin = twin.decode_params(codes_list, scales_list, dtype=np.float64)
        assert all(np.array_equal(a, b) for a, b in zip(deq_twin, model.dequantize_tuple()))


def test_wire_scale_is_float32_and_dequantisation_uses_it():
    """Only the transmitted scale differs from quant_fl's float64 scale, by one float32 rounding."""
    vec = random_model(11, 0.3)
    model = p3.quantize_model(vec)
    decoded = p3.decode_global(p3.encode_global(0, 0, model)).model
    assert decoded.scales == model.wire_scales()
    for i, sl in enumerate(p3.TENSOR_SLICES):
        assert decoded.scales[i] == float(np.float32(model.scales[i]))
        assert abs(decoded.scales[i] - model.scales[i]) <= model.scales[i] * 2.0 ** -24
        assert np.array_equal(decoded.dequantize()[sl],
                              decoded.codes[sl].astype(np.float64) * decoded.scales[i])


def test_round_half_away_from_zero():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999, -0.49999, 126.5, -126.5])
    assert p3.round_half_away_from_zero(x).tolist() == [1, 2, 3, -1, -2, -3, 0, -0, 127, -127]


def test_scale_floor_and_zero_tensor():
    model = p3.quantize_model(np.zeros(p3.N_PARAMS))
    assert model.scales == (1e-7,) * 4
    assert not model.codes.any()
    assert not model.dequantize().any()


def test_quantise_dequantise_error_is_at_most_half_a_step():
    vec = random_model(12, 1.0)
    model = p3.quantize_model(vec)
    for i, sl in enumerate(p3.TENSOR_SLICES):
        assert np.max(np.abs(model.dequantize()[sl] - vec[sl])) <= model.scales[i] / 2 + 1e-12
