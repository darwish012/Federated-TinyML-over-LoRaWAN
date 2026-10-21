"""Reference computations for the v4 tests: what the twin would compute from the same deltas.

The controller is checked against twin_server.Server driven by THIS code, which only
reads (a) the deltas the fake boards really sent (sim.sent) and (b) which positions the
server's delivery log says arrived and from which base slot. The aggregation arithmetic,
the stale-base reconstruction and the quantised downlink are re-implemented here
independently of controller_v4.
"""
from __future__ import annotations

import numpy as np

import protocol_v4 as p4
from study_plan import load_fl_modules


def on_the_wire(m: p4.QuantisedModel) -> p4.QuantisedModel:
    """What a receiver decodes: the same codes with float32 scales."""
    return p4.QuantisedModel(codes=m.codes, scales=m.wire_scales())


def f32_quantised(params) -> p4.QuantisedModel:
    return p4.quantize_model(tuple(np.asarray(p, dtype=np.float32) for p in params))


def twin_reference(sim, phase_index: int, rounds: int | None = None,
                   treat_stale_as_fresh: bool = False, seed: int | None = None):
    """Recompute every GLOBAL of a phase with twin_server.Server (the research module).

    Round r's participants and their base slots come from the delivery log. Returns
    (server, globals_by_round) where globals_by_round[r] is the quantised GLOBAL of round r.
    """
    twin_server, fl = load_fl_modules()
    records = sim.round_records(phase_index)
    rounds = len(records) if rounds is None else rounds
    seed = sim.plan.seed if seed is None else seed
    server = twin_server.Server(fl.init(list(p4.LAYERS), seed))
    globals_ = [f32_quantised(server.p)]
    first_slot = min(g["slot"] for g in sim.out.global_models if g["phase_index"] == phase_index)
    for r in range(1, rounds + 1):
        rec = records[r - 1]
        assert rec["round"] == r
        seed_vec = on_the_wire(globals_[r - 1]).dequantize()
        pairs = []
        for position in sorted(rec["arrived_positions"]):
            delta = on_the_wire(sim.sent[(phase_index, r, position)]).dequantize()
            base_round = rec["base_slot_by_position"][str(position)] - first_slot
            if base_round == r - 1 or treat_stale_as_fresh:
                pairs.append((p4.unflatten(delta), 1.0))
            else:
                client = on_the_wire(globals_[base_round]).dequantize() + delta
                pairs.append((p4.unflatten(client - seed_vec), 1.0))
        server.round(pairs)
        globals_.append(f32_quantised(server.p))
    return server, globals_


def assert_matches_twin(sim, phase_index: int, **kwargs) -> None:
    server, expected = twin_reference(sim, phase_index, **kwargs)
    logged = sim.logged_global_codes(phase_index)
    assert len(logged) == len(expected), (len(logged), len(expected))
    for r, g in enumerate(expected):
        assert np.array_equal(logged[r], g.codes), f"phase {phase_index} GLOBAL round {r}"
    return server
