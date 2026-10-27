"""V43 Section 4: communication accounting (Leg 4).

Reuses, rather than reimplements, the LoRaWAN fragmentation and airtime
primitives already built and tested for V39
(``raw_online_federated_v39_comm.lora_time_on_air_seconds``,
``fragment_lengths``, ``message_airtime_seconds``, ``rounds_per_day``) -- the
standard Semtech LoRa modulation airtime equation and 51-byte
application-payload fragmentation are not re-derived here.

V43 performs no gradient compression (that is V39/V40's scope): the shared
federated triad (PCA-16 + AE + Deep-SVDD) is transmitted uncompressed
(``dense_float32``) in both directions, each round, using V39's own
``codec_body_bytes("dense_float32", parameters)`` convention (4 bytes per
parameter).

Two denominators, never mixed (SS4): this module reports PER CLIENT-ROUND and
PER COMPLETE TRAINING RUN figures that follow the PHYSICAL ENDPOINT (node
count), for both the physical-node and bearing-client scenarios. It never
reports a confidence interval or an effective-sample-size figure -- those are
``raw_online_federated_v43_stats.py``'s job, clustered on the machine-life
experiment, a different unit from the node/client counts used here.

Energy figures are explicitly an ESTIMATE (published typical transceiver
current, not a hardware measurement) -- consistent with memory.txt's standing
note that measured on-hardware latency/RAM/flash/energy remain UNADDRESSED.
"""
from __future__ import annotations

from typing import Mapping

import raw_online_federated_v39_comm as commlib

SPREADING_FACTORS: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
MAX_APPLICATION_PAYLOAD_BYTES = 51
FRAGMENT_HEADER_BYTES = 8
LORAWAN_PHY_OVERHEAD_BYTES = 13
CONSERVATIVE_GATEWAY_DUTY_CYCLE = 0.01
CLIENT_UPLINK_DUTY_CYCLE = 0.01
TTN_UPLINK_AIRTIME_SECONDS_PER_DEVICE_PER_DAY = 30.0
TTN_DOWNLINK_MESSAGES_PER_DEVICE_PER_DAY = 10
SECONDS_PER_DAY = 86400.0

SUPPLY_VOLTS = 3.3
TX_CURRENT_AMPS = 0.044
RX_CURRENT_AMPS = 0.0105

TRAINING_ROUNDS = 12  # raw_online_federated_v41_protocol.DRAFT.json federated_models.rounds, frozen


# --------------------------------------------------------------------------
# Model payload size, computed from the actual frozen bundle -- never
# hardcoded -- so a changed model would change this figure rather than
# silently going stale.
# --------------------------------------------------------------------------
def model_parameter_count(bundle: dict) -> int:
    total = 0
    total += int(bundle["pca"].center.size) + int(bundle["pca"].basis.size)
    total += sum(int(w.size) for w in bundle["ae"].weights) + int(bundle["ae"].head_weight.size) + 1
    total += int(bundle["svdd"].center.size) + sum(int(w.size) for w in bundle["svdd"].weights)
    return total


def model_body_bytes(bundle: dict) -> int:
    return commlib.codec_body_bytes("dense_float32", model_parameter_count(bundle))


# --------------------------------------------------------------------------
# One direction (uplink OR downlink) of one client-round, at one spreading
# factor.
# --------------------------------------------------------------------------
def direction_accounting(body_bytes: int, spreading_factor: int) -> dict:
    accounting = commlib.message_airtime_seconds(
        body_bytes,
        max_application_payload=MAX_APPLICATION_PAYLOAD_BYTES,
        spreading_factor=int(spreading_factor),
        fragment_header_bytes=FRAGMENT_HEADER_BYTES,
        lorawan_phy_overhead_bytes=LORAWAN_PHY_OVERHEAD_BYTES,
    )
    energy_joules = SUPPLY_VOLTS * TX_CURRENT_AMPS * accounting["airtime_seconds"]
    return {**accounting, "spreading_factor": int(spreading_factor), "energy_joules_estimated": energy_joules}


def client_round_accounting(model_bytes: int, spreading_factor: int) -> dict:
    """Uplink (client model update) + downlink (broadcast global model) for
    one client, one federated round, at one spreading factor. Both directions
    are counted (carried from V39's hard_constraints.both_uplink_and_downlink_are_counted)."""
    uplink = direction_accounting(model_bytes, spreading_factor)
    downlink = direction_accounting(model_bytes, spreading_factor)
    return {
        "spreading_factor": int(spreading_factor),
        "uplink": uplink,
        "downlink": downlink,
        "payload_bytes_uplink": model_bytes,
        "payload_bytes_downlink": model_bytes,
        "payload_bytes_round_trip": 2 * model_bytes,
        "lorawan_packets_uplink": uplink["fragments"],
        "lorawan_packets_downlink": downlink["fragments"],
        "airtime_seconds_round_trip": uplink["airtime_seconds"] + downlink["airtime_seconds"],
        "energy_joules_round_trip_estimated": uplink["energy_joules_estimated"] + downlink["energy_joules_estimated"],
    }


def duty_cycle_headroom(airtime_seconds_per_round: float, duty_cycle: float) -> dict:
    daily_budget_seconds = float(duty_cycle) * SECONDS_PER_DAY
    rounds_supportable_per_day = (
        int(daily_budget_seconds // airtime_seconds_per_round) if airtime_seconds_per_round > 0 else 0
    )
    headroom_fraction = (
        1.0 - (airtime_seconds_per_round / daily_budget_seconds) if daily_budget_seconds > 0 else float("nan")
    )
    return {
        "duty_cycle": float(duty_cycle),
        "daily_budget_seconds": daily_budget_seconds,
        "rounds_supportable_per_day": rounds_supportable_per_day,
        "single_round_headroom_fraction": headroom_fraction,
    }


def ttn_fair_use_headroom(uplink_airtime_seconds_per_round: float, downlink_packets_per_round: int) -> dict:
    rounds_supportable_uplink = (
        int(TTN_UPLINK_AIRTIME_SECONDS_PER_DEVICE_PER_DAY // uplink_airtime_seconds_per_round)
        if uplink_airtime_seconds_per_round > 0 else 0
    )
    rounds_supportable_downlink = (
        int(TTN_DOWNLINK_MESSAGES_PER_DEVICE_PER_DAY // downlink_packets_per_round)
        if downlink_packets_per_round > 0 else 0
    )
    return {
        "ttn_uplink_airtime_seconds_per_device_per_day": TTN_UPLINK_AIRTIME_SECONDS_PER_DEVICE_PER_DAY,
        "ttn_downlink_messages_per_device_per_day": TTN_DOWNLINK_MESSAGES_PER_DEVICE_PER_DAY,
        "rounds_supportable_per_day_uplink_limited": rounds_supportable_uplink,
        "rounds_supportable_per_day_downlink_limited": rounds_supportable_downlink,
        "rounds_supportable_per_day": min(rounds_supportable_uplink, rounds_supportable_downlink),
    }


def training_run_accounting(model_bytes: int, spreading_factor: int, n_nodes: int, rounds: int = TRAINING_ROUNDS) -> dict:
    per_round = client_round_accounting(model_bytes, spreading_factor)
    return {
        "spreading_factor": int(spreading_factor),
        "nodes": int(n_nodes),
        "rounds": int(rounds),
        "per_client_round": per_round,
        "aggregate_payload_bytes": per_round["payload_bytes_round_trip"] * int(n_nodes) * int(rounds),
        "aggregate_lorawan_packets": (
            (per_round["lorawan_packets_uplink"] + per_round["lorawan_packets_downlink"]) * int(n_nodes) * int(rounds)
        ),
        "aggregate_airtime_seconds": per_round["airtime_seconds_round_trip"] * int(n_nodes) * int(rounds),
        "aggregate_energy_joules_estimated": per_round["energy_joules_round_trip_estimated"] * int(n_nodes) * int(rounds),
    }


# --------------------------------------------------------------------------
# The two deployment scenarios (SS4). Node counts are the ONLY thing that
# differs between them; per-radio figures above are identical either way.
# --------------------------------------------------------------------------
def scenario_node_counts(cohort_rows: list[dict]) -> dict[str, dict[str, int]]:
    """Physical-node vs bearing-client node counts, per dataset and pooled,
    from a cohort table produced by raw_online_federated_v43_cohort.cohort_table.
    Uses the roster-restricted counts when present (streams_in_supplied_roster/
    experiments_in_supplied_roster), else the full historical counts."""
    out: dict[str, dict[str, int]] = {}
    for row in cohort_rows:
        streams = int(row.get("streams_in_supplied_roster", row["signal_streams"]))
        experiments = int(row.get("experiments_in_supplied_roster", row["machine_life_experiments"]))
        out[row["dataset"]] = {"bearing_client_nodes": streams, "physical_nodes": experiments}
    return out


def full_accounting(
    bundle: dict, cohort_rows: list[dict], spreading_factors: tuple[int, ...] = SPREADING_FACTORS,
) -> dict:
    model_bytes = model_body_bytes(bundle)
    node_counts = scenario_node_counts(cohort_rows)
    total_bearing_client = sum(v["bearing_client_nodes"] for v in node_counts.values())
    total_physical_node = sum(v["physical_nodes"] for v in node_counts.values())

    per_sf = []
    for sf in spreading_factors:
        round_acc = client_round_accounting(model_bytes, sf)
        duty = duty_cycle_headroom(round_acc["airtime_seconds_round_trip"], CONSERVATIVE_GATEWAY_DUTY_CYCLE)
        ttn = ttn_fair_use_headroom(round_acc["uplink"]["airtime_seconds"], round_acc["lorawan_packets_downlink"])
        per_sf.append({
            "spreading_factor": sf,
            "client_round": round_acc,
            "duty_cycle_1pct_headroom": duty,
            "ttn_fair_use_headroom": ttn,
            "training_run_bearing_client_scenario": training_run_accounting(model_bytes, sf, total_bearing_client),
            "training_run_physical_node_scenario": training_run_accounting(model_bytes, sf, total_physical_node),
        })

    return {
        "model_body_bytes": model_bytes,
        "training_rounds": TRAINING_ROUNDS,
        "node_counts_by_dataset": node_counts,
        "total_bearing_client_nodes": total_bearing_client,
        "total_physical_nodes": total_physical_node,
        "per_spreading_factor": per_sf,
        "denominator_note": (
            "node counts above follow the PHYSICAL ENDPOINT (bearing-client or physical-node scenario); "
            "statistical intervals and effective sample size always follow the MACHINE-LIFE EXPERIMENT "
            "(raw_online_federated_v43_stats.py) -- these are different denominators and are never mixed here."
        ),
        "energy_caveat": "energy figures are an ESTIMATE from published typical transceiver current, not a hardware measurement.",
    }


__all__ = [
    "CLIENT_UPLINK_DUTY_CYCLE",
    "CONSERVATIVE_GATEWAY_DUTY_CYCLE",
    "FRAGMENT_HEADER_BYTES",
    "LORAWAN_PHY_OVERHEAD_BYTES",
    "MAX_APPLICATION_PAYLOAD_BYTES",
    "RX_CURRENT_AMPS",
    "SPREADING_FACTORS",
    "SUPPLY_VOLTS",
    "TRAINING_ROUNDS",
    "TTN_DOWNLINK_MESSAGES_PER_DEVICE_PER_DAY",
    "TTN_UPLINK_AIRTIME_SECONDS_PER_DEVICE_PER_DAY",
    "TX_CURRENT_AMPS",
    "client_round_accounting",
    "direction_accounting",
    "duty_cycle_headroom",
    "full_accounting",
    "model_body_bytes",
    "model_parameter_count",
    "scenario_node_counts",
    "training_run_accounting",
    "ttn_fair_use_headroom",
]
