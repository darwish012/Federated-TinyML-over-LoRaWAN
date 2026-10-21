"""Client dropout / packet-loss robustness mechanics for dropout.json.

Reuses fl.py's init, local_train, and the FedAvg/FedAdam constants
(LR/ETA/B1/B2/TAU) exactly. Does not edit fl.py, run_study.py,
run_amendment2.py, or any test file.

failure_semantics (dropout.json):
  - each client keeps a local copy of the last global model it received;
    round 0 (before any training round) every client's copy is the
    server's initial model.
  - each round every client trains fl.local_train(x, its_copy) (mu=0),
    regardless of whether its uplink will be delivered.
  - server delta = unweighted mean, over clients whose uplink arrived
    THIS round, of (client_model - current_server_global); the FedAdam /
    FedAvg step is applied to that delta exactly as fl.Server does,
    using the *same expression* fl.Server.round uses so q=0 is bit-exact.
  - if no uplink arrives this round the server is left unchanged and the
    FedAdam step is skipped entirely (its step counter t does not
    advance, m/v are untouched).
  - after the server update, a client whose downlink was delivered this
    round sets its local copy to the new server global; otherwise it
    keeps its stale copy.
  - final scoring uses each client's own final local copy, not the
    server global.
"""
from __future__ import annotations

import numpy as np

import fl


def train_with_loss(clients, layers, seed, algorithm, rounds, uplink_mask, downlink_mask):
    """Run `rounds` rounds of `algorithm` ("fedavg" | "fedadam") under the
    dropout failure semantics above.

    clients: dict {machine_id: x_rows}, iterated in dict order; that
        order is column order for uplink_mask / downlink_mask.
    uplink_mask, downlink_mask: bool arrays [rounds, n_clients], True =
        delivered.

    Returns (server_global_params, {machine_id: client_final_params}).
    """
    if algorithm not in ("fedavg", "fedadam"):
        raise ValueError(algorithm)
    client_ids = list(clients.keys())
    n = len(client_ids)
    uplink_mask = np.asarray(uplink_mask, dtype=bool)
    downlink_mask = np.asarray(downlink_mask, dtype=bool)
    if uplink_mask.shape != (rounds, n):
        raise ValueError(f"uplink_mask shape {uplink_mask.shape} != {(rounds, n)}")
    if downlink_mask.shape != (rounds, n):
        raise ValueError(f"downlink_mask shape {downlink_mask.shape} != {(rounds, n)}")

    init_params = fl.init(layers, seed)
    server_params = init_params
    m = tuple(np.zeros_like(v) for v in init_params)
    v = tuple(np.zeros_like(vv) for vv in init_params)
    t = 0
    # round 0: every client starts with the server's initial model.
    client_copies = {cid: init_params for cid in client_ids}

    for r in range(rounds):
        old = server_params
        local = []
        for i, cid in enumerate(client_ids):
            p = fl.local_train(clients[cid], client_copies[cid], mu=0.0)
            if uplink_mask[r, i]:
                local.append(p)

        if local:
            # Exactly fl.Server.round's expression, so q=0 (all-True masks)
            # is bit-identical to fl.Server.
            d = tuple(np.mean(np.stack([q[k] - old[k] for q in local]), 0) for k in range(len(old)))
            t += 1
            if algorithm == "fedadam":
                m = tuple(fl.B1 * m[k] + (1 - fl.B1) * d[k] for k in range(len(old)))
                v = tuple(fl.B2 * v[k] + (1 - fl.B2) * d[k] * d[k] for k in range(len(old)))
                server_params = tuple(
                    old[k] + fl.ETA * (m[k] / (1 - fl.B1 ** t)) / (np.sqrt(v[k] / (1 - fl.B2 ** t)) + fl.TAU)
                    for k in range(len(old))
                )
            else:
                server_params = tuple(old[k] + d[k] for k in range(len(old)))
        # else: no uplink this round -> server unchanged, optimiser step
        # skipped entirely, t/m/v untouched.

        for i, cid in enumerate(client_ids):
            if downlink_mask[r, i]:
                client_copies[cid] = server_params
            # else: client keeps its stale copy.

    return server_params, dict(client_copies)


# ---------------------------------------------------------------------------
# Dropout-mask construction. dropout_rng (dropout.json): for repetition `rep`
# of scenario `scenario_index` under training seed `seed`,
#     rep_rng = np.random.default_rng(1000*seed + 100*scenario_index + rep)
# S1/S2 each have 4 loss levels (q in 0.1,0.2,0.3,0.5) and S3 has 3 (f in
# 0.1,0.2,0.3) drawn from the SAME rep_rng. To keep the levels' draws
# independent of one another (so, e.g., adding/removing a level, or the
# order levels are processed in, never perturbs another level's masks),
# rep_rng.spawn(n_levels) is used once to create one independent child
# Generator per level (NumPy SeedSequence spawning: statistically
# independent, deterministic streams). Each level then draws only from its
# own child generator, in the fixed level order below.
# ---------------------------------------------------------------------------
S1_LEVELS = (0.1, 0.2, 0.3, 0.5)
S2_LEVELS = (0.1, 0.2, 0.3, 0.5)
S3_LEVELS = (0.1, 0.2, 0.3)


def rep_rng(seed, scenario_index, rep):
    return np.random.default_rng(1000 * seed + 100 * scenario_index + rep)


def level_rngs(rng, n_levels):
    """One independent child Generator per level, level order fixed by caller."""
    return rng.spawn(n_levels)


def make_s1_masks(level_rng, rounds, n_clients, q):
    """Uplink independently lost w.p. q each (round, client); downlink always delivered."""
    draws = level_rng.random((rounds, n_clients))
    uplink = draws >= q
    downlink = np.ones((rounds, n_clients), dtype=bool)
    return uplink, downlink


def make_s2_masks(level_rng, rounds, n_clients, q):
    """Uplink and downlink each independently lost w.p. q; uplink drawn
    first, then downlink, from the same per-level child generator, so the
    two directions are independent draws but still reproducible."""
    up_draws = level_rng.random((rounds, n_clients))
    down_draws = level_rng.random((rounds, n_clients))
    uplink = up_draws >= q
    downlink = down_draws >= q
    return uplink, downlink


def select_never_participating(devices, f, rng):
    """Choose a fixed fraction f of devices that NEVER deliver an uplink,
    stratified across datasets proportionally: per dataset, floor(f *
    dataset_size) devices are chosen at random from that dataset; any
    remaining slots needed to reach the overall target round(f * n) are
    then filled at random from the devices not already chosen (pooled
    across all datasets). Returns a set of machine_ids.

    devices: list of dicts with "machine_id" and "dataset" keys, in a
        fixed (e.g. sorted-by-machine_id) order.
    """
    n = len(devices)
    target_total = int(np.floor(f * n + 0.5))  # round-half-up

    by_dataset = {}
    for d in devices:
        by_dataset.setdefault(d["dataset"], []).append(d["machine_id"])

    chosen = set()
    for ds, ids in by_dataset.items():
        ids_sorted = sorted(ids)
        k = int(np.floor(f * len(ids_sorted)))
        if k > 0:
            pick = rng.choice(len(ids_sorted), size=k, replace=False)
            chosen.update(ids_sorted[i] for i in pick)

    remaining = target_total - len(chosen)
    if remaining > 0:
        pool = sorted(set(d["machine_id"] for d in devices) - chosen)
        pick = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
        chosen.update(pool[i] for i in pick)
    elif remaining < 0:
        # Defensive: floor-per-dataset should never exceed round(f*n), but
        # if it does (pathological dataset-size distribution), drop the
        # excess at random to hit the target exactly.
        drop = rng.choice(len(chosen), size=-remaining, replace=False)
        chosen_list = sorted(chosen)
        chosen = set(chosen_list[i] for i in range(len(chosen_list)) if i not in set(drop))

    return chosen


def make_s3_masks(rounds, client_ids, never_participating):
    """Uplink is False every round for never-participating clients, True
    every round for everyone else; downlink is always delivered."""
    n = len(client_ids)
    uplink = np.ones((rounds, n), dtype=bool)
    for i, cid in enumerate(client_ids):
        if cid in never_participating:
            uplink[:, i] = False
    downlink = np.ones((rounds, n), dtype=bool)
    return uplink, downlink
