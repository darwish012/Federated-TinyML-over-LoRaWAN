"""Small, deterministic healthy-only model suite (NumPy only).

This module deliberately accepts training matrices only.  It has no dataset
loader and therefore cannot accidentally read the sealed evaluation cohort.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
VARIANTS = ("global_ae", "local_ae", "personalized_ae", "ae_svdd", "synthetic_ae", "personalized_svdd")
ROUNDS, EPOCHS, LR, L2 = 12, 2, 0.03, 0.0001
SERVER_LR, BETA1, BETA2, TAU = 0.01, 0.9, 0.99, 0.001


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _init(seed: int = 17070) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Exactly thesis_participating_node_scaling.initial_model(protocol).
    rng = np.random.default_rng(int(seed))
    s = np.sqrt(2.0 / (22 + 4))
    return rng.normal(0.0, s, (22, 4)), np.zeros(4), rng.normal(0.0, s, (4, 22)), np.zeros(22)


def _copy(a):
    return tuple(np.array(x, dtype=np.float64, copy=True) for x in a)


def _forward(x, a):
    w1, b1, w2, b2 = a
    h = np.tanh(x @ w1 + b1)
    return h @ w2 + b2, h


def _ae_loss_grads(x, a, center=None, *, center_weight=0.0):
    """Full-batch reconstruction (+ optional fixed-centre) objective."""
    w1, b1, w2, b2 = a
    y, h = _forward(x, a)
    n = max(x.shape[0] * x.shape[1], 1)
    dy = 2.0 * (y - x) / n
    dh = dy @ w2.T
    extra = 0.0
    if center is not None and center_weight:
        d = h - center
        extra = center_weight * float(np.mean(d * d))
        dh = dh + (2.0 * center_weight / max(h.size, 1)) * d
    dz = dh * (1.0 - h * h)
    g = (x.T @ dz + L2 * w1, np.sum(dz, axis=0), h.T @ dy + L2 * w2, np.sum(dy, axis=0))
    loss = float(np.mean((y - x) ** 2) + extra + 0.5 * L2 * (np.sum(w1*w1) + np.sum(w2*w2)))
    return loss, tuple(np.asarray(v, dtype=np.float64) for v in g), h, y


def _synthetic(x, seed):
    rng = np.random.default_rng(int(seed))
    out = np.array(x, copy=True)
    for i in range(len(out)):
        k = int(rng.integers(1, 5)); idx = rng.choice(22, size=k, replace=False)
        out[i, idx] += rng.uniform(2.0, 4.0, size=k)
        k2 = int(rng.integers(1, 5)); idx2 = rng.choice(22, size=k2, replace=False)
        out[i, idx2] *= rng.uniform(1.5, 3.0, size=k2)
    return out


def _head_grads(x, a, hw, hb, seed, xs=None):
    _, h = _forward(x, a)
    xs = _synthetic(x, seed) if xs is None else xs
    _, hs = _forward(xs, a)
    z = np.concatenate((h, hs)); y = np.concatenate((np.zeros(len(x)), np.ones(len(x))))
    feat = z  # synthetic head is a balanced latent-feature surrogate
    p = 1.0 / (1.0 + np.exp(-np.clip(feat @ hw + hb, -40, 40)))
    e = p - y
    # Derivative through latent features as well as through the head itself.
    gh = e[:, None] * hw[None, :] / len(y)
    return float(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))), feat.T @ e / len(y), float(np.mean(e)), gh


def _step(a, x, center=None, *, svdd=False, hw=None, hb=None, synthetic_seed=0, synthetic_x=None):
    loss, g, h, _ = _ae_loss_grads(x, a, center, center_weight=0.1 if svdd else 0.0)
    if hw is not None:
        xs = _synthetic(x, synthetic_seed) if synthetic_x is None else synthetic_x
        hl, ghw, ghb, gh = _head_grads(x, a, hw, hb, synthetic_seed, xs)
        # gh is d(BCE)/d latent for actual and synthetic rows.  The synthetic
        # rows use the same network, so both halves contribute to its encoder.
        zgrad = gh
        w1, b1, w2, b2 = a
        g = list(g)
        _, hs = _forward(xs, a)
        dpre = zgrad * (1.0 - np.concatenate((h, hs))**2)
        g[0] = g[0] + 0.1 * (np.concatenate((x, xs)).T @ dpre)
        g[1] = g[1] + 0.1 * np.sum(dpre, axis=0)
        g = tuple(g)
        hw = hw - LR * (0.1 * ghw); hb = hb - LR * (0.1 * ghb); loss += 0.1 * hl
    a = tuple(v - LR * gv for v, gv in zip(a, g))
    return a, hw, hb, loss


def _fedadam(old, locals_, m, v, t):
    delta = tuple(np.mean(np.stack([q[i] - old[i] for q in locals_]), axis=0) for i in range(len(old)))
    nm = tuple(BETA1*m[i] + (1-BETA1)*delta[i] for i in range(len(old)))
    nv = tuple(BETA2*v[i] + (1-BETA2)*delta[i]*delta[i] for i in range(len(old)))
    return tuple(old[i] + SERVER_LR*(nm[i]/(1-BETA1**t))/(np.sqrt(nv[i]/(1-BETA2**t))+TAU) for i in range(len(old))), nm, nv


def _center(x, a):
    _, h = _forward(x, a)
    c = np.mean(h, axis=0)
    return np.where(np.abs(c) < 0.1, np.where(c < 0, -0.1, 0.1), c)


def fit(clients: dict[str, np.ndarray], variant: str, seed: int = 17070) -> dict[str, Any]:
    if variant not in VARIANTS: raise ValueError(f"unknown variant: {variant}")
    if not clients: raise ValueError("clients must be non-empty")
    names = tuple(clients)
    xs = {k: np.asarray(v, dtype=np.float64) for k, v in clients.items()}
    if any(v.ndim != 2 or v.shape[1] != 22 or not len(v) or not np.all(np.isfinite(v)) for v in xs.values()): raise ValueError("each client must be nonempty finite rows x 22")
    base = _init(seed); centers = {k: _center(x, base) for k, x in xs.items()} if variant in ("ae_svdd", "personalized_svdd") else {}
    head = variant == "synthetic_ae"; hw = np.zeros(4); hb = 0.0
    synthetic = {k: _synthetic(x, int.from_bytes(hashlib.sha256(f"{seed}|{k}".encode()).digest()[:8], "little")) for k, x in xs.items()} if head else {}
    m = tuple(np.zeros_like(q) for q in base); v = tuple(np.zeros_like(q) for q in base)
    hm = np.zeros(4); hv = np.zeros(4); hmb = 0.0; hvb = 0.0
    private = {k: _copy(base) for k in names}
    if variant == "local_ae":
        # Each client has its own one-client FedAdam server; no cross-client
        # averaging occurs.  ``global`` is retained only for API parity.
        for k in names:
            q, mm, vv = _copy(base), tuple(np.zeros_like(z) for z in base), tuple(np.zeros_like(z) for z in base)
            for t in range(1, ROUNDS + 1):
                la = q
                sk = int.from_bytes(hashlib.sha256(f"{seed}|{k}".encode()).digest()[:8], "little")
                for e in range(EPOCHS): la, _, _, _ = _step(la, xs[k], centers.get(k), svdd=False, synthetic_seed=sk)
                q, mm, vv = _fedadam(q, [la], mm, vv, t)
            private[k] = q
        return {"variant": variant, "seed": int(seed), "clients": names, "global": base, "private": private, "centers": centers, "head_weight": hw, "head_bias": hb, "protocol": _protocol()}
    for t in range(1, ROUNDS + 1):
        local = []; local_hw = []; local_hb = []
        for j, k in enumerate(names):
            if variant.startswith("personalized"):
                # Shared encoder comes from the current server; decoder stays
                # private from the previous round.
                start = (base[0], base[1], private[k][2], private[k][3])
            else: start = base
            la, lhw, lhb = start, hw, hb
            for e in range(EPOCHS):
                sk = int.from_bytes(hashlib.sha256(f"{seed}|{k}".encode()).digest()[:8], "little")
                la, lhw, lhb, _ = _step(la, xs[k], centers.get(k), svdd=variant.endswith("svdd"), hw=lhw if head else None, hb=lhb, synthetic_seed=sk, synthetic_x=synthetic.get(k))
            local.append(la); local_hw.append(lhw); local_hb.append(lhb)
            if variant.startswith("personalized"): private[k] = la
        if variant.startswith("personalized"):
            # Encoder-only server FedAdam state; decoder arrays never enter it.
            if t == 1: em = tuple(np.zeros_like(base[i]) for i in (0,1)); ev = tuple(np.zeros_like(z) for z in em)
            en = tuple(np.mean(np.stack([q[i] for q in local]), axis=0) for i in (0,1)); d = tuple(en[i]-base[i] for i in (0,1)); em=tuple(BETA1*em[i]+(1-BETA1)*d[i] for i in (0,1)); ev=tuple(BETA2*ev[i]+(1-BETA2)*d[i]*d[i] for i in (0,1)); eupd=tuple(base[i]+SERVER_LR*(em[i]/(1-BETA1**t))/(np.sqrt(ev[i]/(1-BETA2**t))+TAU) for i in (0,1)); base=(eupd[0],eupd[1],base[2],base[3])
        else:
            if head:
                combo, m, v = _fedadam(base + (hw, hb), [q + (local_hw[i], local_hb[i]) for i, q in enumerate(local)], m + (hm, hmb), v + (hv, hvb), t)
                base, hw, hb = combo[:4], combo[4], float(combo[5]); m, hm, hmb = m[:4], m[4], float(m[5]); v, hv, hvb = v[:4], v[4], float(v[5])
            else: base, m, v = _fedadam(base, local, m, v, t)
    if variant.startswith("personalized"):
        private = {k: (base[0], base[1], q[2], q[3]) for k, q in private.items()}
    return {"variant": variant, "seed": int(seed), "clients": names, "global": base, "private": private, "centers": centers, "head_weight": hw, "head_bias": hb, "protocol": _protocol()}


def _model_for(state, machine_id):
    return state["private"].get(machine_id, state["global"]) if state["variant"] in ("local_ae", "personalized_ae", "personalized_svdd") else state["global"]


def reconstruction_scores(state, machine_id, x):
    a = _model_for(state, machine_id); y, _ = _forward(np.asarray(x, dtype=np.float64), a); return np.mean((y-np.asarray(x))**2, axis=1)


def score(state, machine_id, x):
    x = np.asarray(x, dtype=np.float64); a = _model_for(state, machine_id); y, h = _forward(x, a)
    if state["variant"] == "synthetic_ae":
        return 1/(1+np.exp(-np.clip(h@state["head_weight"]+state["head_bias"], -40, 40)))
    s = np.mean((y-x)**2, axis=1)
    if state["variant"].endswith("svdd"): s = s + 0.1*np.mean((h-state["centers"][machine_id])**2, axis=1)
    return s


def global_arrays(state): return tuple(np.asarray(x) for x in state["global"])


def communication(state):
    n = 207 if state["variant"] == "synthetic_ae" else (92 if state["variant"].startswith("personalized") else 202)
    local = state['variant'] == 'local_ae'
    fragments = 0 if local else int(np.ceil((4*n+16)/200))
    return {"float32_parameters": n, "parameter_bytes": 4*n, "header_bytes": 16, "fragment_body_bytes": 200,
            "fragments_per_update": fragments, "fragments_per_global": fragments, "rounds": 12,
            "local_only_no_network": local, "private_decoder_parameters": 110 if state["variant"].startswith("personalized") else 0,
            "private_center_float32_bytes": 16 if state['variant'].endswith('svdd') else 0,
            "claim": "analytical model-only payload; no RF, codec, or compute validation"}
