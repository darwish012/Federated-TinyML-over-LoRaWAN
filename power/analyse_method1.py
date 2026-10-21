"""Method 1 analysis: energy per phase / round / event from a capture_method1.py run.

    python analyse_method1.py runs/m1_YYYYMMDD_HHMM [--vref 1.101] [--shunt 1.0] [--json out.json]

Inputs (raw.log only, the Uno's own clock for everything):
    C <ms> <adc>      current: I = adc * vref / 1023 / shunt
    V <ms> <adc>      Uno supply: Vcc = vref * 1023 / adc (once a minute, interpolated between)
    M <ms> PHASE ...  board phase markers, M <ms> ROUND ... round markers
Board input voltage per sample: V = Vcc - I * shunt (the shunt is in the ground return).
Energy = sum(V * I * dt), dt = time to the next sample (trapezoid-free left rule at 200 Hz;
the rule changes totals by < 0.1 %).

Marker corrections (firmware before the 2026-10-14 fix):
  * After a live acquisition ("score") the firmware restored the PREVIOUS marker. When that was
    "tx" (an event marker), the listening time that followed was labelled "tx": every marker
    that directly follows "score" is relabelled "rx_classc".
  * "tx" only marks the start of an uplink; the on-air time falls inside the following "idle"
    (TX + RX1/RX2 windows). On-air samples are found in the current trace: within 1.5 s after a
    "tx" marker, samples above the listening median + --tx-threshold-ma are "tx_air".
Reported groups: listen (rx_classc + idle + boot/join waits), tx_air, sense, features+score,
train (+ flash read of the prepared rows), other (prepare/replay/...).
"""
import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path

GROUND_COUNTS = []   # A1 (shunt leg B) readings: the ground-path drop the differential reading removes

GROUP = {"rx_classc": "listen", "idle": "listen", "boot": "listen", "tx": "listen",
         "tx_air": "tx_air", "sense": "sense", "features": "features+score", "score": "features+score",
         "train": "train", "sd_read": "train"}


def load(raw):
    cs, vs, ms_ = [], [], []
    for line in open(raw, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) < 3:
            continue
        try:
            if p[0] == "C":
                # "C ms a0 a1" = differential (Kelvin) reading, current = a0 - a1; "C ms a0" = older single-ended
                a1 = int(p[3]) if len(p) > 3 else 0
                cs.append((int(p[1]), int(p[2]) - a1))
                if len(p) > 3:
                    GROUND_COUNTS.append(a1)
            elif p[0] == "V":
                vs.append((int(p[1]), float(p[2])))
            elif p[0] == "M" and p[2] in ("PHASE", "ROUND") and len(p) >= 4:
                ms_.append((int(p[1]), p[2], p[3]))
        except ValueError:
            continue
    cs.sort(); ms_.sort(key=lambda m: m[0])
    return cs, vs, ms_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--vref", type=float, default=1.101)
    ap.add_argument("--shunt", type=float, default=1.0)
    ap.add_argument("--tx-threshold-ma", type=float, default=25.0)
    ap.add_argument("--battery-mah", type=float, default=700.0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cs, vs, markers = load(a.run / "raw.log")
    if not cs or not markers:
        raise SystemExit("no samples or no markers")
    k = a.vref / 1023.0 / a.shunt                      # A per count
    vcc_t = [t for t, _ in vs]
    vcc_v = [a.vref * 1023.0 / adc for _, adc in vs]

    def vcc(t):
        if not vs:
            return 5.0
        i = bisect.bisect_left(vcc_t, t)
        if i <= 0: return vcc_v[0]
        if i >= len(vs): return vcc_v[-1]
        t0, t1 = vcc_t[i - 1], vcc_t[i]
        return vcc_v[i - 1] + (vcc_v[i] - vcc_v[i - 1]) * (t - t0) / (t1 - t0)

    # Phase timeline with the restore fix; rounds from ROUND markers.
    phases, rounds, prev = [], [], None
    for t, kind, name in markers:
        if kind == "ROUND":
            rounds.append((t, name)); continue
        if prev == "score":
            name = "rx_classc"
        phases.append((t, name)); prev = name
    p_t = [t for t, _ in phases]
    r_t = [t for t, _ in rounds]
    tx_starts = [t for t, n in phases if n == "tx"]

    def phase_at(t):
        i = bisect.bisect_right(p_t, t) - 1
        return phases[i][1] if i >= 0 else "unmarked"

    def round_at(t):
        i = bisect.bisect_right(r_t, t) - 1
        return rounds[i][1] if i >= 0 else "before_first_round"

    listen_ma = statistics.median(adc * k * 1000 for t, adc in cs if phase_at(t) == "rx_classc") \
        if any(phase_at(t) == "rx_classc" for t, _ in cs[::50]) else statistics.median(adc * k * 1000 for _, adc in cs)
    thr = listen_ma + a.tx_threshold_ma

    E = defaultdict(float); T = defaultdict(float); Q = defaultdict(float)
    ER = defaultdict(float); TR = defaultdict(float)
    n_air = 0
    for i, (t, adc) in enumerate(cs[:-1]):
        dt = (cs[i + 1][0] - t) / 1000.0
        if dt <= 0 or dt > 0.5:            # gap (Uno supply check) or disorder: skip, not interpolate
            continue
        I = adc * k
        ph = phase_at(t)
        j = bisect.bisect_right(tx_starts, t) - 1
        if j >= 0 and t - tx_starts[j] <= 1500 and ph in ("idle", "tx") and I * 1000 > thr:
            ph = "tx_air"; n_air += 1
        g = GROUP.get(ph, "other:" + ph)
        e = (vcc(t) - I * a.shunt) * I * dt
        E[g] += e; T[g] += dt; Q[g] += I * dt
        r = round_at(t); ER[r] += e; TR[r] += dt
    total_e, total_t, total_q = sum(E.values()), sum(T.values()), sum(Q.values())
    i_avg = total_q / total_t

    # Events: count of acquisitions, trainings, uplinks (from markers).
    n_sense = sum(1 for _, n in phases if n == "sense")
    n_train = sum(1 for _, n in phases if n == "train")
    n_tx = len(tx_starts)
    in_rounds = [r for r in TR if r.startswith("run")]
    print(f"run {a.run.name}: {total_t / 60:.1f} min, {len(cs)} samples, {len(phases)} phase markers, "
          f"{len(in_rounds)} rounds")
    if GROUND_COUNTS:
        g = sorted(GROUND_COUNTS)
        print(f"differential reading: shunt leg B (A1) median {g[len(g) // 2] * k * 1000:.1f} mV-equivalent "
              f"(ground-path drop removed from every sample)")
    print(f"listening median {listen_ma:.1f} mA; TX threshold {thr:.1f} mA ({n_air} on-air samples = {n_air * 5 / 1000:.1f} s)")
    print(f"average current {i_avg * 1000:.2f} mA; average power {total_e / total_t * 1000:.1f} mW at the 5 V input; "
          f"Vcc {min(vcc_v):.3f}-{max(vcc_v):.3f} V")
    print(f"{'group':18s} {'time s':>9s} {'share t':>8s} {'mean mA':>8s} {'energy J':>9s} {'share E':>8s}")
    for g in sorted(E, key=lambda g: -E[g]):
        print(f"{g:18s} {T[g]:9.1f} {100 * T[g] / total_t:7.2f}% {1000 * Q[g] / T[g]:8.1f} {E[g]:9.2f} {100 * E[g] / total_e:7.2f}%")
    per = {}
    if n_sense: per["acquisition (sense + features + score) J"] = (E["sense"] + E["features+score"]) / n_sense
    if n_train: per["training of one machine (flash read + train) J"] = E["train"] / n_train
    if n_tx: per["uplink on air J (all TX samples / uplinks)"] = E["tx_air"] / n_tx
    for name, v in per.items():
        print(f"  per {name}: {v:.3f}")
    rr = sorted(r for r in in_rounds)
    if rr:
        er = [ER[r] for r in rr]; tr = [TR[r] for r in rr]
        print(f"  per round: mean {statistics.fmean(er):.1f} J over {statistics.fmean(tr):.0f} s "
              f"(rounds {min(tr):.0f}-{max(tr):.0f} s)")
    life = a.battery_mah * 0.85 / (i_avg * 1000)
    print(f"  {a.battery_mah:.0f} mAh LiPo (85% usable) at this average current: {life:.1f} h "
          f"(same current through the linear regulator; the 5 V-input extras are not removed)")
    if a.json:
        out = {"run": str(a.run), "vref": a.vref, "shunt": a.shunt, "listen_median_mA": listen_ma, "tx_threshold_mA": thr,
               "duration_s": total_t, "average_current_mA": i_avg * 1000, "average_power_mW_5V": total_e / total_t * 1000,
               "groups": {g: {"time_s": T[g], "mean_mA": 1000 * Q[g] / T[g], "energy_J": E[g], "share_energy": E[g] / total_e}
                          for g in E},
               "events": {"acquisitions": n_sense, "trainings": n_train, "uplinks": n_tx}, "per_event_J": per,
               "rounds": {r: {"energy_J": ER[r], "time_s": TR[r]} for r in ER}, "battery_life_h": life}
        a.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
