"""Merge several method-1 reports (one run captured in parts, e.g. before/after a resume).

    python merge_method1.py PART1/method1_report.json PART2/method1_report.json [--json merged.json]

Groups are summed (time, charge, energy) and re-normalised; per-event costs are recomputed from
the summed group energies and event counts; rounds are listed with their part.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--battery-mah", type=float, default=700.0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    T = defaultdict(float); E = defaultdict(float); Q = defaultdict(float); ev = defaultdict(int)
    rounds, dur, energy = {}, 0.0, 0.0
    for i, path in enumerate(a.reports, 1):
        r = json.loads(path.read_text(encoding="utf-8"))
        for g, v in r["groups"].items():
            if g.startswith("other:unmarked"):      # time before the board was powered/marked
                continue
            T[g] += v["time_s"]; E[g] += v["energy_J"]; Q[g] += v["mean_mA"] * v["time_s"] / 1000.0
        for k, n in r["events"].items():
            ev[k] += n
        for name, v in r["rounds"].items():
            if name.startswith("run"):
                rounds[f"part{i}:{name}"] = v
    tt, te, tq = sum(T.values()), sum(E.values()), sum(Q.values())
    i_avg = tq / tt
    print(f"merged {len(a.reports)} parts: {tt / 60:.1f} min, {len(rounds)} round segments, "
          f"average {i_avg * 1000:.2f} mA, {te / tt * 1000:.1f} mW at the 5 V input")
    for g in sorted(E, key=lambda g: -E[g]):
        print(f"  {g:16s} time {100 * T[g] / tt:6.2f}%  mean {1000 * Q[g] / T[g]:6.1f} mA  energy {E[g]:8.2f} J  {100 * E[g] / te:6.2f}%")
    per = {}
    if ev["acquisitions"]: per["acquisition_J"] = (E["sense"] + E["features+score"]) / ev["acquisitions"]
    if ev["trainings"]: per["training_J"] = E["train"] / ev["trainings"]
    if ev["uplinks"]: per["uplink_on_air_J"] = E["tx_air"] / ev["uplinks"]
    full = [v for k, v in rounds.items()]
    per_round = sum(v["energy_J"] for v in full) / len(full) if full else None
    life = a.battery_mah * 0.85 / (i_avg * 1000)
    print(f"  events {dict(ev)}; per event {({k: round(v, 3) for k, v in per.items()})}; "
          f"per round segment {per_round:.1f} J; 700 mAh LiPo {life:.1f} h")
    if a.json:
        a.json.write_text(json.dumps({"parts": [str(p) for p in a.reports], "duration_s": tt,
                                      "average_current_mA": i_avg * 1000, "average_power_mW_5V": te / tt * 1000,
                                      "groups": {g: {"time_s": T[g], "mean_mA": 1000 * Q[g] / T[g], "energy_J": E[g],
                                                     "share_energy": E[g] / te} for g in E},
                                      "events": ev, "per_event_J": per, "per_round_J": per_round,
                                      "rounds": rounds, "battery_life_h": life}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
