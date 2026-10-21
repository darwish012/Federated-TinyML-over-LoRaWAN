"""Phase durations measured on real boards, from the firmware's serial markers.

    python phase_durations.py LOG [LOG ...] [--from HH:MM:SS] [--to HH:MM:SS] [--json out.json]

Each LOG is a passive (no-reset) serial capture whose lines are "HH:MM:SS <board line>",
e.g. server/night_COM76.log from the 2026-10-04/05 sweep. Board lines used:
    PHASE <name> <millis>      the board enters phase <name> at uptime <millis>
    ROUND <id> <millis>        the board accepted a new GLOBAL (one federated round)
Each PHASE marker closes the previous phase (half-open intervals, board clock), as in
"final lab/phase_energy.py". The window [--from, --to] is in the capture's PC time of day
(local), applied to the line prefix; it is used to select e.g. one sweep phase.

Output per log: total seconds per phase, number of entries, mean/median duration of one
entry, number of ROUND lines, and TX count. The firmware's "tx" marker only marks the
start of an uplink (the "idle" marker follows within a few ms while the radio transmits
and waits for RX1/RX2), so on-air time is not taken from the markers; the datasheet model
computes it from the LoRa airtime formula and the TX count.
"""
import argparse
import json
import statistics
from collections import defaultdict


def parse(path, t_from=None, t_to=None):
    events = []          # (millis, kind, name)
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.split()
        if len(parts) < 4 or parts[1] not in ("PHASE", "ROUND"):
            continue
        clock = parts[0]
        if (t_from and clock < t_from) or (t_to and clock >= t_to):
            continue
        try:
            millis = int(parts[3])
        except ValueError:
            continue
        events.append((millis, parts[1], parts[2]))
    return events


def summarise(events):
    durations = defaultdict(list)
    rounds = 0
    phase, start = None, None
    resets = 0
    for millis, kind, name in events:
        if kind == "ROUND":
            if name != "end":
                rounds += 1
            continue
        if phase is not None:
            if millis < start:          # board reset inside the window: drop the open phase
                resets += 1
            else:
                durations[phase].append((millis - start) / 1000.0)
        phase, start = name, millis
    span = (events[-1][0] - events[0][0]) / 1000.0 if events else 0.0
    out = {"span_s": span, "rounds": rounds, "resets": resets, "phases": {}}
    for name, d in sorted(durations.items()):
        out["phases"][name] = {
            "entries": len(d), "total_s": round(sum(d), 3),
            "mean_s": round(statistics.fmean(d), 3), "median_s": round(statistics.median(d), 3),
            "share": round(sum(d) / span, 5) if span else None}
    out["tx_count"] = out["phases"].get("tx", {}).get("entries", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--from", dest="t_from")
    ap.add_argument("--to", dest="t_to")
    ap.add_argument("--json")
    a = ap.parse_args()
    report = {}
    for path in a.logs:
        s = summarise(parse(path, a.t_from, a.t_to))
        report[path] = s
        print(f"== {path}: span {s['span_s'] / 3600:.2f} h, {s['rounds']} rounds, {s['tx_count']} TX, resets {s['resets']}")
        for name, p in s["phases"].items():
            print(f"   {name:10s} entries {p['entries']:5d}  total {p['total_s']:9.1f} s  "
                  f"mean {p['mean_s']:7.3f} s  median {p['median_s']:7.3f} s  share {p['share']:.4f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
