"""Method 1 capture: read the Uno DAQ (uno_daq.ino) and write phase_energy.py inputs.

    python capture_method1.py --port COM5 --out runs/m1_YYYYMMDD [--vref 1.100] [--shunt 1.0]

Stop with Ctrl-C. Writes into --out:
    raw.log          every Uno line as received (C/M lines), for re-processing
    current.csv      Timestamp,Current_mA  (Timestamp = Uno millis as HH:MM:SS.fff from 00:00)
    markers.log      PHASE/ROUND lines with the board's millis REPLACED by the Uno's millis
    other.log        BATT/SCORE/SELF_CHECK/... lines (Uno millis prefix)
    vcc.csv          uno_ms,vcc_V: the Uno 5V rail once a minute (bandgap vs AVcc, scaled by --vref)

Current and markers share the Uno clock, so phase_energy.py needs the trivial anchor:
    python "../../final lab/phase_energy.py" current.csv markers.log --voltage <V> \
           --sync-millis 0 --sync-time 00:00:00.000 --json phase_report.json
<V> is the board's supply: the Uno 5V rail measured with a multimeter while the board runs,
minus the shunt drop (about 0.1 V at 100 mA on 1 ohm).

Calibration: mA = adc / 1023 * vref / shunt * 1000. The Uno's internal reference is nominally
1.1 V but varies 1.0-1.2 V between chips: measure it (AREF pin with a multimeter while
uno_daq runs) and pass --vref. Use the measured shunt resistance for --shunt.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import serial


def clock(ms):
    s, ms = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vref", type=float, default=1.101,
                    help="Uno internal reference (V); 1.101 V measured 2026-10-14 with an ESP32 eFuse-calibrated ADC")
    ap.add_argument("--shunt", type=float, default=1.0, help="measured shunt resistance (ohm)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    port = serial.Serial(a.port, 115200, timeout=0.5)
    raw = open(a.out / "raw.log", "w", encoding="utf-8")
    cur = open(a.out / "current.csv", "w", encoding="utf-8", newline="")
    mk = open(a.out / "markers.log", "w", encoding="utf-8")
    oth = open(a.out / "other.log", "w", encoding="utf-8")
    vcc = open(a.out / "vcc.csv", "w", encoding="utf-8", newline="")
    vcc.write("uno_ms,vcc_V\n")
    cur.write("Timestamp,Current_mA\n")
    k = a.vref / 1023.0 / a.shunt * 1000.0
    n_c = n_m = 0
    last_ms = -1
    print(f"capturing {a.port} -> {a.out} (vref {a.vref} V, shunt {a.shunt} ohm). Ctrl-C to stop.")
    offset, seen_ms = 0, 0     # after a reconnect the Uno restarts at 0 ms: shift so time stays monotonic
    while not stop["now"]:
        try:
            line = port.readline().decode("ascii", errors="replace").strip()
        except serial.SerialException as exc:     # USB hiccup (cable bumped, re-enumeration): reconnect
            print(f"\n  serial error ({exc}); reconnecting", flush=True)
            raw.write(f"# serial error, reconnecting: {exc}\n")
            try:
                port.close()
            except Exception:
                pass
            while not stop["now"]:
                try:
                    time.sleep(2)
                    port = serial.Serial(a.port, 115200, timeout=0.5)
                    break
                except serial.SerialException:
                    continue
            offset = seen_ms + 1000                # the gap is unknown; the analysis skips gaps > 0.5 s
            raw.write(f"# reconnected; Uno times shifted by {offset} ms\n")
            continue
        if not line:
            continue
        parts = line.split(" ", 2)
        if offset and len(parts) >= 2 and parts[0] in ("C", "M", "V") and parts[1].isdigit():
            parts[1] = str(int(parts[1]) + offset)
            line = " ".join(parts)
        if len(parts) >= 2 and parts[1].isdigit():
            seen_ms = max(seen_ms, int(parts[1]))
        raw.write(line + "\n")
        try:
            if parts[0] == "C" and len(parts) == 3:
                f4 = parts[2].split()
                ms = int(parts[1])
                # "C ms a0 a1" (differential, current = a0 - a1) or the older "C ms a0"
                adc = int(f4[0]) - int(f4[1]) if len(f4) == 2 else int(f4[0])
                if ms <= last_ms:          # phase_energy rejects duplicate/backwards times
                    continue
                last_ms = ms
                cur.write(f"{clock(ms)},{adc * k:.3f}\n")
                n_c += 1
            elif parts[0] == "V" and len(parts) == 3:
                ms, adc = int(parts[1]), float(parts[2])
                if adc > 0:
                    vcc.write(f"{ms},{a.vref * 1023.0 / adc:.4f}\n")
                    print(f"  supply {a.vref * 1023.0 / adc:.3f} V at {ms / 1000:.0f} s" + " " * 20)
            elif parts[0] == "M" and len(parts) == 3:
                ms, text = int(parts[1]), parts[2]
                f = text.split()
                if len(f) == 3 and f[0] in ("PHASE", "ROUND"):
                    mk.write(f"{f[0]} {f[1]} {ms}\n")
                    n_m += 1
                else:
                    oth.write(f"{ms} {text}\n")
        except ValueError:
            continue
        if n_c and n_c % 2000 == 0:
            for fh in (raw, cur, mk, oth, vcc):
                fh.flush()
            if (a.out / "STOP").exists():      # graceful stop for a detached capture
                break
            print(f"  {n_c} samples ({n_c / 200 / 60:.1f} min), {n_m} markers", end="\r")
    for fh in (raw, cur, mk, oth, vcc):
        fh.close()
    port.close()
    print(f"\nstopped: {n_c} samples, {n_m} markers in {a.out}")


if __name__ == "__main__":
    sys.exit(main())
