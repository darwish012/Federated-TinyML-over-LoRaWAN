"""Method 3 of the power study: datasheet currents x measured state durations.

    python datasheet_model.py [--json datasheet_model.json] [--acq-interval-s 600] [--battery-mah 700]

Device: LilyGO T3 v1.6.1 (ESP32-PICO-D4 + SX1276) running the protocol-v4 node, plus an
MPU6050 as the live vibration sensor, with no SD card and the OLED switched off (the planned
sensor-only power build). Every current is a datasheet value (CURRENTS, with its source);
every duration is either measured on the real boards in the 2026-10-04/05 sweep
(phase_durations.py on server/night_COM76.log and night_COM80.log, phase 20 = two machines
per board) or computed (LoRa airtime). Values that are estimates pending the method-1
measurement are marked "estimate".

The model brackets the result instead of picking one number: the ESP32 datasheet gives a
range (30-68 mA at 240 MHz, radio off) that depends on the workload, and the SX1276
datasheet has no point at +14 dBm on PA_BOOST (the T3's only PA output), so TX is bracketed
by the +13 dBm RFO point and the +17 dBm PA_BOOST point.
"""
import argparse
import json
import math

# ---------------------------------------------------------------- datasheet currents (mA)
CURRENTS = {
    # Espressif, ESP32 Series Datasheet, "Power Consumption by Power Modes" (Modem-sleep:
    # CPU on, Wi-Fi/BT off), 240 MHz dual-core: 30 mA ~ 68 mA. Arduino-ESP32 runs at 240 MHz
    # and never starts Wi-Fi/BT here; the node never sleeps (Class C).
    "esp32_wait": {"low": 30.0, "high": 68.0, "src": "ESP32 datasheet, Modem-sleep 240 MHz dual-core, 30-68 mA"},
    "esp32_compute": {"low": 68.0, "high": 68.0, "src": "ESP32 datasheet, upper end of 240 MHz Modem-sleep range"},
    # Semtech SX1276 datasheet (Rev 7), Power Consumption: RX LoRa band 1 (862-1020 MHz),
    # BW 125 kHz, LnaBoost on (LMIC sets LNA max gain + boost): 11.5 mA.
    "sx1276_rx": {"low": 11.5, "high": 11.5, "src": "SX1276 datasheet, IDDR_L band 1, BW125, LnaBoost on"},
    # TX at +14 dBm (firmware LMIC_setDrTxpow(DR_SF7,14)) through PA_BOOST: no datasheet point.
    # Low: RFO +13 dBm = 29 mA; high: PA_BOOST +17 dBm = 87 mA.
    "sx1276_tx": {"low": 29.0, "high": 87.0, "src": "SX1276 datasheet, IDDT: RFO +13 dBm 29 mA / PA_BOOST +17 dBm 87 mA"},
    # InvenSense MPU-6000/6050 Product Specification Rev 3.4: accelerometer normal operating
    # current 500 uA (gyro off); full-chip sleep 5 uA.
    "mpu6050_accel": {"low": 0.5, "high": 0.5, "src": "MPU-6050 PS Rev 3.4, accelerometer normal mode 500 uA"},
    "mpu6050_sleep": {"low": 0.005, "high": 0.005, "src": "MPU-6050 PS Rev 3.4, sleep 5 uA"},
    # Solomon Systech SSD1306 datasheet: sleep-mode current <= 10 uA (display off).
    "oled_off": {"low": 0.010, "high": 0.010, "src": "SSD1306 datasheet, sleep mode <= 10 uA"},
}

# ---------------------------------------------------------------- measured / computed times
MEASURED = {
    # Sweep phase 20 (2 machines per board), boards 5 and 7, 12 rounds each, 3.77 h:
    "round_s": 3.77 * 3600 / 12,                 # 1131 s = 18.85 min per round
    "tx_per_round": (251 + 236) / 2 / 12,        # 20.3 uplinks per board per round
    "machines": 2,
    "train_s_per_machine": (0.139 + 0.222) / 2,  # mean "train" phase, s per machine per round
    "flash_read_s_per_machine": (0.045 + 0.078) / 2,  # "sd_read" phase = prepared rows from internal flash
    "update_fragments": 3,                       # UPDATE 598 B = fragments of 213+213+172 B (+8 B header)
    "ul_repair_ratio": 0.53,                     # phase 20: repair requests / received fragments (all boards)
}
SENSING = {
    # Planned sensor build: MPU6050 accelerometer, 3 axes, 1 kHz, 4096 samples per acquisition.
    "sense_s": 4096 / 1000.0,
    "features_s": 0.05,   # estimate: 3 x 4096-point FFT + descriptors on the ESP32 (method 1 will measure)
    "score_s": 0.002,     # estimate: one H12 autoencoder forward pass (562 parameters)
}


def lora_airtime_s(frm_payload, sf=7, bw=125e3, cr=1, preamble=8, header=True, crc=True):
    """Semtech AN1200.13 time-on-air; PHYPayload = 13 B LoRaWAN overhead + FRMPayload."""
    pl = 13 + frm_payload
    t_sym = (2 ** sf) / bw
    de = 1 if t_sym > 0.016 else 0
    h = 0 if header else 1
    n = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + (16 if crc else 0) - 20 * h) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n * t_sym


def model(bound, acq_interval_s, battery_mah, usable=0.85, v_bat=3.7):
    c = {k: v[bound] for k, v in CURRENTS.items()}
    m = MEASURED
    t_round = m["round_s"]
    long_frames = m["machines"] * m["update_fragments"] * (1 + m["ul_repair_ratio"])
    short_frames = max(m["tx_per_round"] - long_frames, 0.0)
    t_long = lora_airtime_s((221 + 221 + 180) / 3)        # mean UPDATE fragment frame
    t_short = lora_airtime_s(15)                            # READY/STATUS/HELLO-size control
    t_tx = long_frames * t_long + short_frames * t_short
    t_train = m["machines"] * m["train_s_per_machine"]
    t_flash = m["machines"] * m["flash_read_s_per_machine"]
    acq_per_round = t_round / acq_interval_s * m["machines"]   # one acquisition per machine per interval
    t_sense = acq_per_round * SENSING["sense_s"]
    t_feat = acq_per_round * (SENSING["features_s"] + SENSING["score_s"])
    t_listen = t_round - t_tx - t_train - t_flash - t_sense - t_feat

    base_off = c["oled_off"] + c["mpu6050_sleep"]
    phases = {   # name: (seconds per round, current mA)
        "listen (Class C RX, CPU waiting)": (t_listen, c["esp32_wait"] + c["sx1276_rx"] + base_off),
        "tx (on air)": (t_tx, c["esp32_wait"] + c["sx1276_tx"] + base_off),
        "train": (t_train, c["esp32_compute"] + c["sx1276_rx"] + base_off),
        "flash read (prepared rows)": (t_flash, c["esp32_compute"] + c["sx1276_rx"] + base_off),
        "sense (MPU6050 1 kHz)": (t_sense, c["esp32_compute"] + c["sx1276_rx"] + c["oled_off"] + c["mpu6050_accel"]),
        "features + score": (t_feat, c["esp32_compute"] + c["sx1276_rx"] + base_off),
    }
    rows, total_mAs = {}, 0.0
    for name, (t, i) in phases.items():
        q = i * t                                   # mA*s per round
        total_mAs += q
        rows[name] = {"seconds_per_round": round(t, 3), "current_mA": round(i, 3),
                      "charge_mAs": round(q, 2), "energy_mJ_at_vbat": round(q * v_bat, 1)}
    for r in rows.values():
        r["share"] = round(r["charge_mAs"] / total_mAs, 5)
    i_avg = total_mAs / t_round
    life_h = battery_mah * usable / i_avg
    return {"bound": bound, "currents_mA": c, "round_s": round(t_round, 1),
            "frames_per_round": {"long": round(long_frames, 2), "short": round(short_frames, 2),
                                 "airtime_long_s": round(t_long, 4), "airtime_short_s": round(t_short, 4)},
            "acquisitions_per_round": round(acq_per_round, 2), "phases": rows,
            "per_round_charge_mAh": round(total_mAs / 3600, 4),
            "per_round_energy_J_at_vbat": round(total_mAs * v_bat / 1000, 2),
            "average_current_mA": round(i_avg, 2), "battery_mAh": battery_mah, "usable_fraction": usable,
            "battery_life_h": round(life_h, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acq-interval-s", type=float, default=600.0, help="one acquisition per machine every N s")
    ap.add_argument("--battery-mah", type=float, default=700.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {"sources": {k: v["src"] for k, v in CURRENTS.items()}, "measured": MEASURED, "sensing": SENSING,
           "acq_interval_s": a.acq_interval_s}
    for bound in ("low", "high"):
        r = model(bound, a.acq_interval_s, a.battery_mah)
        out[bound] = r
        print(f"\n=== {bound.upper()} bound: average {r['average_current_mA']} mA, "
              f"{r['per_round_charge_mAh']} mAh ({r['per_round_energy_J_at_vbat']} J at 3.7 V) per {r['round_s']/60:.2f}-min round, "
              f"{a.battery_mah:.0f} mAh x {r['usable_fraction']} -> {r['battery_life_h']} h")
        print(f"    frames/round: {r['frames_per_round']}; acquisitions/round: {r['acquisitions_per_round']}")
        for name, p in r["phases"].items():
            print(f"    {name:34s} {p['seconds_per_round']:9.2f} s  {p['current_mA']:7.2f} mA  "
                  f"{p['energy_mJ_at_vbat']:10.1f} mJ  {100 * p['share']:6.2f} %")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
