> Note: this operator README predates protocol rev 4.3; the GLOBAL message is 624 B (see `docs/PROTOCOL_V4.md`).

# TinyFL v4 rev 4.2 firmware

Active entry point: `tinyfl_fl_node.ino` → `node_v4.h`. The older `node_app.h`
is retained for historical v3 reference and is not compiled by this sketch.
The v3 numerical codec/tests remain separate; feature extraction, standardisation,
two-epoch training, quantisation and the 13-of-21/history-max alarm rule are unchanged.

## Provisioning and byte order

The separate `../tinyfl_provision/` sketch is offline: it contains no LMIC or
radio code. Its first boot formats **only LittleFS**, then emits at 115200 8N1:

```text
TFPROV HELLO 42 <16-hex-digit DevEUI, MSB first>
```

It repeats HELLO once per second for at most 120 seconds. The PC's committed
`server/provision_board.py` sends:

```text
TFPROV KEYS <JoinEUI:16hex> <AppKey:32hex> <MCAddr:8hex> <MCNwk:32hex> <MCApp:32hex>
```

Each line ends in LF; CRLF is also accepted. The board parses bounded input,
stores and reads back the five raw values in `tinyflkeys`, compares all 60 bytes,
then responds `TFPROV OK <8hex CRC32>` or `TFPROV ERR <non-secret reason>`.
Neither sketch prints keys. AppKey derivation remains on the operator PC only.

| NVS key | Raw bytes | Offset in provisioning CRC input |
|---|---:|---:|
| joineui | 8 | 0 |
| appkey | 16 | 8 |
| mcaddr | 4 | 24 |
| mcnwk | 16 | 28 |
| mcapp | 16 | 44 |

CRC is zlib-compatible `crc32(joineui || appkey || mcaddr || mcnwk || mcapp)`
over **60 bytes**. NVS keeps hex-written order. Only `os_getArtEui` and
`os_getDevEui` reverse EUIs for LMIC. AppKey and both multicast keys are never
reversed. DevAddr is `mcaddr[0]<<24 | mcaddr[1]<<16 | mcaddr[2]<<8 | mcaddr[3]`.

`tinyfl_fl_node` shows `NOT PROVISIONED` and does not start LMIC when any key
is absent or has the wrong length. It never includes a credentials header or
uses a compiled fallback. The old example credentials file is inactive in v4.
Provisioning and run adoption leave `tinyflmc` untouched; multicast counter
storage remains `fcnt<8-uppercase-hex-digit-DevAddr>`, compatible with v2.
Do not erase NVS while reusing multicast keys; acceptance-to-counter-save has
the existing brief power-loss window, not an atomic hardware anti-replay guarantee.

## Wire and board behaviour

The C++ codec `../common/tinyfl_core/tf_protocol_v4.h` follows the committed
`server/protocol_v4.py`, including its `MAX_SLOT=65534` / `NO_SLOT=65535`
reservation. The prose contract's section 8 says 65535 slots; firmware follows
the executable server limit so the sentinel cannot be mistaken for a held slot.

| Item | v4 value |
|---|---|
| Header | 8 bytes: F1, 04, u16 slot LE, index, count, body length, type |
| UPDATE | 598 bytes; CRC covers bytes 0..593 |
| UPDATE frames | 221 / 221 / 186 bytes; each body begins with u16 position |
| GLOBAL | 590 bytes; CRC covers bytes 0..585 |
| GLOBAL frames | 221 / 221 / 172 bytes |
| Run / board / position | u32 run; u16 board and position |
| Phases / rounds | phase 0..254; round 1..R from GLOBAL, not a fixed 12-round sweep |
| Machines | 1..8 unique, sorted hosted positions |

Slots start at **1**; persisted held slot **0** means no model held. Before
holding a GLOBAL of the adopted run, READY body/header slot is **0xFFFF**.
Otherwise it is **held_slot + 1**: after all UPDATE fragments, after answering
repairs, and immediately after persisting a phase-final model. UPDATE header
slot is the **seed GLOBAL slot**, not the result slot. UPDATE body base_slot
remains the actual CRC-valid model trained from, including stale-copy repairs.
A late fresh GLOBAL of the same training round does not mutate cached UPDATEs.

All ten message types are covered, including run-scoped controls and
REPLAY_DONE for each `(position, phase)`. All hosted positions train; the
server ignores positions outside its plan. There is no compiled phase mask.
Final models go to flash individually; only RUN_CONTROL REPLAY starts replay.
ADOPT archives results and resets the run identity. A different run's round-zero
GLOBAL can adopt; any complete GLOBAL can adopt if no run is stored or the
stored run is complete. Other different-run GLOBALs and controls are ignored.
END retains results/models and returns to Class A idle. Completed boards re-arm
listening after reboot so a server's subsequent ADOPT can reach them.

## SD and persistence

`board.txt` is format 2, e.g.:

```text
board_id 1000
format_version 2
position 513 machine_id example_bearing streams 1
position 65534 machine_id example_other streams 1
```

Raw/check inputs: `m0201s0.raw`, `m0201.chk`, `mfffes0.raw`, `mfffe.chk`.
Four lowercase hexadecimal position digits keep names 8.3-safe. Packed TFR1/TFC1
data geometry is unchanged. No `levels.csv` is read. Streams remain limited
to four, channels to six, train rows to 997 and acquisition count to 100000.
Raw/check inputs are only read; the board writes derived `.x` and scratch files.
Both CRCs, all scales, exact sizes and partitions are checked before allocation.
Every hosted machine's self-check must pass at max error <=1e-3.

Run metadata and held GLOBAL use two generation-numbered CRC checkpoints.
UPDATE bytes are immutable flash caches. Replay snapshots use separate paired
CRC ring-state files committed before the referencing metadata checkpoint.
CSV rollback truncates to the committed byte offset after reboot. Outboxes are
durable; alarms and completion records pause replay until the transaction ends.
Repeated REPLAY commands retransmit completed-pair DONE records without clearing
their completion bits or restarting their scoring. Alarm delivery is unconfirmed:
an interrupted send can repeat, and a completed RF transaction is not proof of
server receipt. Power-loss/card-change tests on hardware remain necessary.

Replay extracts each acquisition once for its current requested batch and
streams phase models from flash one at a time, with independent alarm states.
All 255 phases can be requested using the server's <=207-entry control chunks;
later chunks queue additional work without corrupting an in-progress batch.
CSV columns are exactly `position,index,phase,score,flag`; flags are instantaneous
alarm-rule decisions, not the permanently latched alarm state. Model run/phase,
final flag and CRC are revalidated on each flash read. Flash/model/read/heap
failures stop safely; large phase counts still require measured ESP32 heap and
runtime evidence, not just a host pass.

## Hardware and markers

Radio SPI: SCK5/MISO19/MOSI27/NSS18/RST23/DIO26,33,32. OLED: SSD1306 0x3c,
I2C21/22. SD_MMC defaults to 1-bit CLK14/CMD15/D0=2, DAT3/CS13 held high.
`TINYFL_SD_MMC=0` uses HSPI SCK14/MISO2/MOSI15/CS13, 20MHz. Both builds compile.
LMIC retains EU868 duty-cycle scheduling; ADR is off, TX/RX2 must be DR5.
The post-join hash stagger does not alter OTAA. The loop watchdog is enabled
and fed in numerical yield hooks; brown-out protection is not disabled.

`TINYFL_PHASE_MARKERS=1` emits `PHASE <name> <unsigned millis>` on changes,
with `sd_read`, `features`, `train`, `tx`, `rx_classc`, `idle`, `replay`.
`tx` includes the subsequent Class A receive windows; duty-cycle queue waits
are idle. Replay may retain Class C reception while processing commands, so
markers identify dominant software work, not isolated hardware rail states.
Round markers: `ROUND run<ID>-P<phase>:R<round> <millis>` and
`ROUND end <millis>`. Separate captures at reboot and synchronise board uptime
with the power logger as described by `final lab/phase_energy.py`.

## Reproduce verification (compile only)

From `E:\masters\lorawan_two_node_experiment\firmware\host_tests` with Python
3.12, NumPy, cryptography and Zig 0.14.1 installed:

```powershell
.\build_host.ps1 -Python python
python -B .\test_core.py --report .\REPORT.json
.\build_host.ps1 -Python python -V4 -Out "$PWD\tfcore_v4.dll"
.\build_node_host.ps1 -Python python
python -B .\test_v4.py --report .\REPORT_V4.json
```

The v4 DLL compiles the production portable codec/provisioning/NVS adapter with
fake Preferences; the node DLL compiles the **actual `node_v4.h`** with fake
hardware. The fake serial pair runs the actual Python PC provisioning function,
not its built-in Python fake board. Only synthetic public key vectors and a
temporary roster are used. Reference imports disable bytecode writes.

Arduino CLI uses FQBN `esp32:esp32:ttgo-lora32:Revision=TTGO_LoRa32_v21new`,
local libraries `firmware/lib`, ESP32 3.3.7 and MCCI LMIC 6.0.1. Keep build_opt.h
(Class C/long messages, 16KiB stack, EU868, no fast math/FMA). The existing
`-Wno-int-conversion` handles LMIC's C logging callback with this GCC version;
its C++ warning is harmless. Installed libraries are not modified.
Compile `tinyfl_fl_node` normally, again with
`--build-property 'compiler.cpp.extra_flags=-DTINYFL_SD_MMC=0'`, and
`tinyfl_provision`. **No verification command uploads or opens a serial port.**

See VERIFICATION_V4.md for measured results and the remaining hardware boundary.
