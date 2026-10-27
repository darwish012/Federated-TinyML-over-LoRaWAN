# Federated TinyML over LoRaWAN: Label-Free Failure Detection on Microcontrollers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code release for the paper

> A. Darwish and T. Elshabrawy, "Federated TinyML over LoRaWAN: Label-Free Failure Detection on Microcontrollers," submitted to *IEEE Internet of Things Journal*, 2026.

Remote bearing monitoring over LoRaWAN has to work without fault labels and without sending raw vibration recordings. This repository contains a complete federated learning (FL) pipeline in which LoRaWAN microcontrollers train a shared anomaly detector on their own data. Each node extracts 22 features from raw vibration, standardises them locally and runs two full-batch gradient steps per round on a 562-parameter autoencoder. A Python server aggregates the updates with FedAdam over a fixed 12-round budget. Updates and global models travel as int8 codes with per-tensor scales, so at EU868 data rate DR5 every exchange fits in three LoRaWAN fragments per direction (598 B and 624 B), with application-layer fragment repair and Class C multicast for the downlink. A causal persistence rule turns reconstruction scores into alarms. The release contains the ESP32 firmware, the LoRaWAN multicast companion library, the server, the chronological simulation and hardware twin used for the evaluation, the hardware-versus-twin comparison tool, the power-measurement tooling and the sealed unseen-machine test.

## Highlights

- **On-device FL in plain C++ with no ML libraries.** The numerical core (about 3,100 lines in `firmware/common/tinyfl_core/`) uses only the C/C++ standard library: its own FFT, feature extractor, robust standardisation, autoencoder with hand-written backpropagation, int8 codec with CRC32, alarm state machine and protocol codec. No TensorFlow Lite Micro, Edge Impulse or CMSIS-NN.
- **LMIC multicast extension without patching LMIC.** MCCI LMIC 6.0.1 supports Class C only on a device's own unicast session. `firmware/lib/TinyFLMulticast/` is a companion library that captures multicast frames through LMIC's existing event hook, verifies the MIC, decrypts with LMIC's AES routine and persists the downlink counter. The stock library is not modified.
- **int8 per-tensor codec.** A 598 B UPDATE and a 624 B GLOBAL model each fit in three fragments per direction at DR5 (SF7, 125 kHz); lower data rates need more fragments.
- **Class C multicast with application-layer repair.** One multicast burst delivers the global model to all boards; boards that miss fragments report what is missing and the server repairs it.
- **No ML framework in Python either.** The server, the chronological simulation and the hardware twin use NumPy for all model and training numerics; the autoencoder, local training, FedAvg, FedAdam and FedProx are implemented directly, without PyTorch, TensorFlow, scikit-learn or Flower. (The server also uses paho-mqtt and cryptography for transport; the published-method reimplementations in `simulation/research/published_methods_20261021/` use SciPy and scikit-learn.) The C++ core is checked value for value against the Python reference on a host.
- **10-board LoRaWAN testbed.** Ten LilyGO ESP32/SX1276 boards ran the full federated loop through a real gateway and The Things Stack.

## Architecture

```
  Node (x10)                     Gateway / network server                 Host PC
  LilyGO T3 v1.6.1               Kerlink gateway                          Python server (server/)
  ESP32 + SX1276                 The Things Stack (MQTT, HTTP API)
 +--------------------+  UPDATE   +-----------------------+  MQTT      +----------------------------+
 | SD / MPU6050       |  (Class A,| LoRa gateway          | uplinks    | tts_v4_server.py           |
 | 22 features        |  3 frags) | + network server      |----------->|  controller_v4.py          |
 | 562-param AE       |---------->|                       |            |   - membership, run plan   |
 | 2 local epochs     |           |                       |            |   - FedAdam aggregation    |
 | int8 codec + CRC32 |  GLOBAL   |                       | downlinks  |   - repair, replay, END    |
 | alarm rule         |<----------| Class C multicast     |<-----------|  run directory (logs,      |
 | TinyFLMulticast    |  (3 frags,| (one burst, all nodes)|  + unicast |  masks, global models)     |
 +--------------------+  + repair)|                       |  repair    +-------------+--------------+
          ^                       +-----------------------+                          |
          | serial, once per board                                                   v
   provision_board.py  <-- per-device AppKey, multicast keys           analysis/compare_hw_twin_v4.py
   (tinyfl_provision sketch)                                           replays the run through the
                                                                       int16 hardware twin (simulation/)
```

One federated round: the server multicasts the GLOBAL model (three fragments, repaired until every live board holds it), each board trains locally and sends one three-fragment UPDATE on its own Class A uplink, and the server repairs missing fragments by unicast, aggregates with FedAdam and starts the next round. After the last phase the boards replay their history, score it with the final models and report alarms. The full wire protocol is specified in [`docs/PROTOCOL_V4.md`](docs/PROTOCOL_V4.md).

## Repository layout

```
.
├── firmware/                  ESP32 firmware (Arduino framework) and host-side parity tests
│   ├── tinyfl_fl_node/        the node sketch (entry point: tinyfl_fl_node.ino -> node_v4.h)
│   ├── tinyfl_provision/      offline provisioning sketch (no radio code)
│   ├── common/                portable core (tinyfl_core/) and acquisition/storage headers
│   ├── lib/TinyFLMulticast/   Class C multicast companion library for MCCI LMIC
│   ├── host_tests/            C API wrappers, mock Arduino/LMIC headers, Python parity tests
│   └── lmic_project_config.h  LMIC settings (also supplied as build flags, see build_opt.h)
├── server/                    protocol-v4 server (MQTT bridge, controller, provisioning, enrollment)
│   └── tests/                 pytest suite with fake boards and a fake network
├── simulation/                chronological simulation, FL algorithms, hardware twin
│   └── research/              dated study folders (kept under this name, see "Path assumptions")
├── python_tools/              legacy modules imported by the simulation (earlier TinyFedLoRA work)
├── analysis/                  hardware-versus-twin comparison of a server run
├── power/                     power-measurement firmware and analysis scripts
├── docs/PROTOCOL_V4.md        wire protocol and behaviour contract (rev 4.3)
├── unseen_machines/           sealed unseen-machine test: protocol, runner, frozen models, results
├── LICENSE  CITATION.cff  .gitignore  README.md
```

## What each file does

### `firmware/tinyfl_fl_node/` (node sketch)

| File | Purpose |
|---|---|
| `tinyfl_fl_node.ino` | Arduino entry point; includes `node_v4.h` and raises the loop task stack to 24 KB for training and replay. |
| `node_v4.h` | The v4 node: LMIC join and Class A/C scheduling, SD (SdFat) or sensor-store input, persistence of run state in two CRC-protected generations, UPDATE/GLOBAL fragment handling and repair, replay and alarms, OLED status, phase markers. Also holds the sensor-only power build (`TINYFL_SENSOR`). |
| `node_app.h` | The older v3 application, kept for reference. It is not compiled by the sketch. It falls back to `lab_credentials.example.h` if no `lab_credentials.h` exists. |
| `build_opt.h` | Compiler flags picked up by Arduino: EU868, SX1276, Class C, long messages, beacons/ping off, 16 KiB loop stack, no fast-math and no FMA contraction. |
| `lab_credentials.example.h` | Template for the legacy credentials header (all zeros). The v4 node does not use it; v4 boards get their keys from the provisioning sketch. |
| `README_V4.md` | Firmware notes: provisioning byte order, wire layout, SD card format, persistence, hardware pins, phase markers, how host verification is built. |

### `firmware/tinyfl_provision/`

| File | Purpose |
|---|---|
| `tinyfl_provision.ino` | Entry point of the provisioning sketch. |
| `provision_app.h` | Offline provisioning: formats LittleFS, prints `TFPROV HELLO` with the DevEUI over serial, accepts `TFPROV KEYS`, stores the five key values in NVS, reads them back and replies `TFPROV OK <crc32>`. No LMIC or radio code; keys are never printed. |

### `firmware/common/`

| File | Purpose |
|---|---|
| `tinyfl_core/tinyfl_core_all.cpp` | Single translation unit that pulls in all core `.cpp` files for the Arduino build. |
| `tinyfl_core/tf_common.h/.cpp` | Model-shape and data-limit constants, yield/progress hooks, heap accounting, error buffer, CRC32. |
| `tinyfl_core/tf_fft.h/.cpp` | Real-input power spectrum for any length (odd, prime or mixed radix) in a small memory footprint. |
| `tinyfl_core/tf_features.h/.cpp` | Streaming float32 extractor: one pass per channel, 11 descriptors per channel collapsed to 22 features. |
| `tinyfl_core/tf_source.h` | `TfFile`, `TfFileSystem` and `TfAcquisitionSource` interfaces; SD replay and the sensor are interchangeable behind them. |
| `tinyfl_core/tf_rawfile.h/.cpp` | Reader for the packed int16 raw files and the reference-feature check file, with CRC and size preflight. |
| `tinyfl_core/tf_machine.h/.cpp` | One hosted machine: card files to features, standardisation, training matrix, self-check and history replay. |
| `tinyfl_core/tf_model.h/.cpp` | The 22-12-22 tanh autoencoder: forward pass, full-batch local training, scoring, standardisation and the int8 codec. |
| `tinyfl_core/tf_alarm.h/.cpp` | Causal alarm layer (window 21, 13th largest, frozen threshold) with constant memory per machine and model. |
| `tinyfl_core/tf_protocol.h/.cpp` | Protocol v3 message codec, kept as a regression oracle. |
| `tinyfl_core/tf_protocol_v4.h` | Protocol v4 rev 4.3 codec: header, UPDATE/GLOBAL fragments, control messages. |
| `tinyfl_core/tf_provision_v4.h` | The 60-byte key block and its layout for provisioning. |
| `tinyfl_core/tf_replay_v4.h` | Feature replay that extracts each acquisition once and streams models from flash. |
| `tinyfl_imu_source.h` | Non-producing stub of a live IMU source (no fabricated readings). |
| `tinyfl_mpu6050_source.h` | Live MPU6050 acquisition source for the sensor-only power build (4096 samples at 1 kHz, 3 axes, FIFO). |
| `tinyfl_sensor_store.h` | LittleFS feature store of the sensor-only build: append-only CRC-protected rows, preparation and replay. |
| `tinyfl_nvs_keys.h` | Loads the five provisioned key values from the `tinyflkeys` NVS namespace. |
| `experiment_protocol.h` | Constants of the earlier v2 experiment protocol. Not included by the v4 sketch. |

### `firmware/lib/TinyFLMulticast/`

| File | Purpose |
|---|---|
| `library.properties` | Arduino library metadata. Pass the folder with `--library`. |
| `src/TinyFLMulticast.h/.cpp` | Class C multicast reception beside MCCI LMIC: captures frames through LMIC's event hook, checks the MIC, decrypts, and persists the multicast frame counter. |

### `firmware/host_tests/`

| File | Purpose |
|---|---|
| `build_host.ps1` | Builds the portable core and the C API into a host DLL with `zig c++` (`-V4` for the v4 build). |
| `build_node_host.ps1` | Builds the actual `node_v4.h` against fake hardware into `tfnode_v4.dll`. |
| `run_sensor_store_test.ps1` | Builds and runs `test_sensor_store.cpp`. |
| `tf_capi.cpp` | C API over the core for `ctypes` (v3 parity tests). |
| `v4_capi.cpp` | C API over the v4 codec, provisioning and NVS adapter. |
| `node_v4_capi.cpp` | C API that drives the real node sketch with fake radio, SD and flash. |
| `test_core.py` | Parity of the C++ core against the Python twin (features, training, codec, v3 messages). |
| `test_v4.py` | Interoperability of the v4 C++ code and node with the Python codec, provisioning exchange and persistence. |
| `test_sensor_store.cpp` | Host test of the sensor-only feature store. |
| `mock/` | Mock `Arduino.h`, `lmic.h`, `FS.h`, `SD.h`, `LittleFS.h`, `Preferences.h`, display, `TinyFLMulticast.h` and a `node_platform.h` with in-memory SD/LittleFS. |

### `server/`

| File | Purpose |
|---|---|
| `tts_v4_server.py` | Entry point: MQTT bridge to The Things Stack for protocol v4; run, resume, enrollment window, console commands. |
| `controller_v4.py` | Pure protocol-v4 state machine: phases and rounds, Class C multicast GLOBAL with repair, UPDATE collection, FedAdam aggregation, replay and END. |
| `protocol_v4.py` | Protocol v4 wire codec (8-byte header, message types, int8 codec). |
| `membership_v4.py` | Board membership (rostered, bound, active, suspect, departed). |
| `plan_v4.py` | Run plans: planned phases or an elastic run. |
| `config_v4.py` | Validation of `config.toml`; reports every problem at once and never reads a secret. |
| `run_outputs_v4.py` | Run-directory outputs: events, delivery log, global models, replay masks, alarms, checkpoint. |
| `logging_v4.py` | Structured JSON event log with secret redaction and the console status line. |
| `roster_v4.py` | The roster of authorised DevEUIs (`roster.json`, no keys). |
| `keys_v4.py` | Per-device AppKey derivation (AES-128-ECB of the master key) and the provisioning CRC. |
| `enrollment_v4.py` | Hardened automatic enrollment of rostered boards into The Things Stack. |
| `provision_board.py` | PC side of serial provisioning (`TFPROV HELLO/KEYS/OK`), with a fake board for dry runs. |
| `study_plan.py` | Machine order, levels and the loader for the research modules (`research_root`). |
| `configure_multicast.py` | Writes the multicast session on both sides; targets the legacy `tinyfl_node` credentials header, which is not part of this release. |
| `make_credentials.py` | Converts TTS OTAA values into an LMIC credentials header (legacy sketch). |
| `controller.py`, `protocol.py`, `run_outputs.py`, `tts_class_c_server.py` | The earlier two-node experiment (v2), still imported by v4 modules and tests. |
| `controller_v3.py`, `protocol_v3.py`, `enrollment.py`, `lorawan_join.py`, `tts_v3_server.py` | The earlier 10-board v3 server and its shared AppKey enrollment; `protocol_v3.py` also supplies the int8 codec used by v4. |
| `config.toml.example` | Annotated configuration template. Copy to `config.toml` (not tracked). |
| `requirements.txt` | `paho-mqtt`, `numpy`, `cryptography`, `pyserial`, `pytest`. |
| `tests/` | pytest suite: protocol codecs, controller, membership, liveness, run plans and configuration, provisioning, enrollment, resume, rev 4.3 behaviour, MQTT glue, and fake-network simulators (`v3_sim.py`, `v4_sim.py`, `v4_learning_sim.py`, `v4_ref.py`). |

### `simulation/`

The hardware twin and the study runners live in `simulation/research/`; the root of `simulation/` holds the modules they import. Studies read cached feature files that are not part of this release (see "Datasets and data caches").

| File | Purpose |
|---|---|
| `research/fl_algorithm_comparison_20260929/fl.py` | Generic tanh autoencoder and FedAvg, FedAdam and FedProx (NumPy only). |
| `research/fl_algorithm_comparison_20260929/dropout_fl.py` | Client-dropout and packet-loss mechanics for the robustness study. |
| `research/fl_algorithm_comparison_20260929/run_study.py` | FedAvg/FedAdam/FedProx comparison under one label-free stopping rule, plus a depth arm. |
| `research/fl_algorithm_comparison_20260929/run_amendment.py`, `run_amendment2.py` | Amendment 1 (round-budget study, depth falsification) and amendment 2 (detector-aligned stopping rule). |
| `research/fl_algorithm_comparison_20260929/run_dropout.py` | Client-dropout and packet-loss robustness experiment. |
| `research/fl_algorithm_comparison_20260929/test_*.py`, `*.json` | Unit tests (no caches needed) and the pre-declared protocol and amendment files. |
| `research/hardware_twin_scaling_20260929/twin.py` | Device-side twin of the firmware: float32 arithmetic, standardisation, local training, int8 codec. |
| `research/hardware_twin_scaling_20260929/twin_server.py` | Float64 FedAdam server that aggregates already-dequantised updates; used by the real server. |
| `research/hardware_twin_scaling_20260929/machine_order.py`, `machine_order_v2.py` | The fixed machine order and board assignment (`amendment2_int16_raw_order.json`). |
| `research/hardware_twin_scaling_20260929/raw_common.py`, `raw_reference_features.py`, `raw_features_f32.py`, `raw_int16_v2.py` | Raw-waveform access, the float64 reference extractor, the float32 extractor mirroring the firmware, and the int16 raw representation. |
| `research/healthy_only_causal_20260908/wider_models.py`, `cadence.py` | Healthy-only autoencoder family (H12 is the deployed one) and the causal alarm rule (`dense_flags`). |
| `research/healthy_only_causal_20260908/run_study.py`, `test_*.py`, `protocol.json` | Causal cadence and model study runner, tests and protocol. |
| `research/healthy_only_causal_no_ldm_20260922/` | The same study on the final 54-client cohort (runner, tests, protocol). |
| `research/healthy_only_suite_20260907/` | `models.py`, `calibration.py`, `run_suite.py`, `run_ensembles.py` and protocols imported by the studies above. |
| `research/published_methods_20261021/` | Reimplementation of three published decision rules on the same cohort (`run_comparison.py`, `protocol.json`, `test_summary.py`); needs SciPy and scikit-learn. |
| `research/simulation_completion_20260929/quant_fl.py` | int8 quantisation of every FL exchange; the server codec is tested against it. |
| `thesis_participating_node_scaling.py`, `realtime_federated_simulation.py`, `improved_comprehensive_experiment.py`, `learning_failure_experiments.py`, `run_full_simulation.py` | Earlier simulation framework imported by the study runners. |
| `raw_online_federated_v12.py` ... `raw_online_federated_v63_policy.py` (and `*_protocol*.json`) | Version-numbered modules of the earlier raw-data research pipeline. They are included only because the hardware twin's raw-data layer (`raw_common.py`) imports them. |

### `python_tools/`

`tiny_ae.py`, `cwru_eval.py` and `compare_architectures.py` are modules of the authors' earlier TinyFedLoRA work, imported by the simulation framework above.

### `analysis/`

| File | Purpose |
|---|---|
| `compare_hw_twin_v4.py` | Replays a v4 server run directory through the unchanged int16 hardware twin using the logged delivery masks and base slots, and checks the pre-declared hardware-versus-twin criteria per phase (identical int8 code vectors, strict outcomes, alarm timing, completion). |
| `synth_v4_run.py` | Generates synthetic v4 run directories (test fixtures). |
| `test_compare_hw_twin_v4.py` | Tests of the comparison tool on synthetic runs. |

### `power/`

| File | Purpose |
|---|---|
| `uno_daq/uno_daq.ino` | Arduino Uno DAQ: board current through a 1-ohm low-side shunt (differential A0/A1) plus the board's phase markers, on one clock. |
| `aref_reader/aref_reader.ino` | Calibration helper: reads the Uno's internal reference with the ESP32's calibrated ADC. |
| `capture_method1.py` | Reads the Uno over serial and writes raw log, current, marker, supply-voltage and other logs. |
| `analyse_method1.py` | Energy per phase, round and event from a capture. |
| `merge_method1.py` | Merges method-1 reports captured in parts. |
| `phase_durations.py` | Phase durations measured on real boards from the firmware's serial markers. |
| `datasheet_model.py`, `datasheet_model.json` | Datasheet-current model combined with measured state durations. |

The power scripts mention `phase_energy.py` of a separate lab folder; that script is not part of this release.

### `unseen_machines/`

The pre-registered test of the frozen models on machines outside the development cohort: six wind turbines of one wind farm and a wind-turbine high-speed-shaft bearing. It holds the sealed protocol and amendments, the runner, hash-sealed copies of the pipeline and the three frozen global models, and the results. See [`unseen_machines/README.md`](unseen_machines/README.md).

### `docs/`

| File | Purpose |
|---|---|
| `PROTOCOL_V4.md` | Wire protocol v4 rev 4.3: frame, messages, server and board behaviour, run-directory schema, provisioning, SD layout, limits. |

## Hardware

- 10 x **LilyGO T3 v1.6.1** (ESP32-PICO-D4, Semtech SX1276, EU868, SSD1306 OLED, microSD slot)
- **Kerlink** LoRaWAN gateway with a self-hosted **The Things Stack** (MQTT and HTTP API)
- **MPU6050** accelerometer for the sensor-only power build
- **Arduino Uno** with a 1-ohm shunt for the power measurement (`power/`)

## Software requirements

- Arduino IDE or `arduino-cli`, **ESP32 core 3.3.7**, FQBN `esp32:esp32:ttgo-lora32:Revision=TTGO_LoRa32_v21new`
- **MCCI LMIC 6.0.1** with Class C enabled (the sketch's `build_opt.h` sets the flags; no library files are edited), plus the Adafruit GFX and SSD1306 libraries and SdFat
- **Python 3.12** and **NumPy** for the server, simulation, twin and tests (`pip install -r server/requirements.txt`)
- SciPy and scikit-learn only for the published-method reimplementations in `simulation/research/published_methods_20261021/`
- For the host parity tests: the `ziglang` PyPI package (Zig 0.14.1) and PowerShell; the host DLLs are cross-compiled for `x86_64-windows-gnu`, so these tests are Windows-oriented

## Quick start

### 1. Build and flash the firmware

```powershell
$fqbn = "esp32:esp32:ttgo-lora32:Revision=TTGO_LoRa32_v21new"
# compile only
arduino-cli compile --fqbn $fqbn --library firmware/lib/TinyFLMulticast firmware/tinyfl_fl_node
# compile and flash a board
arduino-cli compile --upload --port COM5 --fqbn $fqbn --library firmware/lib/TinyFLMulticast firmware/tinyfl_fl_node
```

Arduino picks up `build_opt.h` from the sketch folder. The v4 node sketch does **not** include a credentials header: AppKey, JoinEUI and multicast keys are written to each board by the provisioning step below. `lab_credentials.example.h` is only a template for the legacy v3 application (`node_app.h`); if you build that variant, copy it to `lab_credentials.h` locally and fill in your own values, and never commit that file.

### 2. Provision the boards

Set the secrets as environment variables (they are never stored in files or printed):

```powershell
$env:TINYFL_MASTER_KEY   = "<32 hex digits, your operator master key>"
$env:TINYFL_MC_DEVADDR   = "<8 hex digits, multicast DevAddr>"
$env:TINYFL_MC_NWKSKEY   = "<32 hex digits>"
$env:TINYFL_MC_APPSKEY   = "<32 hex digits>"
```

Then for each board: flash the provisioning sketch, send the keys, flash the node firmware.

```powershell
arduino-cli compile --upload --port COM5 --fqbn $fqbn firmware/tinyfl_provision
cd server
python provision_board.py --port COM5      # waits for TFPROV HELLO (up to 120 s), derives this board's AppKey, adds the DevEUI to roster.json
cd ..
arduino-cli compile --upload --port COM5 --fqbn $fqbn --library firmware/lib/TinyFLMulticast firmware/tinyfl_fl_node
```

`provision_board.py --dry-run` exercises the exchange against a fake board without hardware. The replay build reads the raw acquisitions of each board from its microSD card (`board.txt`, `m<PPPP>s<S>.raw`, `m<PPPP>.chk`, format in `docs/PROTOCOL_V4.md` section 7); the card images are produced from the public datasets and are not included.

### 3. Configure and run the server

```powershell
cd server
pip install -r requirements.txt
copy config.toml.example config.toml        # then edit: TTS host/application, gateway ids, multicast_device_id, join_eui
```

`research_root` in `config.toml.example` already points to `../simulation/research`, relative to `server/`; the server imports `twin_server.py`, `fl.py` and `machine_order_v2.py` from there. Provide the Things Stack keys through the environment, then start the server **before** powering the boards:

```powershell
$env:TTS_MQTT_API_KEY = "<application MQTT API key>"
# only when opening an enrollment window:
$env:TTS_GATEWAY_API_KEY = "<gateway traffic read key>"; $env:TTS_DEVICE_API_KEY = "<device registration key>"

python tts_v4_server.py --config config.toml --check-config
python tts_v4_server.py --config config.toml --output results/run1 [--enroll-minutes 10]
python tts_v4_server.py --config config.toml --resume results/run1      # after a crash or Ctrl-C
```

`config.toml.example` documents every `[v4]` key (planned sweep of phases 4, 8, ..., 28 or an elastic run, start rules, repair and liveness timers). A run directory (`events.jsonl`, `delivery_log.jsonl`, `global_models.jsonl`, `replay_masks.json`, alarms, checkpoint) is written under the path given to `--output`.

### 4. Run the host parity tests

The firmware core and the real node sketch are compiled on the host and compared with the Python implementations.

```powershell
pip install numpy cryptography ziglang pytest
cd firmware/host_tests
.\build_host.ps1 -Python python
.\build_host.ps1 -Python python -V4 -Out "$PWD\tfcore_v4.dll"
.\build_node_host.ps1 -Python python
$r = (Resolve-Path ..\..).Path
$env:PYTHONPATH = "$r\server;$r\simulation\research\hardware_twin_scaling_20260929;$r\simulation\research\healthy_only_causal_20260908"
python -B test_v4.py
python -B test_core.py --report REPORT.json
```

The `PYTHONPATH` line is needed because these two test files look for the Python reference code at the authors' original folder layout (see "Path assumptions"). In a clean checkout `test_v4.py` runs all 32 tests; five of the six tests in `test_core.py` pass, and the sixth (real-data features and training) needs the data caches.

Server tests (the suite needs the original layout described under "Path assumptions" for tests that load the research modules):

```powershell
cd server
python -m pytest tests -q
```

### 5. Run the simulation

The study runners live in `simulation/research/<study>/`. Each is a plain script, for example:

```powershell
cd simulation/research/fl_algorithm_comparison_20260929
python -m pytest -q                       # unit tests, no data needed
python run_study.py                       # needs the feature caches, see below
```

The unit tests of `fl_algorithm_comparison_20260929` and `healthy_only_causal_20260908` run without data; the studies themselves need the cached feature files described in the next section.

## Datasets and data caches

The evaluation uses nine public bearing datasets: **PRONOSTIA**, **XJTU-SY**, **DLR**, **Ferrara**, **IMS**, **UNSW**, **KAIST**, **HIT** and **Paderborn**. **None of them is redistributed here.** Obtain each from its original publisher under its own terms. Derived feature caches, SD-card images, run directories, and results are also not included.

The data-preparation step that turns the raw archives into the cached feature files (`simulation/results/healthy_only_suite_20260907/cache/` and related folders) is not part of this release, so the simulation studies and `analysis/compare_hw_twin_v4.py` on a real run cannot be rerun end to end from this repository alone. The code is provided for inspection, for the parts that need no caches (FL algorithms, twin, protocol, server, firmware, parity and unit tests), and as the reference for the numbers below.

## Results summary

Verified numbers from the paper:

- 60 federated rounds on 10 boards at loads of 4 to 20 machines.
- 619 of 672 expected updates delivered (92.1%).
- Multicast used 58 to 78% fewer downlink fragments than a loss-free unicast bound.
- All 65 GLOBAL int8 code vectors broadcast during the runs were identical to the software twin's.
- In chronological replay, the complete federated system reached a 90.5% mean strict success rate (45, 44 and 44 of 49 over three seeds).
- Without retraining, the frozen models met the strict criterion on 6 of 7 unseen field machines in every seed; all five healthy wind turbines stayed silent, and a gearbox failure was detected about 170 days before replacement.
- One on-device local training costs 0.05 to 0.13 J.

## Path assumptions (read before running tests or analysis)

The research code was developed in a working tree with a different folder layout, and some files compute paths relative to it. This release keeps the original *relative* layout where code depends on it: `simulation/` corresponds to `full_pipeline_simulation/`, and the study folders sit in `simulation/research/`; `python_tools/` is a sibling of `simulation/`. Because code was not edited for the release, note the following.

- **`server/config.toml`**: set `research_root = "../simulation/research"` (already so in `config.toml.example`). If left empty, `server/study_plan.py` falls back to a hard-coded location in the original tree (`DEFAULT_RESEARCH_ROOT`), and tests that load the research modules through that default will not find them.
- **Server tests** that load the research modules use that default. To run the full suite unmodified, recreate the original layout around the checkout (a directory junction or symlink from `<parent>/autoencoder_cwru_release/full_pipeline_simulation` to `simulation/`, and from `<parent>/lorawan_two_node_experiment` to the repository root). With that layout 511 tests passed in our check; two documentation-presence tests in `tests/test_plan_config_v4.py` fail because `run_server_v4.ps1` and an operator-guide README section are not part of this release, and ten tests are skipped.
- **`firmware/host_tests/test_core.py`, `test_v4.py`** and **`analysis/compare_hw_twin_v4.py`** build paths such as `<root>/autoencoder_cwru_release/full_pipeline_simulation/research` from their own location. Putting `server/`, `simulation/research/hardware_twin_scaling_20260929` and `simulation/research/healthy_only_causal_20260908` on `PYTHONPATH` (as in the quick start) resolves their imports without editing them.
- Several simulation modules and protocol files contain data-folder paths of the original machine (for example `E:\masters\datasets`); set up your own data directory accordingly.
- Some documentation files (`firmware/tinyfl_fl_node/README_V4.md`, comments in `power/`) refer to the original folder names. `README_V4.md` also lists the GLOBAL size as 590 B; the current code and `docs/PROTOCOL_V4.md` (rev 4.3) use 624 B.

## Security notes

No credentials of any kind are in this repository. Real keys, `lab_credentials.h`, `config.toml`, `secrets*`, rosters, join-nonce stores, logs and run directories are excluded by `.gitignore`. Keys reach the code only through environment variables or hidden prompts; identifiers in the tests and in `config.toml.example` are synthetic placeholders. Never commit your own keys, EUIs, DevAddr or API keys.

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{darwish2026federated,
  author  = {Darwish, Abdelrahman and Elshabrawy, Tallal},
  title   = {Federated {TinyML} over {LoRaWAN}: Label-Free Failure Detection on Microcontrollers},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {submitted}
}
```

## Authors

- **Abdelrahman Darwish**, Faculty of Information Engineering and Technology, German University in Cairo, Cairo, Egypt. ORCID [0009-0007-6179-4974](https://orcid.org/0009-0007-6179-4974)
- **Tallal Elshabrawy**, Faculty of Information Engineering and Technology, German University in Cairo, Cairo, Egypt. ORCID [0000-0003-2270-0713](https://orcid.org/0000-0003-2270-0713)

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Abdelrahman Darwish and Tallal Elshabrawy. The public datasets and third-party libraries (MCCI LMIC, ESP32 Arduino core, SdFat, Adafruit libraries) keep their own licenses.
