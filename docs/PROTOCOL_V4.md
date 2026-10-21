# TinyFL — protocol v4 (general, production contract), rev 4.3

Rev 4.3 (2026-10-02), from the virtual end-to-end testbed (real firmware × real
server, `testbed/`): (a) the GLOBAL carries the phase's **participation mask**, so a
board trains, uploads and replays only its positions that take part (rev 4.2 boards
uploaded every hosted position and the server discarded the extras, which tripled
early-phase round time); GLOBAL is now 624 B (3.2). (b) REPLAY_DONE carries the first
alarm's acquisition index, so a lost ALARM no longer loses the alarm time (3.3).
(c) A board repeats HELLO until it hears its run (3.3). (d) END is repeated until the
boards go quiet, because a board inside its Class A windows misses a multicast (3.4).
(e) The server re-multicasts a GLOBAL promptly when a board that missed it reports
CLASS_C_READY (4). Frame layout, UPDATE and every other message are unchanged.

Rev 4.2 (2026-10-01), from the server implementation: provisioning CRC is over 60
bytes and every key's byte order is fixed (6.1); slots start at 1; CLASS_C_READY's
`awaited_slot` is held_slot + 1 (3.3). The server (`server/*_v4.py`, 447 tests)
implements rev 4.2.

Rev 4.1 (2026-10-01): mid-phase join is defined (5.1) with RUN_CONTROL ADOPT;
REPLAY_DONE is per (position, phase) so all 255 phases are confirmable; the
UPDATE header slot is the round's seed slot (section 1 corrected); the
`replay_masks.json` / `global_models.jsonl` schema is normative (4.1); board.txt
details and the no-replica limit are stated.

Supersedes v3 rev 3.2 (`PROTOCOL_V3.md`, kept for history). Single source of truth
for the **firmware**, the **server**, the **SD-card exporter** and the
**provisioning tool**. All multi-byte values little-endian, packed, no padding.
`f32` = IEEE-754 binary32. CRC32 = zlib CRC32 (init 0, unsigned, LE trailer).

Design goal: one firmware binary and one server for **any number of boards**
(1 … 65 534) and machines (0 … 65 534), boards joining and leaving at round
boundaries, and the thesis' fixed 4→28 sweep as one *planned* configuration of the
same system. The numerical pipeline (features, standardisation, H12 model, local
training, int8 codec, 13-of-21 alarm layer, FedAdam via `twin_server.py`) is
unchanged from v3 and stays bit-compatible with the twin.

## 1. Concepts

- **Board**: one LoRaWAN end device (DevEUI from the efuse MAC). Identity of its
  *machines* comes from its SD card (`board.txt`), not from the firmware.
- **Machine / position**: u16 global machine slot of the run, `0 … 65534`
  (`0xFFFF` = none). A board hosts 1 … `max_machines_per_board` (≤ 8) positions.
- **Run**: one server session, `run_id` u32 (random, non-zero), persisted by server
  and boards.
- **Phase**: a sequence of rounds trained from one initial model `fl.init(seed)`.
  The thesis sweep has 7 phases (levels 4…28); an elastic run usually has 1.
  `phase_index` u8, 0 … 254.
- **Slot** (`round_id`, u16): a run-wide monotonic counter. Every GLOBAL gets the
  next slot; an UPDATE fragment's header carries the slot of the GLOBAL that opened
  its round (the round's seed, see 3.1). Never reused within a run; no aliasing.

## 2. Frame (8-byte header, rev 4)

```
0  u8  magic 0xF1        4  u8  frag_index
1  u8  version 0x04      5  u8  frag_count
2  u16 round_id (slot)   6  u8  body_len  (≤ 213)
                         7  u8  type
```
Max application payload 221 B (DR5). Receivers validate magic, version, type,
`frag_index < frag_count`, `frag_count` (3 for UPDATE/GLOBAL fragments, 1 for
control frames) and `body_len` before storing anything. Control frames
(`frag_count = 1`) carry no logical CRC (LoRaWAN MIC protects them).
UPDATE/GLOBAL logical messages end with CRC32 over all preceding logical bytes.

## 3. Messages

| type | dir | name | form |
|---|---|---|---|
| 0x01 | up | UPDATE_FRAGMENT | body = u16 position + ≤ 211 B of UPDATE |
| 0x02 | up | CLASS_C_READY | control, 3.3 |
| 0x03 | down (multicast) | GLOBAL_FRAGMENT | body = ≤ 213 B of GLOBAL |
| 0x04 | up | GLOBAL_STATUS | control, 3.3 |
| 0x05 | down (unicast) | UPDATE_REPAIR | control, 3.3 |
| 0x06 | up | BOARD_HELLO | control, 3.3 |
| 0x07 | up | ALARM | control, 3.3 |
| 0x08 | up | SELF_CHECK | control, 3.3 |
| 0x09 | up | REPLAY_DONE | control, 3.3 |
| 0x0A | down (multicast) | RUN_CONTROL | control, 3.4 |

### 3.1 UPDATE (598 B → 3 fragments: 211/211/176 payload, frames 221/221/186)

```
off type name
0   u16  position
2   u32  run_id
6   u8   phase_index
7   u8   round            1..R: the training round this update completes
8   u16  base_slot        slot of the complete, CRC-valid GLOBAL actually trained from
10  u16  n_train_rows     saturates at 65535
12  f32  scale[4]         per-tensor (W1, b1, W2, b2), finite, > 0
28  i8   delta[562]       int8 codes of (local_model - base_global);
                          W1[i][j] at i*12+j (i input 0..21, j hidden 0..11),
                          b1[12], W2[j][k] at j*22+k (k output 0..21), b2[22]
590 f32  train_loss       mean reconstruction loss after local training
594 u32  crc32            over bytes 0..593
```
Each UPDATE_FRAGMENT body is `u16 position` followed by the next ≤ 211 bytes of
the UPDATE. The fragment header `round_id` is the slot of the GLOBAL that opened
the round (the round's seed); the server uses it for routing only. `base_slot`
inside the CRC-covered payload is authoritative for the training base.

Quantisation: values cast to float32; `scale = max(max|x|/127, 1e-7)`;
`q = roundf(x/scale)` half away from zero; clip [-127, 127]; -128 invalid.

**Stale bases.** A board trains only from a complete CRC-valid GLOBAL of the
same run and phase. If it holds an older one than the round's seed, it trains
from it and reports it in `base_slot`; the server reconstructs
`client = global[base_slot] + delta` and aggregates `client - current_global`
(twin stale-copy semantics). Every GLOBAL of the phase is kept by the server.

### 3.2 GLOBAL (rev 4.3: 624 B → 3 fragments: 213/213/198, frames 221/221/206)

```
0   u32  run_id
4   u8   phase_index
5   u8   round            0 = initial model fl.init(seed); r = after round r
6   u8   rounds_in_phase  R (12 for the thesis sweep)
7   u8   flags            bit0 = final model of the phase (round == R)
                          bit1 = every position takes part (mask ignored; elastic)
8   f32  scale[4]
24  i8   weights[562]     absolute weights, tensor order as 3.1
586 u16  mask_base        first position covered by the mask
588 u8   mask[32]         bit i of byte j set ⇔ position mask_base + 8j + i takes
                          part in this phase (positions outside the window do not)
620 u32  crc32
```
**Participation (rev 4.3).** A board's *active* positions for a phase are its hosted
positions that take part per the GLOBAL (bit1, or the mask). The mask is identical
in every GLOBAL of a phase. A board trains, uploads (3.1) and replays (3.4) only its
active positions; a board with none stays in Class C idle for that phase (it keeps
accepting GLOBALs and its periodic CLASS_C_READY, so it stays live for the next
phase). A planned phase's position set must fit one 256-position window (the server
rejects other configs); elastic phases set bit1.
The header `round_id` is this GLOBAL's slot. A board accepts any GLOBAL of its
run whose slot is newer than the one it holds (it may have missed some). A
GLOBAL whose `run_id` differs from the stored run is accepted **only** if
`round == 0` (a phase start); then the board archives and resets its run state
(see 5). Final models (flag bit0) are stored in flash per phase.

### 3.3 Control bodies (exact)

| type | body | bytes |
|---|---|---|
| CLASS_C_READY 0x02 | u32 run_id, u16 awaited_slot | 6 |
| GLOBAL_STATUS 0x04 | u32 run_id, u16 slot, u8 missing_mask (bits 0..2) | 7 |
| UPDATE_REPAIR 0x05 | u32 run_id, u16 position, u8 missing_mask (bits 0..2) | 7 |
| BOARD_HELLO 0x06 | u16 board_id, u8 fw_version, u8 flags, u32 stored_run_id (0 = none), u8 n, u16 positions[n] (n ≤ 8) | 9 + 2n |
| ALARM 0x07 | u32 run_id, u16 position, u8 phase_index, u32 acquisition_index, f32 score, f32 threshold | 19 |
| SELF_CHECK 0x08 | u16 position, u16 rows_checked, f32 max_abs_diff, f32 median_abs_diff | 12 |
| REPLAY_DONE 0x09 | u32 run_id, u16 position, u8 phase_index, u32 acquisitions_scored, u8 alarm_latched (0/1), u32 first_alarm_index (0xFFFFFFFF = none; rev 4.3) | 16 |

`round_id` header field: CLASS_C_READY and GLOBAL_STATUS carry the slot they
refer to; UPDATE_REPAIR the slot of the round; all others `0xFFFF`.
**Slots start at 1** (slot 0 never exists; 0 means "none held").
**CLASS_C_READY `awaited_slot` (rev 4.2)**: after HELLO and before holding any GLOBAL
of the run: `0xFFFF` (any phase start). Otherwise always **held_slot + 1**, the
next GLOBAL the board waits for, sent after the board's last UPDATE fragment of a
round, after answering repairs, and after storing a phase's final model. The
server reads `awaited_slot == seed_slot + 1` as "this board's upload or repair is
finished".
**ALARM vs REPLAY_DONE (rev 4.3)**: ALARM is sent once, unacknowledged. REPLAY_DONE
(repeated on every repeated REPLAY) is authoritative: `alarm_latched = 1` ⇔
`first_alarm_index != 0xFFFFFFFF`, and that index equals the ALARM's
`acquisition_index`. A missing ALARM loses only its diagnostic score/threshold.
**HELLO repetition (rev 4.3)**: HELLO is unacknowledged. After boot a board sends
HELLO instead of each periodic CLASS_C_READY until it has received, since boot, a
CRC-valid GLOBAL or a RUN_CONTROL of the run it then holds. The server treats a
repeated HELLO with the same DevEUI, board_id, positions and stored run as a no-op
(liveness only); it never resets state.
HELLO flags: bit0 SD mounted, bit1 self-check passed, bit2 IMU source, bit3 run
complete (replay done for the stored run). GLOBAL_STATUS `missing_mask` = 0 means
the board holds the complete GLOBAL.

### 3.4 RUN_CONTROL 0x0A (multicast, single frame)

`u32 run_id, u8 command, u8 n_phases, u8 phases[n_phases]`
- `command 3 = ADOPT` (n_phases = 0): a board whose stored run differs from
  `run_id` archives and clears its run state and adopts `run_id` (5.1). The server
  multicasts ADOPT at run start, **every `adopt_period_seconds` (default 120) while
  WAITING_START** (rev 4.3: boards completed in a previous run are silent in Class C
  and only answer an ADOPT, with HELLO), again whenever a HELLO reports a
  different non-zero `stored_run_id`, and (rev 4.3) whenever a rostered DevEUI that
  is not BOUND sends any uplink (its HELLO was lost; rate-limited to one ADOPT per
  `adopt_period_seconds`).
  **Rev 4.3: every received ADOPT makes the board send one HELLO** (after adopting
  if the run differs; also when it already holds that run) at its next transmit
  opportunity, unless its stored run equals `run_id` and is complete. This is how a
  board learns the server has not bound it.
- `command 1 = REPLAY`: every board that holds the final model of each listed
  phase replays its machines' full history scoring with those models, sends ALARM
  (once per position × phase) and one REPLAY_DONE per (position, phase) it scored.
  Repeated by the server
  until every expected board has reported REPLAY_DONE or `post_run_collect_seconds`.
- `command 2 = END`: run finished; boards stay in Class A idle, keep results, and
  send nothing more for this run. **Delivery (rev 4.3)**: a board misses any
  multicast while transmitting or inside its RX1/RX2 windows, and END has no
  acknowledgement, so the server (ENDING stage) multicasts END no earlier than
  `downlink_guard_seconds` (default 4) after the last uplink it heard, and again
  whenever any board of the run sends an uplink after an END (at most once per
  `end_repeat_seconds`, default 30). Silence is the acknowledgement: the run is DONE
  after `end_quiet_seconds` (default 420, > 2 × the boards' 180 s idle READY period)
  with no uplink since the last END, or at `end_linger_seconds` (default 1800);
  boards still talking then are listed as `end_unconfirmed` in `summary.json`.
Boards ignore RUN_CONTROL for another run_id. **There is no compile-time level
mask in v4**: which phases exist and which are replayed is decided by the server.

## 4. Server behaviour (normative)

- **Run plan** (config): `mode = "elastic"` or `"planned"`.
  - *planned*: a list of phases, each with `positions` (explicit set or "first N")
    and `rounds` (default 12). The thesis sweep = 7 phases, first 4, 8, …, 28.
  - *elastic*: one (or more) phases of R rounds; participants = whoever is ACTIVE.
- **Board membership** state machine, per DevEUI:
  `ROSTERED → BOUND (first valid HELLO; board_id ↔ DevEUI, positions recorded) →
  ACTIVE (acknowledged the current phase's seed GLOBAL) → SUSPECT (missed
  `suspect_after_rounds` rounds) → DEPARTED (missed `depart_after_rounds`)`.
  A DEPARTED board that speaks again re-enters BOUND and is re-admitted at the next
  round boundary. A HELLO whose board_id is bound to another DevEUI, or whose
  positions collide with another board's, is rejected and logged.
- **Start**: when `min_boards` (default 1) boards are BOUND with HELLO bits 0 and 1
  (or bit 2), or at `start_after_seconds` if ≥ 1 is; *planned* mode additionally
  reports which planned positions have no board (`require_all_positions`, default
  false: absent positions are logged as never participating, not fatal).
- **Round**: participants = ACTIVE boards hosting positions of the phase. The round
  closes when every participant position delivered, was abandoned after
  `max_repair_rounds`, or at `round_timeout_seconds`. Late boards join at the next
  round. A phase with zero participants pauses until one appears or
  `idle_phase_timeout_seconds`.
- **Logs** (append-only, flushed per record): delivery log per round (participants,
  arrived positions, base_slot per position, stale positions, joins/leaves,
  fragments, repairs, timings), `global_models.jsonl` (codes + scales + CRC),
  alarms, self-checks, replay-done, membership events, and `replay_masks.json` for
  the twin (per phase: `uplink[round][position]`, `downlink[round][position]`,
  `base_slot[round][position]`, positions = every position seen in the phase).
- **Prompt GLOBAL repair (rev 4.3)**: in DOWNLINK, when a required board that does
  not hold the GLOBAL reports CLASS_C_READY for it (or GLOBAL_STATUS with missing
  bits), the server re-multicasts the missing fragments `downlink_guard_seconds`
  after that uplink, subject only to the gateway duty cycle, instead of waiting for
  the periodic repair timer. Never multicast to a board inside its RX windows.
  Prompt repairs are at most one per board per `prompt_repair_min_gap_seconds`
  (default 30), count toward `max_global_repair_rounds`, and never postpone the
  periodic repair timer (rev 4.3, testbed: STATUS storms otherwise starved it).
- **Unicast queue lifecycle (rev 4.3)**: the network server keeps unicast downlinks
  until the device's next uplink. The server therefore sends UPDATE_REPAIR with
  `down/replace` (never `down/push`), so at most one repair is pending per device,
  and clears (`down/replace` with an empty list) every device that still has a
  pending repair when its round closes. `down/sent`/`down/failed` events are matched
  to the request by correlation id, never by device alone.
- **Mid-phase join**: a board that becomes BOUND while a phase is running is
  ACTIVE from the first round whose seed GLOBAL it reports holding (GLOBAL_STATUS
  missing_mask 0 for that slot, or an UPDATE with that `base_slot`). Before that
  its positions are logged as absent (uplink and downlink false).
- **Crash recovery**: the server checkpoints run state (run_id, plan, phase, round,
  slot counter, FedAdam state, every GLOBAL of the current phase, membership) to
  disk atomically after every round; `--resume <run_dir>` continues the same run.
- **Secrets** only from environment variables; no key is ever logged.

### 4.1 Run-directory schema for the twin (normative)

`replay_masks.json`: `{"schema": "tinyfl-replay-masks-v4", "run_id", "seed",
"mode": "planned"|"elastic", "phases": [ {"phase_index", "rounds_in_phase",
"rounds_completed", "complete", "positions": [u16 …] (every position seen in the
phase; matrix columns follow this order), "planned_positions" (planned mode),
"uplink": [[bool]], "downlink": [[bool]], "base_slot": [[u16|null]]} ]}` where
row k is training round k+1 and `base_slot` is null exactly where uplink is false.
`global_models.jsonl`: one line per GLOBAL: `run_id, phase_index, round, slot,
codes[562], scales_f32[4], crc32`. Slots strictly increase across the run.
`delivery_log.jsonl`: one `{"record": "round", "phase_index", "round", …}` per
closed round with `participants, arrived_positions, stale_positions,
joined_positions, left_positions` and the fragment/repair counters
(`update_fragments_received, update_repair_requests, update_repair_downlinks_sent,
global_fragments_sent, global_repair_rounds, global_bursts`).
`alarms.csv`: `run_id, position, phase_index, acquisition_index, score,
threshold, source` (`alarm`, or `replay_done` when only REPLAY_DONE carried it:
score and threshold empty). `replay_done.csv`: `run_id, position, phase_index,
acquisitions_scored, alarm_latched, first_alarm_index`. Per phase,
`replay_masks.json` adds `"participation": {"all": bool, "mask_base", "positions"}`
as multicast in that phase's GLOBALs.
`analysis/compare_hw_twin_v4.py --fields` prints the same list.

## 5. Board behaviour (normative)

- Identity: DevEUI from efuse MAC; AppKey **per device** from NVS (written by the
  provisioning tool, 6); refuses to join without one.
- Boot: mount SD → validate `board.txt`/data files → SELF_CHECK all machines →
  OTAA join → HELLO → Class C READY for slot 0xFFFF ("any phase start").
- On a GLOBAL of its run: its active positions are those the GLOBAL marks as taking
  part (3.2, rev 4.3); if `round == 0` train round 1, else train round `round + 1`
  unless `flags.final`. Send one UPDATE per **active** position, fragments back to
  back, positions ascending. Store the participation with each phase's final model;
  REPLAY scores, ALARMs and REPLAY_DONEs only the positions active in that phase.
- **Radio scheduling (rev 4.3)**: Class C is switched off only when an uplink can
  start immediately. A board never hands LMIC an uplink that must wait for the duty
  cycle (it keeps its own 1% accounting from the frames' airtime, or queries LMIC's
  band availability) — a board waiting for the duty cycle stays in Class C. A
  GLOBAL_STATUS is itself the board's "listening" report: after it the board
  returns to Class C waiting without sending an extra READY or HELLO.
- HELLO is repeated per 3.3 until the run is heard. After END: no further uplink for
  that run (a new ADOPT or phase-start GLOBAL of another run starts a new one).
  **Exception, after a reboot** of a board whose stored run is complete: it sends
  HELLO (bit3 set) and its SELF_CHECKs, then repeats that HELLO every 180 s until it
  hears any RUN_CONTROL or GLOBAL (of its run or a new one), so a new server run can
  find and ADOPT it. Without a reboot a completed board stays silent in Class C.
- Persist (two-generation CRC-protected files): run_id, phase/round, held GLOBAL,
  final models per phase, outbox, replay progress. A reboot resumes.
- **Adopting a run (5.1)**: a board adopts a run (archive `results.csv`, clear run
  state, store the new `run_id`) on (a) RUN_CONTROL ADOPT, (b) any GLOBAL of
  `round == 0`, or (c) any CRC-valid GLOBAL when it holds no run
  (`stored_run_id == 0`) or its stored run is complete (HELLO bit3). Nothing else
  erases run state.
- **Mid-phase join**: a board of the current run that holds no GLOBAL of the current
  phase activates its active positions on the first complete GLOBAL of the phase
  it receives (any round r), and trains round r+1 from it (`base_slot` = its slot).
  It never trains from a partial GLOBAL.
- On RUN_CONTROL REPLAY: replay as in v3 (one pass, every listed phase model with
  its own alarm state, `results.csv` rows `position,index,phase,score,flag`).
- Watchdog enabled; brown-out reset safe; OLED shows board id, phase, round,
  activity, progress, alarms; serial `PHASE <name> <millis>` markers as v3.

## 6. Provisioning and enrollment (production)

- **Per-device keys**: `AppKey(DevEUI) = AES-128-ECB(master_key, DevEUI ‖ 0x00×8)`.
  The master key never leaves the operator PC.
- **Provisioning tool** (PC, serial): flashes/uses `tinyfl_provision`, which formats
  LittleFS (NVS kept), reports DevEUI over serial, receives its derived AppKey (and
  the shared multicast session keys) over serial, stores them in NVS, and confirms by
  CRC. The tool appends the DevEUI to the **roster** (`server/roster.json`, no keys).
- **Enrollment**: the server registers a DevEUI only if it is in the roster, the
  window is open, the JoinRequest is exactly 23 bytes with MType JoinRequest, the
  JoinEUI matches, and the MIC (AES-CMAC over the first 19 bytes, constant-time
  compare) verifies under that DevEUI's derived key. Seen (DevEUI, DevNonce) pairs
  are persisted and replays rejected. Registration is idempotent with rollback of
  partial JS/NS/AS creation. The roster, not the quota, is the authorisation.

### 6.1 Serial provisioning exchange (115200 8N1, ASCII lines ending "\n")

1. Board (running `tinyfl_provision`) formats LittleFS, then prints
   `TFPROV HELLO <fw_version> <deveui 16 hex, MSB first>` once per second until
   step 2 arrives (up to 120 s).
2. PC sends `TFPROV KEYS <joineui 16 hex MSB first> <appkey 32 hex> <mc_devaddr 8 hex>
   <mc_nwkskey 32 hex> <mc_appskey 32 hex>`.
3. Board stores the five values in NVS namespace `tinyflkeys` (keys `joineui`,
   `appkey`, `mcaddr`, `mcnwk`, `mcapp`) as raw bytes **in exactly the order the hex
   was written** (no reversal), reads them back and prints
   `TFPROV OK <crc32 as 8 hex digits of the 8+16+4+16+16 = 60 raw bytes in that
   order>`, or `TFPROV ERR <reason>`. The firmware converts on use: LMIC wants the
   JoinEUI and DevEUI least-significant byte first, so it reverses `joineui` when
   loading; `mcaddr` bytes are the multicast DevAddr most-significant byte first
   (u32 value = `mcaddr[0]<<24 | … | mcaddr[3]`); the AppKey and session keys are
   used as stored. `AppKey(DevEUI)` derivation uses the DevEUI as printed in
   TFPROV HELLO (most-significant byte first) followed by 8 zero bytes. The PC verifies the CRC, then appends the DevEUI to
   `server/roster.json` (`{"devices": [{"dev_eui": ..., "provisioned_utc": ...}]}`).
4. The board never prints key material. The multicast frame counter namespace
   `tinyflmc` is not touched. `tinyfl_fl_node` reads `tinyflkeys`; if absent it
   shows `NOT PROVISIONED` and does not join (no compile-time key fallback).

## 7. SD card (v4)

`board.txt` (text, LF): `board_id <u16>` then `format_version 2` then one line per
machine `position <u16> machine_id <id> streams <n>`. `board_id` 1 … 65534;
file-name hex digits lowercase. Replica machines (section 8) are not supported in
v4.0: with more boards than machines, the extra boards get no card. Data files are byte-identical
to v3 section 5 (`TFR1` raw int16, `TFC1` self-check; both already carry a u16
position). File names `m<PPPP>s<S>.raw`, `m<PPPP>.chk`
(PPPP = 4 hex digits of the position, so 8.3-safe up to 65 534).
`manifest.json` → renamed `manifest.jsn` (8.3). No `levels.csv`: phases come from
the server.

## 8. Limits (stated, not hidden)

Protocol: 65 534 boards and positions, 255 phases, 65 535 slots per run, ≤ 8
machines per board; a *planned* phase's position set must fit one 256-position
window (rev 4.3; elastic phases are unlimited). Radio: per-device 1% duty cycle keeps round time flat as
boards are added; uplink collisions at one gateway grow with the number of
boards (pure ALOHA), multicast downlink stays 3 fragments per round. Data: the
thesis cohort has 54 machines (41 fit 512 MB cards); more boards than that would
carry replica machines, which add communication evidence only.
