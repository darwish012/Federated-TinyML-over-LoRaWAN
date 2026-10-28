# Frozen unseen-machine evaluation

Complete: True; protocol v3, Amendment 2.

Amendment 2 was issued after the pre-registered onset estimator found no sustained departure for Turbine 5 (event 1) and stopped the run, and before any Turbine 5 detector score was computed or any control result was viewed. It adopts the data authors' documented defect onset (beginning of year one, 1.0 year) for event 1, following the rule that documented onsets take precedence over the estimated proxy.

| Seed | Arm | Strict successes /6 | T5 event-1 fault detected | Event-1 lead (days) | First alarm in fault period | Controls silent /5 | Evaluable /6 | Processed /6 |
|---:|---|---:|---|---:|---|---:|---:|---:|
| 17070 | A: RMS > 0.5 g | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17070 | B: variable-speed operating window | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17071 | A: RMS > 0.5 g | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17071 | B: variable-speed operating window | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17072 | A: RMS > 0.5 g | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17072 | B: variable-speed operating window | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |

| Secondary unit | Seed | Arm | Status / applicability | Strict success | Healthy alarm | Fault detected | First alarm (years / record-day) | Lead (days / record-days) | First alarm in fault period | Before day 50 |
|---|---:|---|---|---|---|---|---:|---:|---|---|
| T5 event 2: planet bearing / gearbox | 17070 | A: RMS > 0.5 g | evaluable | True | False | True | 1.523988818692 | 173.86308397274698 | True | not applicable |
| T5 event 2: planet bearing / gearbox | 17070 | B: variable-speed operating window | evaluable | True | False | True | 1.53904657838945 | 168.36323724325337 | True | not applicable |
| T5 event 2: planet bearing / gearbox | 17071 | A: RMS > 0.5 g | evaluable | True | False | True | 1.523988818692 | 173.86308397274698 | True | not applicable |
| T5 event 2: planet bearing / gearbox | 17071 | B: variable-speed operating window | evaluable | True | False | True | 1.53767635733641 | 168.86371048287623 | True | not applicable |
| T5 event 2: planet bearing / gearbox | 17072 | A: RMS > 0.5 g | evaluable | True | False | True | 1.52672542646071 | 172.8635379852257 | True | not applicable |
| T5 event 2: planet bearing / gearbox | 17072 | B: variable-speed operating window | evaluable | True | False | True | 1.53904657838945 | 168.36323724325337 | True | not applicable |
| HSS | 17070 | A: RMS > 0.5 g | applicable | not applicable | null | null | 49 | 1 | not defined | True |
| HSS | 17070 | B: variable-speed operating window | not applicable (no speed column) | not applicable | null | null | 49 | 1 | not defined | True |
| HSS | 17071 | A: RMS > 0.5 g | applicable | not applicable | null | null | 47 | 3 | not defined | True |
| HSS | 17071 | B: variable-speed operating window | not applicable (no speed column) | not applicable | null | null | 47 | 3 | not defined | True |
| HSS | 17072 | A: RMS > 0.5 g | applicable | not applicable | null | null | 45 | 5 | not defined | True |
| HSS | 17072 | B: variable-speed operating window | not applicable (no speed column) | not applicable | null | null | 45 | 5 | not defined | True |

Event 2 begins with a new output-shaft bearing. The accelerometer is on the output-shaft bearing housing, not on the failed planet bearing; event 2 is a harder, indirect test. It is separate from the six-turbine primary denominator. If its proxy has no onset, both arms are not evaluable (no proxy onset), with null outcomes/timing; remaining units continue.

HSS Arm B: not applicable (no speed column); its results are identical to Arm A. Tach-derived speed is prohibited. HSS has no defined healthy/fault partition; first-alarm membership in a fault period is not applicable.

## No-filter sensitivity

| Seed | Arm | Strict successes /6 | T5 event-1 fault detected | Event-1 lead (days) | First alarm in fault period | Controls silent /5 | Evaluable /6 | Processed /6 |
|---:|---|---:|---|---:|---|---:|---:|---:|
| 17070 | No-filter sensitivity | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17071 | No-filter sensitivity | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |
| 17072 | No-filter sensitivity | 5/6 | False | null | False | 5/5 | 6/6 | 6/6 |

Pooled counts are repeated machine-seed evaluations, not independent machines:

- A: RMS > 0.5 g: 15/18 strict successes; 18/18 evaluable; 18/18 processed.
- B: variable-speed operating window: 15/18 strict successes; 18/18 evaluable; 18/18 processed.
- No-filter sensitivity: 15/18 strict successes; 18/18 evaluable; 18/18 processed.

## Partition and onset audit

Documented onsets take precedence for every fault unit/event. Event 1 uses t* = 1.0 year for A, B and no-filter sensitivity; event 2 uses the unchanged historical proxy. Arm B inherits Arm A onset and partition boundaries. Its scaler and threshold use kept commissioning rows; persistence advances over kept rows only. No onset or partition is re-estimated for B.

| Unit | Arm | Status | Onset (years) | Speed P25 / P75 (cycles/min) | Train kept/base | Calibration kept/base | Healthy test kept/base | Fault kept/base | Reason |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| Turbine 1 | A: RMS > 0.5 g | evaluable | None | not applicable | 1246/1246 | 416/416 | 416/416 | 0/0 |  |
| Turbine 1 | B: variable-speed operating window | evaluable | None | 888.125525 / 1140.79225 | 622/1246 | 208/416 | 191/416 | 0/0 |  |
| Turbine 1 | No-filter sensitivity | evaluable | None | not applicable | 1789/1789 | 596/596 | 597/597 | 0/0 |  |
| Turbine 2 | A: RMS > 0.5 g | evaluable | None | not applicable | 1234/1234 | 412/412 | 412/412 | 0/0 |  |
| Turbine 2 | B: variable-speed operating window | evaluable | None | 889.346125 / 1132.5065 | 616/1234 | 182/412 | 189/412 | 0/0 |  |
| Turbine 2 | No-filter sensitivity | evaluable | None | not applicable | 1803/1803 | 601/601 | 601/601 | 0/0 |  |
| Turbine 3 | A: RMS > 0.5 g | evaluable | None | not applicable | 681/681 | 227/227 | 227/227 | 0/0 |  |
| Turbine 3 | B: variable-speed operating window | evaluable | None | 942.3956 / 1121.65 | 341/681 | 101/227 | 104/227 | 0/0 |  |
| Turbine 3 | No-filter sensitivity | evaluable | None | not applicable | 1602/1602 | 534/534 | 534/534 | 0/0 |  |
| Turbine 4 | A: RMS > 0.5 g | evaluable | None | not applicable | 973/973 | 325/325 | 325/325 | 0/0 |  |
| Turbine 4 | B: variable-speed operating window | evaluable | None | 842.4258 / 1112.132 | 487/973 | 172/325 | 147/325 | 0/0 |  |
| Turbine 4 | No-filter sensitivity | evaluable | None | not applicable | 1600/1600 | 533/533 | 534/534 | 0/0 |  |
| Turbine 5 event1 | A: RMS > 0.5 g | evaluable | 1.0 | not applicable | 271/271 | 91/91 | 91/91 | 94/94 |  |
| Turbine 5 event1 | B: variable-speed operating window | evaluable | 1.0 | 997.03495 / 1140.5455 | 135/271 | 40/91 | 44/91 | 38/94 |  |
| Turbine 5 event1 | No-filter sensitivity | evaluable | 1.0 | not applicable | 415/415 | 139/139 | 139/139 | 200/200 |  |
| Turbine 6 | A: RMS > 0.5 g | evaluable | None | not applicable | 977/977 | 326/326 | 326/326 | 0/0 |  |
| Turbine 6 | B: variable-speed operating window | evaluable | None | 968.8705 / 1134.406 | 489/977 | 148/326 | 161/326 | 0/0 |  |
| Turbine 6 | No-filter sensitivity | evaluable | None | not applicable | 1771/1771 | 591/591 | 591/591 | 0/0 |  |
| Turbine 5 event2 | A: RMS > 0.5 g | evaluable | 1.48702507770712 | not applicable | 90/90 | 30/30 | 31/31 | 262/262 |  |
| Turbine 5 event2 | B: variable-speed operating window | evaluable | 1.48702507770712 | 894.9439749999999 / 1110.0439999999999 | 44/90 | 8/30 | 10/31 | 82/262 |  |

Every turbine/seed outcome and first-alarm source row is in per_unit.csv. Source-row memberships, Arm A boundary rows, onset diagnostics and exclusion counts are in partition_onset_audit.json. Scores retain source rows, times and fixed partition labels.

Lead times include premature healthy-test alarms, which cause strict failure. SND lead is (nominal replacement year - first alarm year) × 365.25. No alarm gives null timing and false detection. HSS lead is 50 minus the first alarm record-day.

## Limitations

Filename-to-turbine mapping remains ASSUMED, not publisher-verified. The time-only check found no registered contradiction. Nominal replacement times of 1.2 and 2.0 years are approximate.

Turbine 2's documented early electrical sensor issue was retained; it is an electrical measurement issue, not a known bearing failure.

HSS days are ordered record-days, including both March 17 records. Filename timestamps are in hss_record_days.csv. Day 50 is a terminal proxy, not an exact failure timestamp; commissioning health is assumed, and this is one small case study.

Seed repetitions and six turbines in one farm are correlated. Event 1 uses documented onset at 1.0 year; event 2 uses the retrospective proxy if available. Strict success is conditional on these partitions. HSS has no strict score or fault partition. Not-evaluable units remain in registered denominators; their outcomes are null.

Event-1 labels use the authors' documented defect period with nominal t* = 1.0 year and assumed healthy commissioning before it; this is not a precise independent physical damage timestamp. Event-2 labels use the retrospective proxy when available. Earlier development selection bias remains. No unseen unit trains or updates the global models; no extra external-data tuning is authorized.

Source/model hashes are in protocol.json; environment and time-only evidence are in execution_environment.json and metadata_identity_check.json.
