# Pre-registered unseen-machine test: frozen FedAdam H12 — v3, Amendment 2

Protocol ID: `SND_HSS_FROZEN_H12_20261026`. **Version 3, Amendment 2**, issued **2026-10-26T14:35:55.506272+00:00** on 26 October 2026. V1 and v2 MD/JSON, seals and runners are preserved byte-identically as versioned archives. Only current `run_unseen.py` / `protocol.json` are stage-2 entry points. All v2 rules remain except the explicit Amendment-2 changes below.

Amendment 2 was issued after the pre-registered onset estimator found no sustained departure for Turbine 5 (event 1) and stopped the run, and before any Turbine 5 detector score was computed or any control result was viewed. It adopts the data authors' documented defect onset (beginning of year one, 1.0 year) for event 1, following the rule that documented onsets take precedence over the estimated proxy.

The supplied v2 stop history states 39 completed unit-seed rows for Turbines 1–4 and no Turbine-5 detector score. That administrative history was supplied by Opus/the author; no result was opened to prepare v3. This is a disclosed post-stop amendment, not a claim that v3 preceded all target analysis. V3 is frozen before its fresh rerun. `PROTOCOL_SHA256.txt` seals current and archived v1/v2 MD/JSON; current JSON seals source/models, runner, helpers and archives. No core source/model changed. Opus reviews, commits and delivers; Codex makes no commit.

**Delivery:** work stays in `C:\Users\Dell\AppData\Local\Temp\codex_unseen_stage1_20261026`; Opus runs the reviewed `DELIVER_TO_D.ps1`. It copies only new/changed allowlisted top-level artifacts, verifies hashes, preserves immutable archives and existing deviation history, and never copies into data or results directories. Existing `frozen/` artifacts are verified and retained. The runner refuses execution from staging and refuses any output destination except `D:\external_unseen\results_v3`. V2 `D:\external_unseen\results\` remains untouched as stop evidence.

## 1. Hypothesis, units and outcomes

The hypothesis is that the global H12 detector trained on the paper's 54 units transfers to unseen turbines after device-local commissioning: it remains silent in held-out assumed-healthy periods, detects Turbine 5's output-shaft bearing degradation before replacement, and leaves the five healthy controls silent in their held-out tests.

The **primary population is all six SND turbines** from one farm. For each seed 17070, 17071 and 17072:

- Turbine 5 strictly succeeds only if it has **no alarm anywhere in its held-out healthy test** and **at least one alarm in its assigned fault period**.
- Each control (Turbines 1, 2, 3, 4 and 6) strictly succeeds only if it **never alarms in its entire held-out test**, the last approximately 20% of its included record. Its first 80% is commissioning, not held-out evaluation. Controls do not have assigned fault periods.
- Report every turbine's healthy-alarm, fault-detection and strict-success outcome, the strict count out of six per seed, and the pooled count out of 18 turbine-seed evaluations. Preserve every failure. Eighteen evaluations do not represent 18 independent machines; there are six machines and one farm. No significance test or target accuracy gate is registered.
- For Turbine 5 event 1, report the first post-commissioning alarm time in source years, its source CSV row, `first_alarm_time - 1.2` years (negative means before replacement), `1.2 - first_alarm_time` years of lead time, and **lead days = (1.2 - first_alarm_time) × 365.25**, per seed and arm. Report whether the first alarm falls inside the assigned fault period, separately from any fault-period detection. Include a premature healthy-test alarm in this timing, even if it causes strict failure. No alarm gives null timing, false detection and false first-alarm fault membership; do not substitute the last record. Not-evaluable outcomes/timing remain null.

The **secondary outcomes are Turbine 5 event 2** (Section 3b) and the **single MathWorks HSS lifecycle**, separate from the six-turbine primary denominator. HSS is reported for every seed and A/B arm: first alarm record-day, **lead in record-days = 50 - first alarm day**, and whether the first alarm occurs strictly before day 50. An alarm only on day 50 is not advance warning. Failure is documented by the end of the record; day 50 is a terminal proxy, not an exact failure timestamp. There is **no HSS strict score** or assigned fault period; first-alarm fault-period membership is explicitly not defined / not applicable. HSS B is identical to A and labelled **not applicable (no speed column)**. Do not pool either secondary outcome with the SND primary denominator.

These endpoints test a fixed detector with local commissioning. They do not test federated adaptation on the new machines, nor establish independent physical damage onset.

## 2. Target sources, identity and chronology

SND source: [dataset 2024-248](https://researchdata.se/en/catalogue/dataset/2024-248), DOI [10.5878/bcmv-wq08](https://doi.org/10.5878/bcmv-wq08), and Martin-del-Campo, Sandin and Strömbergsson, *A dictionary learning approach to monitoring of wind turbine drivetrain bearings* ([author manuscript](https://arxiv.org/abs/1902.01426)). Publisher metadata describes axial acceleration on the output-shaft bearing housing. The operational component label used here is **output-shaft bearing**, following the dataset and the author's instructions; an HSS reference elsewhere in the manuscript is not used to relabel the SND component. The two external sources are separate case populations.

The following mapping is **an assumption registered before reading even the target time columns**, not a verified publisher filename key:

| File under `D:\external_unseen\snd_windfarm` | Assumed identity | Role |
|---|---|---|
| DATASET02.csv | Turbine 1 | Healthy control |
| DATASET03.csv | Turbine 2 | Healthy control; early electrical sensor issue retained |
| DATASET04.csv | Turbine 3 | Healthy control |
| DATASET05.csv | Turbine 4 | Healthy control |
| DATASET06.csv | Turbine 5 | Output-shaft bearing fault before replacement |
| DATASET07.csv | Turbine 6 | Healthy control |

Each numeric row is one acquisition: column 1 time in years since recording began, column 2 speed in cycles/minute, and columns 3–16386 the 16384 acceleration samples in g. Sampling rate is 12800 Hz. Speed is not a detector input.

**The first stage-2 operation on target numeric fields is a metadata-only identity/chronology check.** It decodes and converts only the first field of each row. It does not parse speed or vibration. For all six files it records row count, first/last and minimum/maximum time, median positive time step, recording gaps and any nonincreasing step. File I/O necessarily reads bytes containing the remaining fields, but those fields are not interpreted or exposed by the metadata operation.

The blind gap definition is a step strictly greater than `max(7/365.25 years, 10 × median positive step)`. A gap has an event signature if its before/after interval intersects the event's nominal time ±0.1 years. Check the documented approximately 1.2-year bearing replacement and approximately 2-year gearbox replacement. These numerical guard tolerances are prespecified safeguards, not publisher facts or a validated identity classifier.

**Stop before vibration analysis** if time is invalid/nonfinite, fewer than two rows are available, any duplicate/reversal/reset leaves chronology unresolved, a record does not bracket both events, or another file or files show gaps at both event times while assumed Turbine 5 does not. Write the contradiction and time-only evidence to `results_v3\metadata_identity_check.json` and `results_v3\REPORT.md`; append the stop to `DEVIATIONS.md`. Do not permute the mapping or examine vibration to resolve it.

Absent gaps, or gaps shared across several turbines, do not prove or disprove identity. If these blind checks find no contradiction, proceed under **ASSUMED_MAPPING_NOT_CONTRADICTED_NOT_VERIFIED**, retaining that limitation in the result. Time metadata alone cannot establish identity. Any independently discovered publisher filename key that contradicts the assumption requires stopping and a documented pre-analysis correction by the author, not an automatic remap.

MathWorks source: [data repository](https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data) and [official example](https://www.mathworks.com/help/predmaint/ug/wind-turbine-high-speed-bearing-prognosis.html). Owner: Eric Bechhoefer; supplied license: CC BY-NC-SA 4.0. All 50 `data-*.mat` files under `D:\external_unseen\mathworks_hss\WindTurbineHighSpeedBearingPrognosis-Data-main` enter in ascending filename timestamp order. Use `vibration` only; do not load/use `tach` or derive speed features. Header inspection, explicitly permitted in stage 1, found `vibration` shape 585936×1 and `tach` shape 2446×1 in the first file; all vibration shapes and header names/classes are fixed in the JSON. Headers and opaque byte digests are not signal statistics.

The official example treats both March 17 measurements as separate days. Preserve all 50 ordered **record-days**, including those two records. Report filename timestamps alongside record-day indices rather than claiming 50 unique calendar dates. Nominal sampling rate is 97656 Hz and one complete six-second record is one acquisition.

## 3. Inclusion and fixed replacement boundary

**Arm A keeps the unchanged v1 operating inclusion rule:** include a segment only if its raw, uncentered RMS is strictly greater than 0.5 g. RMS is `sqrt(mean(raw_float64_vibration**2))` over all 16384 raw samples, before the V12 float32 input cast. It is an eligibility calculation only; do not tune its cutoff or add a speed gate to A. Arm B adds the fixed speed inclusion of Section 3a and is reported alongside A, never replacing it. Event-1 onset follows Amendment 2.

The registered sensitivity uses **no operating filter**, with all other v2 rules fixed except the documented-onset amendment, for the six-turbine primary population (event 1 for Turbine 5). A and the no-filter sensitivity use the same documented event-1 onset t* = 1.0 year, then independently partition their own included pre-1.0 rows and fit their own training-only scaler, calibration and causal scoring. B inherits A's onset and partitions. Report sensitivity separately after A/B and the secondary outcomes; never replace A with it. No additional no-filter event-2 arm is introduced.

For **Turbine 5 event 1**, retain **time <1.2 years** and exclude **time ≥1.2 years**, including equality, before vibration values for those rows are parsed. This is the nominal documented output-shaft bearing replacement cutoff, frozen despite its approximate documentation. No record from the replacement bearing or later planet-bearing/gearbox event enters **event-1** onset estimation, commissioning, scaling or scoring. The ~2-year event remains in the metadata identity check and separately defines event 2 below. Do not move either nominal cutoff to a visually attractive gap.

Controls use their complete records, subject only to the arm's operating rule. Keep Turbine 2's documented early electrical sensor issue without correction, censoring, or special commissioning treatment; explicitly report that it is an electrical measurement issue rather than a known bearing failure. HSS has no operating RMS filter. Invalid schema, missing acquisitions, nonfinite vibration and insufficient required A/no-filter partition rows retain their explicit stops. Proxy no-onset retains its stop rule except the event-2 nonfatal not-evaluable outcome below; no outcome-driven exclusion is allowed.

## 3a. Arm B: variable-speed operating window (Amendment 1)

This physics-based adaptation for variable-speed machines was decided before target analysis. Apply the identical fixed procedure to every SND turbine and the separate event-2 unit. Always report it alongside A.

- Start from **Arm A inclusion, Arm A's documented or proxy onset t*** for the applicable Turbine-5 event, and **Arm A's partition memberships/boundaries**. Do not re-estimate onset or re-partition. Controls retain A's full-record boundaries.
- Read column 2 (speed, cycles/min) from that unit's **Arm A training rows only** to compute **P25 and P75 using NumPy linear percentiles**. The operating window is **[P25, P75], inclusive**. No calibration/test/fault speed influences these bounds. The quantile levels and procedure are identical for all turbines; no per-turbine tuning is permitted.
- Keep only A rows whose speed is inside the window in **every partition**: training, calibration, healthy test and fault period. Read those speeds for row inclusion only. Speed is never an EDA descriptor or model input. No tach-derived speed is permitted.
- Fit the imported unit-local scaler on **kept training rows only**; set theta from **kept training+calibration history H** using the unchanged rule and calibration-only fallback. Score the kept rows causally in time order; persistence runs over **kept rows only**, continuing across inherited boundaries and gaps.
- Apply the v1 guards: **at least five retained training rows, five calibration rows and five healthy-test rows**; additionally **at least one fault-period row for each Turbine-5 event**. Controls have no fault partition. The 105-row proxy guard applies only when estimating an onset (event 2); event 1 uses documented onset and the unchanged healthy-partition guards. B does not re-estimate onset. If any required B partition fails, report that unit/arm **not evaluable**, with null outcomes/timing, counts and reason; do not scale, score or rescue it. Keep the primary denominator six, report evaluable counts separately, and continue other registered units/arms. Event 1 never invokes the proxy in A or no-filter. If event 2 has no proxy onset, both A/B are not evaluable (no proxy onset) and the run continues; other registered guards remain unchanged.
- Report strict success, healthy alarms, fault detection, timing and lead days per seed exactly as for A. Record **retained/base counts for every partition**, source-row memberships, the fixed A boundaries/t* and the training-derived window bounds. No retained row moves to a different partition.

**HSS: not applicable (no speed column).** MAT records have no speed column, and v1 forbids using tach or deriving speed. HSS B is identical to A: same rows, 15/5 commissioning, scaler, theta, 21–50 testing and causal history. Report both labelled arms; do not fit a window or drop HSS rows.

## 3b. Secondary Turbine 5 event 2 (Amendment 1)

**Turbine 5, event 2: planet-bearing failure leading to gearbox replacement.** Select Turbine-5 rows with **time strictly greater than 1.2 years and strictly less than 2.0 years**. Exclude equality at both boundaries and all rows at/after 2.0 before converting their vibration values. The 2.0-year endpoint is the nominal documented gearbox replacement, approximately dated. This segment starts with a **new output-shaft bearing** after the first replacement.

Treat it as a **separate unseen unit**. Apply A inclusion and the same imported historical onset estimator **on this segment only**, with its existing no-onset rule and imported 60/20/20 split of its own pre-onset rows. The authors do not document degradation onset for this event. **The sole event-2 stop exception:** if the proxy returns no onset (including insufficient data), record **not evaluable (no proxy onset)** for A and inherited B, all three seeds, with null outcomes/timing and no scaler, threshold or detector scoring; continue all remaining units, including HSS. Retain six event-2 output rows and the unchanged primary denominator. Other schema/data/partition errors keep their stops. Fit its own training-only scaler and local theta and score causally. Do not borrow event-1 fit rows, onset, scaler, threshold or persistence history. Apply B as in Section 3a, using **this event's own A training rows, onset and boundaries**. Use all three frozen seeds and unchanged global models and detector parameters.

Strict success requires a silent healthy test and at least one fault-period alarm. Report first post-commissioning alarm year/source row, healthy/fault/strict outcomes, whether the first alarm is in the fault period, signed alarm-minus-2.0 and lead years, and **lead days = (2.0 - first alarm year) × 365.25**, per seed and A/B arm. Include premature alarms and preserve null timing for no alarms. This secondary outcome stays outside the six-turbine primary denominator.

The accelerometer is on the **output-shaft bearing housing, not on the failed planet bearing**. Event 2 is a harder, indirect test. It shares a turbine/farm and measurement channel with event 1; it is not an independent seventh primary turbine.

## 4. Documented-onset precedence, historical proxy and partitions

**General rule for every fault unit/event:** when data owners/authors document when degradation or a defect was present, use that documented onset. Only when no such onset is documented use the unchanged pre-registered historical proxy with its existing stop rule, subject to the explicit event-2 nonfatal exception. This rule does not invent a strict HSS onset or score; HSS remains timing only.

**Turbine 5 event 1:** Martin-del-Campo et al. state that the decrease in model fidelity "in the beginning of year one" corresponds to the defective-bearing period; the failure description states replacement after 1.2 years. Source: supplied `snd_paper.txt`, arXiv:1902.01426v2, failure descriptions and model-fidelity discussion. The author adopts nominal **t* = 1.0 year** on the time column (years since recording start). All included rows with **time <1.0** are assumed healthy; **1.0 <= time <1.2** is fault testing; **time >=1.2** stays excluded before vibration parsing. This applies to A, inherited B and the no-filter sensitivity. A and no-filter each apply the unchanged imported 60/20/20 `_three_way` and guards to their own included healthy rows. Event 1 does not extract proxy descriptors or call the onset estimator and therefore cannot hit its no-onset stop. Its partition guard stops remain. Lead time still references 1.2 years. Audit onset is exactly 1.0, separately from the first eligible fault row's timestamp/source row.

**Turbine 5 event 2:** no documented degradation onset is available. Import `frozen_v19_eda_features.acquisition_features` and `frozen_v19_analytic_onset.estimate` on its own included **1.2 < time <2.0** segment only; B inherits A. No control onset is estimated. Do not derive an onset from detector scores, alarm time, another event or the strongest-change diagnostic.

The exact historical rule is:

1. Extract raw RMS, kurtosis, crest factor, spectral entropy, and band-4 relative power. The last term is the fraction of non-DC power in normalized spectral-bin positions 0.4–1.0. These five raw EDA descriptors are separate from the detector's signed-log features.
2. The first 40 included records fit each descriptor's median and scale. Scale falls back in order from 1.4826 MAD to IQR/1.349, population standard deviation, then one. Clip standardized values to [-20,20], and combine by root mean square across the five descriptors.
3. Apply a trailing median with `w=clip(ceil(0.01*n),3,21)`, allowing partial initial windows. The first 100 smoothed values set the reference to the larger of their 99th percentile and their median plus six scaled MADs, with the exact implementation's 1e-12 floor.
4. At least 105 included records are required. Find the first five-record window whose end has zero-based index at least 100 and contains at least three strict exceedances. Backdate t* to the first exceedance in that confirming window, exactly as the imported function does.

The historical implementation includes interpolation and full-record diagnostics; it is a retrospective departure proxy. The full included record determines smoothing length. Its five descriptor families overlap the detector, and physical damage onset is unknown. Results are conditional on this partition, not prospective damage-onset validation.

**No proxy substitution:** if an undocumented event has no estimated onset or insufficient data, do not invent a 50% boundary, borrow another onset or use an alarm. The default remains STOP / not evaluable. For event 2 only, retain both arms/all seeds as **not evaluable (no proxy onset)** with null outcomes/timing and continue, without changing any denominator. Event 1 uses its documented onset directly.

For each Turbine-5 event, all included records strictly before its own t* are assumed healthy; the t* record and later included records before that event's replacement are fault testing. Partition the assumed-healthy sequence with the **same imported `_three_way` function and five-row minimum guards** as the paper's cache provenance. If healthy length is h, require h≥15; training ends at `max(5,floor(0.60*h))`; calibration ends at `min(max(train_end+5,floor(0.80*h)),h-5)`; the remainder is healthy testing. No boundary record belongs to two partitions. Apply this independently for A and the event-1 no-filter sensitivity; B filters A memberships without moving the boundaries.

For every control, apply this identical 60/20/20 function to its full included record; only the last approximately 20% is held-out testing. No control onset estimation is performed.

For HSS, record-days 1–15 are training (30% of all days); 16–20 are calibration (next 10%); 21–50 are testing. These commissioning rows are assumed healthy. The local scaler fits days 1–15 only. There is no HSS onset proxy or strict healthy/fault partition.

## 5. Frozen cohort, features and global model

The paper's active source is `autoencoder_cwru_release\full_pipeline_simulation\research\healthy_only_causal_no_ldm_20260922\run_study.py`, with `wider_models.py` and `cadence.py` in `research\healthy_only_causal_20260908`. `paper\RESULTS_TRACEABILITY.md` identifies the active saved results as `results\healthy_only_causal_no_ldm_20260922`: global_h12, dense cadence, strict counts **45/44/44**. The earlier sealed `research\external_dtu_fe8_20260922\frozen_config.json` is the protocol template and independently fixes seed 17070's same 54-client checkpoint.

The original training population is 54 logical units in nine datasets: DLR, Ferrara, HIT, IMS, KAIST, Paderborn, PRONOSTIA, UNSW and XJTU. LDM-UPB is absent. Every saved state has 54 unique client IDs. **No unseen SND or HSS unit joins federation, trains model parameters, updates FedAdam, or personalizes a decoder.** No retraining was needed in stage 1.

Use the exact saved final global 22→12→22 autoencoder: tanh hidden layer, linear reconstruction, 562 parameters; per-record mean squared reconstruction error over 22 coordinates. Training used a fixed budget of 12 FedAdam rounds, equal-client weighting, two full-batch local steps per round, local learning rate 0.03, weight-matrix L2 1e-4, server learning rate 0.01, betas 0.9/0.99, tau 0.001, and seeds 17070/17071/17072. Twelve rounds is a budget, not a convergence claim. Preserve the checkpoint's NumPy float64 arrays/arithmetic; the paper's 2248-byte float32 storage accounting does not authorize quantization or a change of the software model.

The located original files are in `E:\masters\autoencoder_cwru_release\full_pipeline_simulation\results\healthy_only_causal_no_ldm_20260922`:

| Original final model | SHA-256 |
|---|---|
| global_h12_seed17070.pkl | 28e697d32045c36d0497d5c967490f61d96cbe61bb38bed9ca0a77aa02896521 |
| global_h12_seed17071.pkl | 43078d1006a30d157e165017f166059a305640680e78607fd6cfe0fc66cc8298 |
| global_h12_seed17072.pkl | 5edf39760ec8c077a9b5aa59539dac3c645b8170d3f81a95b135ea4d9c123dac |

Each hash matches its saved checkpoint JSON. Byte-identical runnable copies and checkpoints are sealed under `frozen\models\` in this bundle; stage 2 loads those copies, not a newly trained model or a mutable result selected later. The source paths and source hashes remain recorded in `protocol.json` and `source_snapshot_manifest.json`.

Import the unchanged V12 `local_preprocess_current_acquisition` to produce its 100-coordinate representation, then select exactly:

`[0,1,2,3,4,5,6,7,28,29,30,64,65,66,67,68,69,70,71,92,93,94]`.

These are mean and signed-most-extreme channel summaries of eight time descriptors (RMS, peak-to-peak, kurtosis, skewness, crest, clearance, shape and impulse factors) and three spectral descriptors (entropy, centroid, spread). Time descriptors first use signed log1p. The intermediate representation includes other coordinates which are not selected. Both targets are one-channel acquisitions, so mean and signed-extreme summaries coincide, as for single-channel cohort data such as Ferrara; retain all 22 coordinates, including these repeated summaries.

Use one entire source segment per acquisition. The time descriptors use the complete segment. For a spectrum longer than 16384 samples, **only the already-existing V12 spectral step** takes `x[::ceil(n/16384)][:16384]`, centers, Hann-windows, performs rFFT and excludes DC. SND needs no spectral length reduction; HSS uses that existing long-record branch. Spectral-bin coordinates are normalized to [0,1], not a common physical-Hz grid. This striding is not anti-aliased resampling and does not remove sampling-rate heterogeneity. Do not add signal filtering, detrending outside the existing spectral centering, envelope transforms, record subdivision, rate conversion, feature smoothing or target-specific bands.

For each unseen unit and each arm, use the exact imported `data_cache.physical_matrix([features100], train_indices)`: per-feature median center; scale is `max(1.4826*MAD, population_std)` from **training rows only**, replacing scale by one if at most 1e-8; transform by absolute asinh. Calibration/test/fault rows do not fit this scaler. The streamwise maximum is an identity for these single-stream units. Scaling is unit-local and fixed after commissioning, not shared across turbines or updated in testing.

## 6. Local calibration and causal application

Every unseen unit receives every seed's unchanged final global model. For its own training and calibration score history H, compute the causal full-window statistic q: the 13th-largest score in the latest 21 acquisitions. Threshold is **theta = 2 × max finite q over H**. If H has no complete 21-record window, use **theta = 2 × NumPy's linear 99.5th percentile of calibration scores only**. Do not substitute training+calibration point scores for this documented short-history fallback.

HSS has 15+5=20 commissioning scores, so it necessarily uses this fallback on calibration days 16–20. The fallback changes the reference calculation only; the operational alarm still requires a complete 21-score window and at least 13 strict exceedances.

Import `wider_models.score` and `cadence.dense_flags`. An alarm occurs when q is **strictly greater** than theta. Suppress all flags through the final calibration index inclusive. Score each acquisition in chronological order. The threshold remains fixed, and the trailing score history continues across the commissioning/test boundary, estimated onset and calendar gaps. Never reset state at onset, after a gap, or at the first alarm. The original dense rule is a rolling decision, not an irreversible latch. No future score enters any decision; offline onset partitioning remains retrospective as explicitly stated above.

## 7. Prohibitions, integrity and stop conditions

V1/v2 access attestations remain in their unchanged archives. Amendment 2 used staged source/protocol artifacts and the supplied authors' paper text only. No target numeric data or result file was read; `D:\external_unseen\results` was not accessed. The runner was never imported or executed. Validation is syntax compilation, static AST/seal comparisons and isolated argument parsing only. The reported v2 stop is author/Opus-supplied administrative history.

After target access, do not change k, m, alpha, features, preprocessing, model weights or architecture, rounds, inclusion, cutoffs, partition fractions, onset algorithm or mapping to improve results. No per-turbine tuning, speed stratification beyond the fixed Amendment-1 B inclusion, tach-derived features, seed selection, dropped failures or silent reruns/rescues. Log all deviations and administrative/scientific stops in `D:\external_unseen\DEVIATIONS.md` with time, reason, access state and consequences. Amendment 2 is explicitly disclosed above as post-stop; no unseen score or viewed control result informed it. All other post-result changes remain prohibited.

Before executing imported code or unpickling checkpoints, stage 2 verifies all six current/archive protocol hashes and all sealed runner/source/model hashes. The source files under `frozen\pipeline` and `frozen\python_tools` are byte-identical copies, not rewritten implementations. The source snapshot includes the import closure and metadata JSON required by the original imports. Stage 2 forbids falling back to live E: source. Then, after the time-only gate, it verifies all six source CSVs against the original download hashes and all 50 MAT files plus their downloaded archive against frozen opaque SHA-256 values. It records its Python/package versions and fixes BLAS thread counts to one. No analysis starts on hash drift.

Once `results_v3\ANALYSIS_STARTED.json` exists, the runner refuses any automatic rerun, overwrite or resume. An exception produces an incomplete report and `STOP.json`; incomplete observed rows do not silently replace the registered denominator. A missing proxy onset remains not evaluable, never success or a reduced-denominator study. Event 2 records this outcome and continues instead of producing a global STOP; all other stop rules remain. A metadata-only command can be repeated only with identical metadata and before analysis starts.

## 8. Stage-2 commands and reporting

After Opus has reviewed and frozen/committed the registration and delivered the bundle to D:, run these exact commands, in this order, **in stage 2 only**:

```powershell
python -B D:\external_unseen\run_unseen.py --protocol D:\external_unseen\protocol.json --metadata-only
python -B D:\external_unseen\run_unseen.py --protocol D:\external_unseen\protocol.json --run
```

Only proceed to the second command if the first exits successfully. The full-run command independently repeats the same time-only gate and compares its evidence with the saved gate before interpreting vibration. Neither v3 command has been run during this amendment. Rerun every unit/control from scratch; do not import or resume any v2 output.

All analytical outputs are under `D:\external_unseen\results_v3\`:

- `summary.json`: A/B per-seed strict counts /6 and pooled /18, separate no-filter counts, evaluable/processed counts, every event-1/event-2 timing/outcome including lead days, separate HSS A/B record-day timing, completion status and environment.
- `per_unit.csv`: one row per primary SND turbine/arm/seed, secondary event-2 unit/arm/seed and HSS arm/seed; failures, not-evaluable rows/reasons/counts and null alarm times retained.
- `REPORT.md`: **first table** per seed and A/B arm: strict successes /6, event-1 detection and lead days, first-alarm fault membership, controls silent x/5, evaluable/processed counts. **Second table**: event 2 and HSS per seed and A/B arm, outcomes/timing, lead days (HSS record-days), first-alarm fault membership or explicit not-applicable. Then **no-filter sensitivity**, then **partition/onset audit and limitations**. State facts without defensive or promotional language.
- `metadata_identity_check.json`, `partition_onset_audit.json`, `execution_environment.json`, `hss_record_days.csv`: time-only identity evidence, onset and source-row partition audit, runtime provenance and day/timestamp mapping.
- `scores\*.csv`: causal score/alarm series with source row/day, time coordinate and partition, permitting independent review without rewriting source data.
- `ANALYSIS_STARTED.json` and `COMPLETE.json` or `STOP.json`: execution status. A completed run has **54 primary SND turbine-arm-seed rows** (6 turbines × A/B/no-filter × 3 seeds), **six secondary event-2 rows** (A/B × 3 seeds) and **six HSS rows** (A/B × 3 seeds): **66 total**, including explicit B partition-guard and both event-2 no-proxy not-evaluable rows if any. This is an expected design count, not an observed target result. HSS A/B duplicate one case; the primary remains six turbines per seed/arm.

Report that all SND controls are from the same farm, seed repetitions are correlated, event-1 labels use nominal documented onset and assumed healthy commissioning, while event-2 labels use the proxy when available, nominal replacement dates are approximate, and HSS is a small case study with assumed commissioning health. Successful transfer does not remove the development selection bias noted in the earlier DTU protocol. No extra external-data tuning is authorized.

## Amendment history and v3 delivery

Amendment 1 was frozen before v2 analysis; its full decisions and wording remain in the unchanged v2 archives. Amendment 2 follows the v2 no-sustained-departure stop for event 1 and changes only onset precedence, documented event-1 t*=1.0, the event-2 no-proxy continuation, disclosures and fresh results directory. Models, features, scaler, k=13, m=21, alpha=2, inclusion rules, arms, controls' partitions, HSS timing-only design and reporting order/lead days remain fixed.

After Opus reviews/freezes/commits and delivers the changed top-level artifacts with hash verification, stage 2 uses current `run_unseen.py` and v3 `protocol.json`. No commit, D: delivery, target/result access or analysis execution was performed by Codex during Amendment 2.
