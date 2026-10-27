# Pre-registered unseen-machine test: frozen FedAdam H12

Protocol ID: `SND_HSS_FROZEN_H12_20261026`. Stage 1 of 2, 26 October 2026.

This document and `protocol.json` are frozen before any target vibration analysis. Their byte SHA-256 values are in `PROTOCOL_SHA256.txt`. The JSON also seals the runner, source snapshot manifest, imported pipeline files and model copies. Opus, rather than Codex, commits this registration. No stage-2 command has been executed.

**Delivery limitation:** the managed Codex sandbox denied writing to `D:\external_unseen`. The complete bundle is staged at `C:\Users\Dell\AppData\Local\Temp\codex_unseen_stage1_20261026`. Opus must copy its contents byte-for-byte to `D:\external_unseen` before execution. The runner refuses execution from the staging location. This administrative deviation is recorded in `DEVIATIONS.md`; it does not change any scientific rule. No analysis output is written to E:.

## 1. Hypothesis, units and outcomes

The hypothesis is that the global H12 detector trained on the paper's 54 units transfers to unseen turbines after device-local commissioning: it remains silent in held-out assumed-healthy periods, detects Turbine 5's output-shaft bearing degradation before replacement, and leaves the five healthy controls silent in their held-out tests.

The **primary population is all six SND turbines** from one farm. For each seed 17070, 17071 and 17072:

- Turbine 5 strictly succeeds only if it has **no alarm anywhere in its held-out healthy test** and **at least one alarm in its assigned fault period**.
- Each control (Turbines 1, 2, 3, 4 and 6) strictly succeeds only if it **never alarms in its entire held-out test**, the last approximately 20% of its included record. Its first 80% is commissioning, not held-out evaluation. Controls do not have assigned fault periods.
- Report every turbine's healthy-alarm, fault-detection and strict-success outcome, the strict count out of six per seed, and the pooled count out of 18 turbine-seed evaluations. Preserve every failure. Eighteen evaluations do not represent 18 independent machines; there are six machines and one farm. No significance test or target accuracy gate is registered.
- For Turbine 5, report the first post-commissioning alarm time in source years, its source CSV row, `first_alarm_time - 1.2` years (negative means before replacement), and `1.2 - first_alarm_time` years of lead time. Include a premature healthy-test alarm in this timing, even if it causes strict failure. No alarm gives null timing and false detection; do not substitute the last record.

The **secondary outcome is the single MathWorks HSS lifecycle**, reported separately for every seed: first alarm record-day, lead in record-days to the day-50 terminal endpoint, and whether the first alarm occurs strictly before day 50. An alarm only on day 50 is not advance warning. Failure is documented by the end of the record; day 50 is a terminal proxy, not an exact failure timestamp. There is **no HSS strict score** and no pooling with the SND denominator.

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

**Stop before vibration analysis** if time is invalid/nonfinite, fewer than two rows are available, any duplicate/reversal/reset leaves chronology unresolved, a record does not bracket both events, or another file or files show gaps at both event times while assumed Turbine 5 does not. Write the contradiction and time-only evidence to `results\metadata_identity_check.json` and `results\REPORT.md`; append the stop to `DEVIATIONS.md`. Do not permute the mapping or examine vibration to resolve it.

Absent gaps, or gaps shared across several turbines, do not prove or disprove identity. If these blind checks find no contradiction, proceed under **ASSUMED_MAPPING_NOT_CONTRADICTED_NOT_VERIFIED**, retaining that limitation in the result. Time metadata alone cannot establish identity. Any independently discovered publisher filename key that contradicts the assumption requires stopping and a documented pre-analysis correction by the author, not an automatic remap.

MathWorks source: [data repository](https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data) and [official example](https://www.mathworks.com/help/predmaint/ug/wind-turbine-high-speed-bearing-prognosis.html). Owner: Eric Bechhoefer; supplied license: CC BY-NC-SA 4.0. All 50 `data-*.mat` files under `D:\external_unseen\mathworks_hss\WindTurbineHighSpeedBearingPrognosis-Data-main` enter in ascending filename timestamp order. Use `vibration` only; do not load/use `tach` or derive speed features. Header inspection, explicitly permitted in stage 1, found `vibration` shape 585936×1 and `tach` shape 2446×1 in the first file; all vibration shapes and header names/classes are fixed in the JSON. Headers and opaque byte digests are not signal statistics.

The official example treats both March 17 measurements as separate days. Preserve all 50 ordered **record-days**, including those two records. Report filename timestamps alongside record-day indices rather than claiming 50 unique calendar dates. Nominal sampling rate is 97656 Hz and one complete six-second record is one acquisition.

## 3. Inclusion and fixed replacement boundary

The **primary SND analysis uses the data authors' operating rule**: include a segment only if its raw, uncentered RMS is strictly greater than 0.5 g. RMS is `sqrt(mean(raw_float64_vibration**2))` over all 16384 raw samples, before the V12 float32 input cast. It is an eligibility calculation only; do not tune its cutoff or add a speed gate.

The registered sensitivity uses **no operating filter**, with all other rules fixed. Each arm runs its own inclusion → historical onset → chronological partition → training-only scaler → calibration → causal scoring sequence. Thus removal of the operating filter can alter the estimated onset and partition membership. Do not select a common onset after observing which arm performs better. Report sensitivity separately, never replace the primary with it.

For Turbine 5, retain **time <1.2 years** and exclude **time ≥1.2 years**, including equality, before vibration values for those rows are parsed. This is the nominal documented output-shaft bearing replacement cutoff, frozen despite its approximate documentation. No record from the replacement bearing or the later planet-bearing/gearbox event enters onset estimation, commissioning, scaling or scoring. The ~2-year event is used only for the metadata identity check. Do not move the 1.2 cutoff to a visually attractive gap.

Controls use their complete records, subject only to the arm's operating rule. Keep Turbine 2's documented early electrical sensor issue without correction, censoring, or special commissioning treatment; explicitly report that it is an electrical measurement issue rather than a known bearing failure. HSS has no operating RMS filter. Invalid schema, missing acquisitions, nonfinite vibration, insufficient partition rows or missing onset cause an explicit stop, not outcome-driven exclusion.

## 4. Historical onset and chronological partitions

Only Turbine 5 receives a degradation-onset estimate, from **its own included pre-replacement record**, separately per inclusion arm. Import `frozen_v19_eda_features.acquisition_features` and `frozen_v19_analytic_onset.estimate`, the executable historical rule described in `paper\supplement.tex`, Section S1. Do not estimate onset from global model scores, another turbine, an alarm time or the source phrase “beginning of year one.” That phrase is context, not an exact damage label.

The exact historical rule is:

1. Extract raw RMS, kurtosis, crest factor, spectral entropy, and band-4 relative power. The last term is the fraction of non-DC power in normalized spectral-bin positions 0.4–1.0. These five raw EDA descriptors are separate from the detector's signed-log features.
2. The first 40 included records fit each descriptor's median and scale. Scale falls back in order from 1.4826 MAD to IQR/1.349, population standard deviation, then one. Clip standardized values to [-20,20], and combine by root mean square across the five descriptors.
3. Apply a trailing median with `w=clip(ceil(0.01*n),3,21)`, allowing partial initial windows. The first 100 smoothed values set the reference to the larger of their 99th percentile and their median plus six scaled MADs, with the exact implementation's 1e-12 floor.
4. At least 105 included records are required. Find the first five-record window whose end has zero-based index at least 100 and contains at least three strict exceedances. Backdate t* to the first exceedance in that confirming window, exactly as the imported function does.

The historical implementation includes interpolation and full-record diagnostics; it is a retrospective departure proxy. The full included record determines smoothing length. Its five descriptor families overlap the detector, and physical damage onset is unknown. Results are conditional on this partition, not prospective damage-onset validation.

**No documented substitution for a missing historical onset was found.** The scale fallbacks above are documented; the “strongest positive change” returned by the estimator is a diagnostic, not a documented replacement for missing sustained onset. If the estimator returns no onset or insufficient data, stop and report not evaluable. Do not invent a 50% boundary, borrow a control's onset, use one year, silently drop Turbine 5 or reduce the primary denominator. A different fallback would need an explicitly documented amendment before target analysis; it is not authorized by this protocol.

For Turbine 5, all included records strictly before t* are assumed healthy; the t* record and later included records before replacement are fault testing. Partition the assumed-healthy sequence with the **same imported `_three_way` function and five-row minimum guards** as the paper's cache provenance. If healthy length is h, require h≥15; training ends at `max(5,floor(0.60*h))`; calibration ends at `min(max(train_end+5,floor(0.80*h)),h-5)`; the remainder is healthy testing. No boundary record belongs to two partitions.

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

Before this seal: no target time rows, speed values, vibration samples, RMS, features, onsets, model scores, alarms, plots or sample summaries have been inspected or computed. Stage-1 target access was restricted to directory/file names, README/license and publisher text, existing download hash text, MAT names/shapes/classes via `whosmat`, and SHA-256 of opaque MAT bytes. Those hashes fix file identity without interpreting samples. Source code, manuscript and already-saved cohort models/evidence were inspected. The runner itself was never executed, even for `--help` or metadata mode.

After target access, do not change k, m, alpha, features, preprocessing, model weights or architecture, rounds, inclusion, cutoffs, partition fractions, onset algorithm or mapping to improve results. No per-turbine tuning, speed stratification, tach-derived features, seed selection, dropped failures or silent reruns/rescues. Log all deviations and administrative/scientific stops in `D:\external_unseen\DEVIATIONS.md` with time, reason, access state and consequences. A result-driven amendment does not retain the original preregistered status.

Before executing imported code or unpickling checkpoints, stage 2 verifies both protocol hashes and all sealed runner/source/model hashes. The source files under `frozen\pipeline` and `frozen\python_tools` are byte-identical copies, not rewritten implementations. The source snapshot includes the import closure and metadata JSON required by the original imports. Stage 2 forbids falling back to live E: source. Then, after the time-only gate, it verifies all six source CSVs against the original download hashes and all 50 MAT files plus their downloaded archive against frozen opaque SHA-256 values. It records its Python/package versions and fixes BLAS thread counts to one. No analysis starts on hash drift.

Once `results\ANALYSIS_STARTED.json` exists, the runner refuses any automatic rerun, overwrite or resume. An exception produces an incomplete report and `STOP.json`; incomplete observed rows do not silently replace the registered denominator. A stop for missing onset remains not evaluable, not a success or a newly reduced-denominator study. A metadata-only command can be repeated only with identical metadata and before analysis starts.

## 8. Stage-2 commands and reporting

After Opus has reviewed and frozen/committed the registration and delivered the bundle to D:, run these exact commands, in this order, **in stage 2 only**:

```powershell
python -B D:\external_unseen\run_unseen.py --protocol D:\external_unseen\protocol.json --metadata-only
python -B D:\external_unseen\run_unseen.py --protocol D:\external_unseen\protocol.json --run
```

Only proceed to the second command if the first exits successfully. The full-run command independently repeats the same time-only gate and compares its evidence with the saved gate before interpreting vibration. Neither command has been run in stage 1.

All analytical outputs are under `D:\external_unseen\results\`:

- `summary.json`: primary and no-filter per-seed counts and pooled totals, every Turbine-5 timing/outcome, separate HSS timing, completion status and environment.
- `per_unit.csv`: one row per SND turbine/arm/seed plus HSS unit/seed; failures and null alarm times retained.
- `REPORT.md`: primary/sensitivity tables, timing, sensor issue, assumptions, limitations and any stop.
- `metadata_identity_check.json`, `partition_onset_audit.json`, `execution_environment.json`, `hss_record_days.csv`: time-only identity evidence, onset and source-row partition audit, runtime provenance and day/timestamp mapping.
- `scores\*.csv`: causal score/alarm series with source row/day, time coordinate and partition, permitting independent review without rewriting source data.
- `ANALYSIS_STARTED.json` and `COMPLETE.json` or `STOP.json`: execution status. A completed run has 36 SND turbine-arm-seed rows and three HSS rows. This is an expected design count, not an observed target result.

Report that all SND controls are from the same farm, seed repetitions are correlated, healthy/fault labels are proxy-partition assumptions, nominal replacement dates are approximate, and HSS is a small case study with assumed commissioning health. Successful transfer does not remove the development selection bias noted in the earlier DTU protocol. No extra external-data tuning is authorized.

## Decisions to review before stage 2

The author/Opus should review the **assumed filename mapping**, the **nominal time <1.2-year exclusion**, and the **stop-if-no-historical-onset rule** before permitting target sample analysis. They are fully specified here so execution has no outcome-dependent choice. No positive identity key or documented no-onset substitution has been established. If the author requires a publisher-confirmed mapping, a more precise independently sourced replacement date or a different documented fallback, resolve and log that before any vibration is analysed; preserve this version and issue a clearly identified amendment rather than editing it silently. The filesystem delivery limitation must also be resolved by copying the already prepared bundle.
