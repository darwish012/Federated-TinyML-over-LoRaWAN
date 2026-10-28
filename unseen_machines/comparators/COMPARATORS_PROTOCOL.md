# Comparator addendum: frozen stage-1 protocol

Protocol: `SND_HSS_COMPARATOR_ADDENDUM_20261027_STAGE1`, version 1. Sealed at 2026-10-27T10:48:32.527991+00:00.

The proposed framework's unseen-machine results were already known to the author and to the addendum preparers when this addendum was written. The comparators' unseen-machine results were not known. This is a prospective specification of additional comparators after disclosure of the proposed method, not a claim that this addendum preceded the original unseen test. No target numeric data were read during stage 1.

## Scope and stage boundary

All writes in stage 1 are in the registered C: staging directory. No commit and no delivery to D: occur in stage 1. No SND CSV fields, HSS MAT signal values, results_v3/scores file contents (including hashes), or target per_unit.csv values were read. The existing scores directory was listed by filename only. Public v3 documents, frozen code/models, and the development cohort on E: were used.

## Comparators and frozen settings

**C1 applies.** Import the byte-identical saved run_comparison.py functions; never run its original runner. Use the same v3 FedAdam H12 scores, local standardized 22-feature x, and original train/calibration partitions. Report Wang 3-sigma, regime-2 GEV, and retrospective scalar-score two-means as primary comparators. Secondary variants are pointwise Wang 4/6-sigma and unclustered GEV. No persistence-augmented variants of published rules are computed or reported. Published comparators are evaluated unmodified.

Amendment C-1 (2026-10-27, before any comparator result): persistence-augmented variants of published rules removed by author decision.

Wang baseline is first 100 training+calibration acquisitions when available, otherwise first min(30,H). Threshold is mean + sigma*population std, strict >. GEV clusters commissioning x with KMeans(n_init=10, random_state=seed), uses chronological block maxima (size 10, final partial block included), SciPy GEV MLE and 0.99 return level. Fewer than three maxima, degenerate maxima or invalid fit use the existing 0.995 empirical quantile fallback. Save fits, maxima, warnings and fallbacks. Two-means clusters the full scalar score trajectory, labels acquisition 0 healthy, and holds an alarm from the first changed label at zero-based index >=10, masked through calibration. Its retrospective nature is retained. For the proposed method and FedAvg/FedProx only, persistence needs a full 21 rows, 13 exceedances, and carries across commissioning/evaluation boundaries.

**C2 applies.** Train both optimizers afresh on exactly 54 non-LDM development clients for 12 rounds, seeds 17070/17071/17072, architecture 22-12-22 tanh/linear, float64. Use the original initialization, two full-batch local steps at 0.03 with weight L2=1e-4, equal-client aggregation; FedProx mu=0.01. No target data trains weights. The FedAdam replication gate passed with zero maximum absolute weight difference and 45/44/44 strict successes. FedAvg and FedProx each reproduced 46/44/43 on the same 49 scored development units. Stop on any discrepancy rather than proceeding. Apply C2 with unchanged v3 local scaling and causal dense 13-of-21, alpha=2.

**C3: not applicable.** The declared pca_and_personalized_h12/history_phase method has historical development counts 40/43/41. Its private decoder is trained per participating client across federated rounds. The frozen code has no procedure to commission a new private decoder; unknown IDs fall back to the global decoder. Initialization, training schedule or federation replay for new units would be new choices. No C3 outcomes are computed and no global fallback is relabeled personalized.

**C4: optional arm not included.** The existing logES function hard-codes 25600 Hz and its loaders select two-channel PRONOSTIA/XJTU data. Applying it at 12800 Hz SND or 97656 Hz HSS would require a sampling-rate/front-end adapter change. No new choice or waveform access was made for this optional arm.

## Units, partitions and alarm rules

Use v3 Amendment 2 Arm A only, authors' raw RMS >0.5 g inclusion for SND. Preserve identity assumptions, chronology, nominal replacement cutoffs, event-1 documented onset 1.0 year, event-2 historical onset, training-only feature normalization, and the original 60/20/20 SND partitions and guards. Event 2 is an independent processing unit, not an independent machine. No event-1 rows at/after 1.2; event 2 strictly 1.2<time<2.0. Five controls use their complete filtered record with only the final ~20% evaluated.

HSS uses all 50 ordered record-days including both March 17 records, training days 1-15, calibration 16-20 and testing 21-50. Load vibration only, never tach. For the project rule, 20 commissioning scores cannot fill a 21-score window: theta = 2*q0.995(calibration scores). Persistence continues through commissioning and all later boundaries with no latch/reset.

## Outcomes and common yardstick

Primary: strict successes out of 7 machines per seed/method: Turbines 1-4 and 6, T5 event 1, HSS. Controls succeed iff silent across all held-out testing. T5 event 1 succeeds iff silent in its healthy test and at least one fault-period alarm. HSS succeeds iff no alarm on record-days 21-44 inclusive AND at least one alarm on record-days 45-50 inclusive. The author specifies this as the paper yardstick; original v3 itself reported HSS timing without a strict score. Apply the addendum yardstick identically to every method. Alarm on day 50 counts for this strict yardstick but fails the original advance-warning criterion (<50).

Secondary for every method: event-2 healthy alarm/detection/strict success and lead days, controls silent /5, HSS first-alarm day and lead in record-days. SND lead=(replacement year-first post-commissioning alarm year)*365.25, including premature alarms. No alarm means null timing. Event 2 stays outside /7; not-evaluable outcomes remain null without denominator reduction.

HSS onset sensitivity invokes the unchanged historical V19 estimator on its waveform-derived descriptors, independent of detector scores. Its original minimum is 100+5=105 acquisitions, so 50 HSS records produce insufficient_post_commission_data and no onset. Retain null sensitivity outcomes for every method; never shorten the guard. If an onset is returned, assess the same fixed alarm vectors on days 21-50 before/from onset, without refitting commissioning/models.

## Mandatory proposed-method reference gate

Before computing comparator outcomes, reconstruct proposed scores and flags and require exact equality to the original v3 Arm A score CSVs; check source rows, coordinate, partition labels, per-unit outcomes, timing and thresholds/fallbacks. Verify the public reference below. Any mismatch stops the run; no tolerance, tuning, rescue or automatic rerun.

| Seed | SND strict /6 | Addendum strict /7 | Controls silent /5 | HSS first alarm | T5 event-2 first alarm (y) | Event-2 lead days |
|---:|---:|---:|---:|---:|---:|---:|
| 17070 | 5 | 6 | 5 | 49 | 1.523988818692 | 173.86308397274698 |
| 17071 | 5 | 6 | 5 | 47 | 1.523988818692 | 173.86308397274698 |
| 17072 | 5 | 6 | 5 | 45 | 1.52672542646071 | 172.8635379852257 |

Event 1 was missed for all three seeds with no post-commissioning alarm. Event 2 had no healthy-test alarm and was detected for all three Arm A seeds. These are known public reference values, not newly opened comparator results.

## Source/model integrity and execution

The JSON contains exact source-file hashes with byte-identical snapshot locations, the v3 frozen dependency hash map, all model/checkpoint hashes, package versions, complete settings, and the bundle artifact hash map. Development cache/provenance SHA-256 values and per-unit reproduction outcomes are retained in development_validation.json and every model checkpoint. The two protocol files are sealed by COMPARATORS_SHA256.txt. The runner verifies the seal and every bundle artifact, then verifies the original v3 protocol/source/model seals before target processing.

| C2 model | SHA-256 |
|---|---|
| frozen/models/fedavg_h12_seed17070.pkl | `64b4f05f57dba6d354463509a89988d6dafa137cd021d0d668f37a09ac7aabb9` |
| frozen/models/fedavg_h12_seed17071.pkl | `26a3003065a37b2c3cbde5ec582d94de99ef779c9fa2e111453514b00d4963cc` |
| frozen/models/fedavg_h12_seed17072.pkl | `17f0794ea95e1b5283dd3a22e283bbda938e9e8a34fee99b4c71831f32f159e6` |
| frozen/models/fedprox_h12_seed17070.pkl | `f59bbfb6e476a47ef70099fa6c6f6c4c227676bd3b76a6d72a1dae7c32a14db4` |
| frozen/models/fedprox_h12_seed17071.pkl | `a129255b2fc34ee309aa49bf89aa4e7c8d991b77c9676433c6e88ad89da314dd` |
| frozen/models/fedprox_h12_seed17072.pkl | `192c99f98b454609958e36c3dcea06533bbcc5232b63861a3ed19986d4a2dcb3` |

Only Opus delivers with `DELIVER.ps1`; its `-VerifyOnly` mode reads staging only. Delivery refuses an existing destination, copies only sealed artifacts, checks resolved paths stay within the fixed roots and verifies every copied hash. It never copies data or results.

Stage 2 refuses analysis anywhere except `D:\external_unseen\comparators` as both script directory and working directory. It reads original data/frozen/results files read-only and writes only `results/summary.json`, `results/per_unit.csv`, and `results/REPORT.md`. An existing results directory causes refusal. Runtime write guards block other writes and subprocesses. Failures produce an incomplete factual report, not a completed comparator table.

```powershell
Set-Location -LiteralPath D:\external_unseen\comparators
python -I -B .\run_comparators.py --protocol .\comparators_protocol.json --run
```

The report retains all failures, nulls, applicability limits, GEV diagnostics and the retrospective two-means caveat. No promotional language, outcome-driven selection, target tuning, additional operating arms or significance tests are registered. Seed repetitions, one-farm dependence, assumed identity, approximate replacements and retrospective onset labels remain limitations.
