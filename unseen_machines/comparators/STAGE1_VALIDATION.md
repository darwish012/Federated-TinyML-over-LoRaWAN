# Stage-1 validation

Stage 1 completed in `C:\Users\Dell\AppData\Local\Temp\codex_unseen_comparators_20261027`. No commit. No delivery or stage-2 analysis was executed.

## Amendment C-1 (2026-10-27)

Amendment C-1 (2026-10-27, before any comparator result): persistence-augmented variants of published rules removed by author decision.

Published comparators are evaluated unmodified. Primary methods are proposed FedAdam H12, Wang 3-sigma, regime-2 GEV, two-means, FedAvg H12 and FedProx H12. Secondary variants are pointwise Wang 4/6-sigma and unclustered GEV. No persistence-augmented published-rule variants are computed or reported. The proposed framework and FedAvg/FedProx retain their own causal 13-of-21 rule. Frozen source snapshots, development models, partitions and outcome definitions are unchanged.

The amendment rerun passed Python compilation of all four stage-1 scripts, all nine synthetic checks, the separate argument-only dry run, local runner `--verify-only`, and `DELIVER.ps1 -VerifyOnly`. `seal_stage1.py` regenerated both protocol files and their SHA-256 seals. No development retraining, delivery, stage-2 execution or commit was performed for this amendment.

## Development reproduction

`python -I -B .\train_development.py` retrained on the exact 54-client non-LDM development cohort. It selected metadata before opening any cache and verified original cache/provenance SHA-256 values. There were 49 scored development units (43 fault units, six controls), with five additional unscored training clients. The process denied D: file reads outside frozen code/models and denied writes outside staging.

| Algorithm / gate | 17070 | 17071 | 17072 | Expected |
|---|---:|---:|---:|---|
| FedAdam strict /49 | 45 | 44 | 44 | 45/44/44 |
| Maximum absolute difference from frozen FedAdam weights | 0 | 0 | 0 | <=1e-10 |
| FedAvg strict /49 | 46 | 44 | 43 | 46/44/43 |
| FedProx mu=0.01 strict /49 | 46 | 44 | 43 | 46/44/43 |

All gates passed. Training/evaluation took approximately six seconds. `development_training.log` records the execution; `development_validation.json` retains the input hashes, all per-device C2 outcomes, counts, and exact FedAdam replication evidence. Six retrained C2 pickles and six JSON checkpoints are under `frozen/models/`. Each checkpoint records model SHA-256, development metrics, training settings and input dependencies. No existing output on E: or D: was modified.

C3 is not applicable: the frozen personalized model supplies no new-client commissioning adaptation procedure. The 40/43/41 development counts are historical paper values, not a claimed stage-1 reproduction. No C3 unseen scores or adaptation settings were invented. Optional C4 is omitted because its existing logES implementation fixes the development sampling rate and raw adapters.

## Code and synthetic checks

The final Python files were compiled with `python -I -B -m py_compile` (explicit compilation writes bytecode only within staging). A syntax error found in the first compile was corrected before the successful compile and subsequent tests.

`python -I -B .\test_stage1.py` passed all nine checks recorded in `synthetic_test_results.json` and `synthetic_validation.log`. A process audit hook prohibited every D: file read. Its CSV fixtures were synthesized entirely in staging and are not delivered as target results.

1. HSS strict yardstick handles days 21, 44, 45, and 50, no-alarm misses, and premature-plus-late alarms correctly; absent proxy onset produces null sensitivity outcomes.
2. The calibration-only short-history fallback and causal 13-of-21 confirmation produce the expected synthetic first alarm.
3. The exact six-primary/three-secondary method roster is enforced. C1 flags and fit metadata match the frozen pointwise Wang/GEV and two-means functions; an interception guard rejects any persistence-enabled C1 call. Wang/GEV fit parameters remain unchanged when held-out scores change.
4. A synthetic end-to-end proposed-method gate checks cached CSV schema, memberships, exact scores/alarms, per-unit outcomes, threshold/fallback metadata and public-reference checks across eight units.
5. A 1e-12 score discrepancy stops before comparator evaluation.
6. An alarm-vector discrepancy stops before comparator evaluation.
7. All 216 method/seed/unit outcome rows (nine methods, three seeds, eight units; 27 aggregate cells) aggregate with the common /7 denominator and render into exactly the three registered result files, including the C3 not-applicable row and HSS sensitivity table. The synthetic report contains no removed persistence variants.
8. Argument parsing dry run succeeds; real analysis from staging refuses before any target access.
9. The runtime audit hook blocks writes outside the three registered result files.

`python -I -B .\run_comparators.py --dry-run` only parses arguments and prints the fixed destination. It opens no protocol, source, score, or waveform files. `--verify-only` verifies the local seal, snapshots, model hashes and development evidence without target access. `DELIVER.ps1 -VerifyOnly` validates the complete delivery manifest from staging without creating or copying to D:. These local checks are performed on the final sealed bundle.

The live paper changed during preparation, after the first snapshot. Table III's comparator counts remained 40/43/41, 45/44/44 and 46/44/43 for both FedAvg/FedProx. The paper snapshot was refreshed before sealing and checked byte-for-byte against the live source. No training source, setting, model or development result changed in response.

## Scope of validation

Stage-2 target execution remains intentionally unperformed. Its raw-data adaptation and exact v3 reference gate are checked synthetically; they have not been claimed to pass on the target data. Stage 2 must stop if its original-v3 reconstruction, alarm vectors, membership, per-unit numbers, public reference, source seals or package versions differ. It cannot rerun or overwrite an existing results directory automatically.

The runner imports exact saved decision functions and the original frozen v3 feature, scaling, partition, onset, model and alarm implementations. Its original source runners are never invoked. It requires the fixed D: script directory and working directory. Only summary.json, per_unit.csv and REPORT.md under the new comparators/results directory may be written. Original data, frozen artifacts and v3 results are read-only.

## Stage-1 data boundary

No SND CSV fields were read; no HSS MAT signal values were read; no file under `D:\external_unseen\results_v3\scores\` was opened or hashed; no target `per_unit.csv` values were read. Only score filenames were listed. The known proposed results came from the user request and the explicitly permitted public v3 REPORT.md. Protocol/source text, frozen development models and E: development caches were read as authorized. All authored files and outputs stayed in staging. No D: write and no commit occurred.
