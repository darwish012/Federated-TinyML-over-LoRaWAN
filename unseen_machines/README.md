# Unseen-machine test

This folder is the sealed record of the unseen-machine evaluation reported in Section V-B of the paper. The frozen FedAdam H12 models from the three initialization seeds (17070, 17071, 17072) were applied, without retraining, to machines that never took part in development or federated training.

## Machines

| Source | Machines | Sensor and cadence |
|---|---|---|
| Swedish wind farm (Martin-del-Campo, Sandin and Strömbergsson, *Int. J. Comput. Intell. Syst.* 14(1), 2021; data DOI [10.5878/bcmv-wq08](https://doi.org/10.5878/bcmv-wq08)) | Six turbines over 46 months. Turbines 1–4 and 6 stayed healthy. Turbine 5 had an output-shaft bearing defect (replaced after 1.2 years) and then a planet-bearing failure (gearbox replaced after 2.0 years). | Accelerometer on the output-shaft bearing housing, 16,384 samples at 12.8 kHz about every 12 h |
| Wind-turbine high-speed-shaft bearing (E. Bechhoefer; [MathWorks copy](https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data), CC BY-NC-SA 4.0) | One 2-MW high-speed-shaft bearing that failed after 50 days | One 6-s record per day at about 97.7 kHz |

The datasets are not redistributed here. `download_sha256.txt` lists the SHA-256 of every file used.

## Results (from `results_v3/`)

| Machine | Outcome (seeds 17070 / 17071 / 17072) | Strict |
|---|---|---|
| Turbines 1–4 and 6 (controls) | Silent throughout testing | 5/5 per seed |
| Turbine 5, output-shaft bearing | Missed | 0/1 per seed |
| High-speed-shaft bearing | First alarm on day 49 / 47 / 45 of 50; no alarm on days 21–44 | 1/1 per seed |
| **Total** | | **6/7 per seed** |
| Turbine 5, gearbox event | Detected 174 / 174 / 173 days before replacement, with a silent healthy test | detected |

A speed-window variant (each turbine's commissioning P25–P75 speed range) and an unfiltered variant gave the same strict outcome for every turbine and seed. `results_v3/REPORT.md` holds the full tables, the partition and onset audit and the limitations.

## Protocol

- `PROTOCOL.md` / `protocol.json`: protocol version 3, fixed before any target analysis. Versions 1 and 2 are kept unchanged (`PROTOCOL_v1.md`, `PROTOCOL_v2.md` and the matching JSON files). `PROTOCOL_SHA256*.txt` seal each version.
- `DEVIATIONS.md`: both amendments with timestamps. Amendment 1 added the speed-window arm, the gearbox event and lead-time reporting before any analysis. Amendment 2 adopted the data authors' documented defect onset (1.0 year) for the output-shaft event after the historical onset estimator had stopped the version-2 run, and before any Turbine 5 score or control result was viewed. Rule: a documented onset takes precedence, and the historical estimator is used otherwise.
- `results_v2_stopped/`: the stopped version-2 run, kept as evidence.
- `STAGE1_VALIDATION*.md`, `validate_stage1*.py`: static checks of the runner before execution.

## Code

- `run_unseen.py`: the runner as executed. It verifies every seal, imports only the frozen pipeline copy under `frozen/`, checks the dataset hashes and writes `results_v3/`. It was pinned to the authors' data location (`D:\external_unseen`) and refuses to run elsewhere. To rerun it, edit that path check and the file paths in `protocol.json`, which also changes their hashes.
- `frozen/pipeline/`, `frozen/python_tools/`: hash-sealed copies of the feature extractor, model and decision code used in the paper.
- `frozen/models/`: the three frozen global models (Python pickles) with their training checkpoints. Load pickles only from sources you trust.
- `frozen/evidence/`: hash-sealed reference files. Three manuscript snapshots listed in `protocol.json` (`paper__main.tex`, `paper__supplement.tex`, `paper__RESULTS_TRACEABILITY.md`) are omitted from this public copy, so the seal check reports them as missing.
- `DELIVER_TO_D.ps1`, `STAGING_NOTICE.txt`: delivery script and notice from the staging step.

Paths inside the JSON files refer to the authors' machine.

## Comparator test

`comparators/` holds the sealed comparison with the published decision rules on the same seven machines: the proposed framework 6/7 per seed, the published rules 0/7 to 4/7. See [`comparators/README.md`](comparators/README.md).
