# Comparator test on the unseen machines

This folder is the sealed record of the comparison reported in Table V of the paper. The published decision rules were run unmodified on the same seven unseen machines as the proposed framework, under the same strict success definition.

## Results (from `results/`)

| Method | Strict of 7 (seeds 17070 / 17071 / 17072) |
|---|---|
| Proposed framework (FedAdam H12) | 6 / 6 / 6 |
| FedAvg H12 | 6 / 6 / 6 |
| FedProx H12 | 6 / 6 / 6 |
| GEV regime rule | 4 / 3 / 4 |
| Wang 3-sigma rule | 1 / 1 / 1 |
| Javanmardi 2-means rule | 0 / 0 / 0 |

The seven machines are the five healthy SND turbines, Turbine 5's output-shaft bearing event and the MathWorks high-speed-shaft bearing. For that bearing, strict success means no alarm on days 21–44 and at least one alarm on days 45–50. `results/REPORT.md` also lists secondary threshold variants (Wang 4- and 6-sigma, GEV without clustering) and per-machine outcomes.

## Files

- `COMPARATORS_PROTOCOL.md` / `comparators_protocol.json`: the protocol, sealed by `COMPARATORS_SHA256.txt` before any comparator was run on the unseen machines.
- `run_comparators.py`: the runner as executed. Like `../run_unseen.py`, it was pinned to the authors' data location and refuses to run elsewhere.
- `train_development.py`, `development_*`: training of the FedAvg and FedProx models on the development machines only.
- `frozen/`: hash-sealed copies of the comparator code, the FedAvg and FedProx models (Python pickles; load only from sources you trust) and reference files. The manuscript snapshot `frozen/evidence/paper_main.tex` is omitted from this public copy, so the seal check reports it as missing.
- `seal_stage1.py`, `test_stage1.py`, `STAGE1_VALIDATION.md`, `synthetic_*`: static checks and a synthetic test of the runner before execution.
- `results/`: `REPORT.md`, `summary.json` and `per_unit.csv`.
