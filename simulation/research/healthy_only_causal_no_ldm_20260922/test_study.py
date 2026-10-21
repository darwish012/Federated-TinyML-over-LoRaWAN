import json
from pathlib import Path

import numpy as np

import run_study


def _manifest():
    p = Path(run_study.CACHE) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_selected_cohort_is_54_49_43_6_5_and_no_ldm():
    devices = run_study.select_devices(_manifest())
    assert len(devices) == 54
    assert sum(d["truth"] is not None for d in devices) == 43
    assert sum(d["truth_kind"] == "healthy_control" for d in devices) == 6
    assert sum(d["truth"] is None and d["truth_kind"] != "healthy_control" for d in devices) == 5
    assert all(d["dataset"] != "ldm_upb" and not d["machine_id"].startswith("ldm_upb::") for d in devices)


def test_dependency_paths_contain_selected_cache_only():
    manifest = _manifest(); devices = run_study.select_devices(manifest)
    paths = run_study._selected_paths(manifest, devices)
    cache_root = Path(run_study.CACHE).resolve()
    cache_paths = [str(p).lower() for p in paths if p.suffix == ".npz" and p.resolve().parent == cache_root]
    assert len(cache_paths) == 54
    assert all("ldm_upb" not in p for p in cache_paths)
    assert str(Path(run_study.CACHE, "manifest.json")).lower() in [str(p).lower() for p in paths]


def test_selected_cache_integrity_matches_manifest():
    manifest = _manifest(); devices = run_study.select_devices(manifest)
    run_study.verify_selected_cache_integrity(manifest, devices)


def test_combinations_and_evaluation_keep_original_timeline():
    f = {k: np.array([0, 1, 0, 1], dtype=bool) for k in run_study.RULES}
    f.update({"pca": np.array([1, 1, 0, 0], bool), "global_h4": np.array([1, 0, 1, 0], bool), "feature_envelope": np.array([1, 1, 1, 0], bool)})
    np.testing.assert_array_equal(run_study.combine(f, "pca_and_global_h4"), [1, 0, 0, 0])
    meta = {"machine_id": "d::x", "dataset": "d", "truth": 10, "truth_kind": "onset", "training_health_verified": True}
    arrays = {k: np.array(v, dtype=int) for k, v in {"healthy_test_indices": [4, 5], "unhealthy_test_indices": [10, 11], "unscored_test_indices": []}.items()}
    flags = np.zeros(12, bool); flags[10] = True
    row = run_study.evaluate(meta, arrays, flags)
    assert row["strict_success"] and row["fault_delay_acquisitions"] == 0


def test_variant_is_84_fixed_cells_without_running_study():
    assert len(run_study.RULES) == 14
    assert len(run_study.read(run_study.PROTOCOL)["seeds"]) == 3
    assert 14 * 2 * 3 == 84
