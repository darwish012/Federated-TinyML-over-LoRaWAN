"""Independent saved-output checks; does not train or regenerate the study."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/published_methods_20261021"


def main():
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert s["complete"] and s["headline_device_parity_passed"]
    assert not s["historical_exact_reproduction_established"]
    assert len(s["cells"]) == 48  # 12 decision rules x 3 + 2x2 factorial x 3
    seeds = (17070, 17071, 17072)
    assert len({(c["part"], c["front_end"], c["rule"], c["seed"]) for c in s["cells"]}) == 48
    for c in s["cells"]:
        rows, m = c["devices"], c["metrics"]
        assert c["seed"] in seeds
        n = 49 if c["part"] == "09d" else 25
        assert len(rows) == len({r["machine_id"] for r in rows}) == m["scored"] == n
        assert not any(r["machine_id"].startswith("ldm_upb::") for r in rows)
        assert m["strict_successes"] == sum(r["strict_success"] for r in rows)
        assert m["healthy_alarm_devices"] == sum(r["healthy_alarm"] for r in rows)
        assert m["faults_detected"] == sum(r["fault_detected"] for r in rows if r["expected"] == "fault")
        for r in rows:
            assert r["strict_success"] == (not r["healthy_alarm"] and (r["fault_detected"] or r["expected"] == "healthy_control"))
            if c["front_end"] == "22feat_h12":
                path = OUT/"scores"/f'h12_flags_seed{c["seed"]}.npz'
                key = r["machine_id"].replace("::", "__")+"__"+c["rule"]
            else:
                path = OUT/"scores"/(r["machine_id"].replace("::", "__")+f'_loges_seed{c["seed"]}.npz')
                key = "flags_"+c["rule"]
            with np.load(path, allow_pickle=False) as z:
                flags = z[key]
            cache = ROOT/"results/healthy_only_suite_20260907/cache"/(r["machine_id"].replace("::", "__")+".npz")
            with np.load(cache, allow_pickle=False) as z:
                h, f = z["healthy_test_indices"].astype(int), z["unhealthy_test_indices"].astype(int)
                assert not np.any(flags[:int(z["calibration_indices"][-1])+1])
            assert bool(np.any(flags[h])) == r["healthy_alarm"]
            assert bool(np.any(flags[f])) == r["fault_detected"]
            assert int(np.sum(flags[h])) == r["healthy_alarm_acquisitions"]
    project = [c["metrics"]["strict_successes"] for c in s["cells"] if c["part"] == "09d" and c["rule"] == "project"]
    assert project == [45, 44, 44], project
    for r in s["logged_comparison"]:
        assert r["matches"] == (r["logged"] == r["new"])
    for p, digest in (s["dependencies_sha256"] | s["artifacts_sha256"]).items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
    for p in (OUT/"loges").glob("*.sources.json"):
        record = json.loads(p.read_text())
        assert record["raw_count"] == record["cached_count"] and record["length_mismatch"] == 0
    assert len(list((OUT/"loges").glob("*.sources.json"))) == 25
    print("PASS: 48 cells, 49/25 identities, independent alarm re-scoring, frozen H12 parity, logged-count comparisons, input and artifact hashes")


if __name__ == "__main__":
    main()
