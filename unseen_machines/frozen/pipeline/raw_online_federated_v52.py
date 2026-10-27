"""V52 one-shot fresh-machine validation of the frozen V51 policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import raw_online_federated_v12 as v12
import raw_online_federated_v29_models as models
import raw_online_federated_v41 as v41
import raw_online_federated_v43 as v43
import raw_online_federated_v43_grid as v43grid
import raw_online_federated_v45_grid as v45grid
import raw_online_federated_v46_machine as machine
import raw_online_federated_v48_ablation as ablation
import raw_online_federated_v51_policy as v51policy


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v52_protocol.json"
PREREGISTRATION = HERE / "V52_PREREGISTRATION.md"
ACQUISITION_AMENDMENT = HERE / "V52_ACQUISITION_AMENDMENT.md"
ACQUISITION_CORRECTION = HERE / "V52_ACQUISITION_CORRECTION.md"
DATA_ROOT = Path(r"E:\masters\datasets\ldm_upb")
MEMBER_ROOT = DATA_ROOT / "v52_members"
V19_B12 = RESULTS / "raw_online_federated_v19" / "feature_streams" / "heldout_test" / "ldm_upb__B12.npz"
MODEL_ROOT = v45grid.v42.V41R1_ROOT
MODEL = MODEL_ROOT / "models" / "all_source.npz"
MODEL_LOCK = MODEL_ROOT / "v41_model_lock.json"
M = 64
FRESH = ("B01", "B02", "B04", "B05")
LAUNCHER_ARTIFACTS = frozenset({"v52_console_transcript.txt"})


class IntegrityError(RuntimeError):
    """Raised when a frozen V52 dependency or phase lock is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - authoritative dataset integrity checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([_clean(row) for row in rows])
    temporary.replace(path)


def specification() -> dict:
    spec = read_json(PROTOCOL)
    selected = tuple(spec["selection"]["selected_in_frozen_order"])
    if selected != FRESH:
        raise IntegrityError(f"implementation/protocol cohort mismatch: {selected}")
    detector = spec["frozen_detector"]
    if int(detector["score_calibration"].split("m=")[-1].split(",")[0]) != M:
        raise IntegrityError("V52 calibration is not m=64")
    policy = detector["v51_policy"]
    if (float(policy["threshold"]), int(policy["holdoff"]), int(policy["persistence_k"]), int(policy["persistence_window"])) != (1.05, 8, 6, 7):
        raise IntegrityError("V52 does not contain the frozen V51 policy")
    return spec


def root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RunId must match [A-Za-z0-9_-]+")
    root = (RESULTS / f"raw_online_federated_v52_{run_id}").resolve()
    if RESULTS.resolve() not in root.parents:
        raise ValueError("V52 result root escaped results/")
    return root


def dependencies() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "preregistration": PREREGISTRATION,
        "acquisition_amendment": ACQUISITION_AMENDMENT,
        "acquisition_correction": ACQUISITION_CORRECTION,
        "driver": Path(__file__).resolve(),
        "tests": HERE / "raw_online_federated_v52_tests.py",
        "launcher": HERE / "start_v52_guarded.ps1",
        "v12_adapter": HERE / "raw_online_federated_v12.py",
        "v43_grid": HERE / "raw_online_federated_v43_grid.py",
        "v48_ablation": HERE / "raw_online_federated_v48_ablation.py",
        "v48_protocol": HERE / "raw_online_federated_v48_protocol.json",
        "v51_policy": HERE / "raw_online_federated_v51_policy.py",
        "v51_protocol": HERE / "raw_online_federated_v51_protocol.json",
        "model_lock": MODEL_LOCK,
        "all_source_model": MODEL,
        "b12_reference_features": V19_B12,
    }


def require_phase(root: Path, phase: str) -> dict:
    audit_path = root / "v52_audit_lock.json"
    if not audit_path.is_file():
        raise IntegrityError("V52 audit phase must run first")
    audit = read_json(audit_path)
    for name, path in dependencies().items():
        if audit["dependency_sha256"].get(name) != sha256(path):
            raise IntegrityError(f"V52 dependency changed after audit: {name}")
    if phase == "audit":
        return audit
    phase_path = root / f"v52_{phase}_lock.json"
    if not phase_path.is_file():
        raise IntegrityError(f"V52 {phase} phase must run first")
    return read_json(phase_path)


def _b12_adapter_parity() -> dict:
    spec = v12.ClientSpec("ldm_upb", "B12", "heldout_test", "run_to_failure", "failed")
    with np.load(V19_B12, allow_pickle=False) as archive:
        reference = np.asarray(archive["features"], dtype=np.float32)
    rows = 0
    mismatch_rows = 0
    maximum = 0.0
    started = time.perf_counter()
    for acquisition in v12.iter_raw_acquisitions(spec):
        if rows >= len(reference):
            raise IntegrityError("B12 raw stream is longer than its frozen cache")
        current = v12.local_preprocess_current_acquisition(acquisition.channels)
        difference = float(np.max(np.abs(current - reference[rows])))
        maximum = max(maximum, difference)
        mismatch_rows += int(not np.array_equal(current, reference[rows]))
        rows += 1
    return {
        "rows": rows,
        "reference_rows": len(reference),
        "exact_mismatch_rows": mismatch_rows,
        "max_absolute_error": maximum,
        "passed": bool(rows == len(reference) and mismatch_rows == 0 and maximum == 0.0),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _metadata_row(path: Path, base: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": int(stat.st_size),
        "created_local": datetime.fromtimestamp(stat.st_ctime).astimezone().isoformat(),
        "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


def _metadata_inventory(paths: list[Path], base: Path) -> dict:
    rows = [_metadata_row(path, base) for path in sorted(paths, key=lambda item: item.as_posix().lower())]
    encoded = "\n".join(f"{row['path']}\t{row['bytes']}" for row in rows).encode("utf-8")
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "filename_size_inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "entries": rows,
    }


def _candidate_cache_snapshot(spec: dict, data_root: Path = DATA_ROOT, results: Path = RESULTS) -> dict:
    bearings = {}
    for bearing in FRESH:
        archive = data_root / spec["archives"][bearing]["file"]
        partial = Path(str(archive) + ".part")
        member_dir = data_root / "v52_members" / bearing
        members = [] if not member_dir.is_dir() else [path for path in member_dir.iterdir() if path.is_file()]
        bearings[bearing] = {
            "complete_monolithic_archive": None if not archive.is_file() else _metadata_row(archive, data_root),
            "partial_monolithic_archive": None if not partial.is_file() else _metadata_row(partial, data_root),
            "production_member_cache": _metadata_inventory(members, data_root),
            "complete_mat_files": sum(path.suffix.lower() == ".mat" for path in members),
            "partial_member_files": sum(path.suffix.lower() == ".part" for path in members),
        }
    probes = []
    if results.is_dir():
        for run_root in results.glob("raw_online_federated_v52_*"):
            probes.extend(path for path in run_root.rglob("*.mat") if path.is_file())
    return {
        "bearing_caches": bearings,
        "retained_mat_transport_probes": _metadata_inventory(probes, results),
        "inspection_scope": "filenames, byte counts, and filesystem timestamps only; MAT values were not parsed",
    }


def _candidate_derived_artifacts(results: Path = RESULTS) -> list[str]:
    exact_names = {
        "v52_acquire_lock.json", "v52_derive_lock.json", "v52_score_lock.json",
        "v52_gate.json", "V52_RESULTS.md", "prediction_ledger.csv",
        "machine_predictions.csv",
    }
    evidence = []
    if not results.is_dir():
        return evidence
    for run_root in results.glob("raw_online_federated_v52_*"):
        for path in run_root.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = {part.lower() for part in path.relative_to(run_root).parts[:-1]}
            if path.name in exact_names or relative_parts.intersection({"feature_streams", "predictions"}):
                evidence.append(path.relative_to(results).as_posix())
    return sorted(evidence)


def audit_phase(root: Path) -> None:
    if root.exists():
        unexpected = sorted(path.name for path in root.iterdir() if path.name not in LAUNCHER_ARTIFACTS)
        if unexpected:
            raise IntegrityError(f"V52 RunId already contains artifacts: {unexpected}")
    missing = [str(path) for path in dependencies().values() if not path.is_file()]
    if missing:
        raise IntegrityError(f"missing V52 dependency: {missing}")
    spec = specification()
    cache_snapshot = _candidate_cache_snapshot(spec)
    derived_artifacts = _candidate_derived_artifacts()
    parity = _b12_adapter_parity()
    if not parity["passed"]:
        raise IntegrityError(f"B12 adapter parity failed: {parity}")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "v52_audit_lock.json", {
        "protocol_sha256": sha256(PROTOCOL),
        "dependency_sha256": {name: sha256(path) for name, path in dependencies().items()},
        "candidate_cache_snapshot": cache_snapshot,
        "candidate_feature_or_prediction_artifacts": derived_artifacts,
        "candidate_feature_or_prediction_artifacts_present": bool(derived_artifacts),
        "candidate_values_parsed_or_scored_by_this_audit": False,
        "historical_candidate_access_provable_from_filesystem": False,
        "adapter_parity": parity,
    })
    print(f"V52 AUDIT ok: {root}", flush=True)


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def _curl_json(url: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "curl.exe", "-sS", "-L", "--fail", "--ssl-no-revoke",
        "--connect-timeout", "60", "--retry", "5", "--retry-all-errors",
        "--output", str(target), url,
    ], check=True)
    return read_json(target)


def _container_entries(root: Path, bearing: str, record: int) -> list[dict]:
    inventory_path = root / "container_inventories" / f"{bearing}.json"
    url = f"https://zenodo.org/api/records/{record}/files/{bearing}.zip/container"
    inventory = _curl_json(url, inventory_path)
    pattern = re.compile(rf"^{bearing}/vibrationData/data_{bearing}_M(\d+)\.mat$", re.I)
    selected = []
    for item in inventory.get("entries", []):
        match = pattern.fullmatch(str(item["key"]))
        if match:
            selected.append({
                "index": int(match.group(1)), "key": str(item["key"]),
                "size": int(item["size"]), "crc": int(item["crc"]),
                "url": str(item["links"]["content"]),
            })
    selected.sort(key=lambda row: row["index"])
    if not selected or [row["index"] for row in selected] != list(range(1, len(selected) + 1)):
        raise IntegrityError(f"non-contiguous Zenodo member inventory: {bearing}")
    return selected


def _member_target(bearing: str, item: dict) -> Path:
    return MEMBER_ROOT / bearing / Path(str(item["key"])).name


def _member_ok(path: Path, item: dict) -> bool:
    return path.is_file() and path.stat().st_size == int(item["size"]) and crc32(path) == int(item["crc"])


def _download_member(bearing: str, item: dict) -> tuple[int, int]:
    target = _member_target(bearing, item)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _member_ok(target, item):
        return int(item["index"]), int(item["size"])
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 7):
        command = [
            "curl.exe", "-sS", "-L", "--fail", "--ssl-no-revoke",
            "--connect-timeout", "60", "--max-time", "600", "--retry", "3",
            "--retry-all-errors", "--output", str(partial), str(item["url"]),
        ]
        try:
            subprocess.run(command, check=True)
            if partial.stat().st_size == int(item["size"]) and crc32(partial) == int(item["crc"]):
                partial.replace(target)
                return int(item["index"]), int(item["size"])
        except (subprocess.CalledProcessError, OSError):
            pass
        print(f"V52 RETRY {bearing} M{int(item['index']):04d} attempt={attempt}", flush=True)
    raise IntegrityError(f"member download/integrity failure: {bearing} M{int(item['index']):04d}")


def acquire_phase(root: Path) -> None:
    require_phase(root, "audit")
    spec = specification()
    manifest = []
    for bearing in FRESH:
        transport_record = int(spec["acquisition_transport"]["transport_record"])
        entries = _container_entries(root, bearing, transport_record)
        total, started = 0, time.perf_counter()
        print(f"V52 DOWNLOAD {bearing}: {len(entries)} authoritative MAT members", flush=True)
        # Four bounded streams amortize the ZIP-container scan cost for later
        # members on the faster immutable transport record. Per-file retries
        # and size/CRC checks make abrupt response closes recoverable.
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_download_member, bearing, item): item for item in entries}
            completed = 0
            for future in as_completed(futures):
                _, size = future.result()
                total += size
                completed += 1
                if completed % 25 == 0 or completed == len(entries):
                    print(f"V52 DOWNLOAD {bearing}: {completed}/{len(entries)} members, {total / 1e9:.2f} GB verified", flush=True)
        logical = [{"index": row["index"], "key": row["key"], "size": row["size"], "crc": row["crc"]} for row in entries]
        manifest_path = root / "member_manifests" / f"{bearing}.json"
        write_json(manifest_path, logical)
        manifest.append({
            "bearing": bearing, "members": len(entries), "uncompressed_bytes": sum(row["size"] for row in entries),
            "member_manifest": str(manifest_path), "member_manifest_sha256": sha256(manifest_path),
            "source_archive_file": spec["archives"][bearing]["file"],
            "source_archive_bytes": spec["archives"][bearing]["bytes"],
            "source_archive_md5": spec["archives"][bearing]["md5"],
        })
        print(f"V52 VERIFIED {bearing}: {len(entries)} members elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_json(root / "v52_acquire_lock.json", {
        "protocol_sha256": sha256(PROTOCOL),
        "logical_member_archives": manifest,
        "all_source_members_size_and_crc_match": True,
        "monolithic_archives_downloaded": False,
        "candidate_outcome_opened": False,
    })
    print(f"V52 ACQUIRE ok: {root}", flush=True)


def _verify_archives(root: Path) -> None:
    lock = require_phase(root, "acquire")
    expected = {row["bearing"]: row for row in lock["logical_member_archives"]}
    for bearing in FRESH:
        manifest = Path(expected[bearing]["member_manifest"])
        if sha256(manifest) != expected[bearing]["member_manifest_sha256"]:
            raise IntegrityError(f"member manifest changed after acquisition: {bearing}")
        entries = read_json(manifest)
        if len(entries) != int(expected[bearing]["members"]):
            raise IntegrityError(f"member count changed after acquisition: {bearing}")
        for item in entries:
            target = _member_target(bearing, item)
            if not target.is_file() or target.stat().st_size != int(item["size"]):
                raise IntegrityError(f"member missing or wrong size after acquisition: {target}")


def _features(root: Path, bearing: str) -> np.ndarray:
    path = root / "feature_streams" / f"ldm_upb__{bearing}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["features"], dtype=np.float32)
            indices = np.asarray(archive["acquisition_index"], dtype=np.int32)
        if not np.array_equal(indices, np.arange(len(values), dtype=np.int32)):
            raise IntegrityError(f"non-causal feature cache: {bearing}")
        return values
    rows = []
    started = time.perf_counter()
    acquire = read_json(root / "v52_acquire_lock.json")
    record = next(row for row in acquire["logical_member_archives"] if row["bearing"] == bearing)
    entries = read_json(Path(record["member_manifest"]))
    for item in entries:
        if int(item["index"]) != len(rows) + 1:
            raise IntegrityError(f"non-causal raw chronology: {bearing}")
        member = _member_target(bearing, item)
        if not _member_ok(member, item):
            raise IntegrityError(f"member integrity changed before feature extraction: {member}")
        channels = v12._ldm_signal(member.read_bytes())  # exact frozen V12 channel reader
        rows.append(v12.local_preprocess_current_acquisition(channels))
        if len(rows) % 250 == 0:
            print(f"V52 FEATURE {bearing}: {len(rows)} acquisitions", flush=True)
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 100 or len(values) <= v43grid.op_start(M):
        raise IntegrityError(f"ineligible feature stream {bearing}: {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=values, acquisition_index=np.arange(len(values), dtype=np.int32))
    temporary.replace(path)
    print(f"V52 FEATURE {bearing}: complete n={len(values)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return values


def _prepare(values: np.ndarray, client_id: str, bundle: dict) -> ablation.PreparedBearing:
    n = len(values)
    center, scale = models.robust_center_scale(values[v45grid.calib.INPUT_CAL])
    normalized = models.normalize(values, center, scale)
    raw = {
        "FED_AE_RECON": bundle["ae"].reconstruction_score(normalized),
        "FED_PCA16": bundle["pca"].score(normalized),
        "FED_DEEP_SVDD": bundle["svdd"].score(normalized),
    }
    zscores = v43grid.zscores_at(raw, M, n)
    decision = ablation.specification()["decision"]
    thresholds = v43grid.epc_moderate(raw, ablation.BRANCH_ORDER, M, float(decision["nominal_z"]), client_id, int(v43.specification()["epc"]["seed"]))
    guard = v43grid.guard_slice(M)
    return ablation.PreparedBearing(client_id, n, M, guard.stop - guard.start, v43grid.op_start(M), zscores, thresholds)


def _first(alarm: np.ndarray) -> int:
    positions = np.flatnonzero(np.asarray(alarm, dtype=bool))
    return int(positions[0]) if len(positions) else -1


def derive_phase(root: Path) -> None:
    _verify_archives(root)
    bundle = v45grid.load_bundle("all_source")
    rows, hashes = [], {}
    for bearing in FRESH:
        values = _features(root, bearing)
        prepared = _prepare(values, f"ldm_upb::{bearing}", bundle)
        ae = machine.fuse_machine([ablation.subset_evidence(prepared, "AE")], "bearing_latch_then_any")
        reference = machine.fuse_machine([ablation.subset_evidence(prepared, "AE_PCA_SVDD")], "bearing_latch_then_any")
        trace = v51policy.apply_policy(np.asarray(ae["score"]), prepared.op_start, threshold=1.05, k=6, window=7, holdoff=8)
        candidate_alarm = np.asarray(trace["alarm"], dtype=bool)
        prediction = root / "machine_predictions" / f"ldm_upb__{bearing}.npz"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            bearing=np.asarray(bearing),
            acquisition_index=np.arange(len(values), dtype=np.int32),
            v51_alarm=candidate_alarm,
            v51_score=np.asarray(trace["score"], dtype=np.float64),
            v48_alarm=np.asarray(reference["alarm"], dtype=bool),
            v48_score=np.asarray(reference["score"], dtype=np.float64),
        )
        temporary.replace(prediction)
        hashes[bearing] = sha256(prediction)
        rows.append({
            "machine_id": f"ldm_upb::{bearing}", "bearing": bearing, "dataset": "ldm_upb", "stream_count": 1,
            "n": len(values), "op_start": prepared.op_start,
            "v51_first_alarm": _first(candidate_alarm), "v48_first_alarm": int(reference["first_alarm"]),
            "prediction_sha256": hashes[bearing],
        })
        print(f"V52 PREDICT {bearing}: n={len(values)} V51={rows[-1]['v51_first_alarm']} V48={rows[-1]['v48_first_alarm']}", flush=True)
    write_csv(root / "v52_prediction_ledger.csv", rows)
    write_json(root / "v52_derive_lock.json", {
        "protocol_sha256": sha256(PROTOCOL), "prediction_hashes": hashes,
        "machines": len(rows), "predictions_locked_before_timing_classification": True,
        "candidate_outcome_or_timing_class_opened": False,
    })
    print(f"V52 DERIVE ok: {root}", flush=True)


def timing(first_alarm: int, n: int) -> str:
    if int(first_alarm) < 0:
        return "missed"
    if int(first_alarm) < int(np.ceil(0.60 * n)):
        return "premature"
    if int(first_alarm) < int(np.floor(0.90 * n)):
        return "valid"
    return "late"


def _summaries(rows: list[dict], prefix: str) -> dict:
    categories = [str(row[f"{prefix}_timing"]) for row in rows]
    return {
        "machines": len(rows),
        "valid": categories.count("valid"), "premature": categories.count("premature"),
        "late": categories.count("late"), "missed": categories.count("missed"),
        "preterminal": sum(int(row[f"{prefix}_first_alarm"]) >= 0 and int(row[f"{prefix}_first_alarm"]) < int(row["terminal"]) for row in rows),
        "early_operation": sum(bool(row[f"{prefix}_early_operation"]) for row in rows),
    }


def score_phase(root: Path) -> None:
    derive = require_phase(root, "derive")
    ledger = list(csv.DictReader((root / "v52_prediction_ledger.csv").open(encoding="utf-8")))
    rows = []
    for base in ledger:
        bearing = base["bearing"]
        prediction = root / "machine_predictions" / f"ldm_upb__{bearing}.npz"
        if sha256(prediction) != derive["prediction_hashes"][bearing]:
            raise IntegrityError(f"prediction changed after lock: {bearing}")
        n, op_start = int(base["n"]), int(base["op_start"])
        terminal = int(np.floor(0.90 * n))
        early_stop = op_start + int(np.floor(0.10 * max(terminal - op_start, 1)))
        v51_first, v48_first = int(base["v51_first_alarm"]), int(base["v48_first_alarm"])
        rows.append({
            "machine_id": base["machine_id"], "bearing": bearing, "n": n, "op_start": op_start,
            "proxy_onset_60pct": int(np.ceil(0.60 * n)), "terminal": terminal, "early_operation_stop": early_stop,
            "v51_first_alarm": v51_first, "v51_timing": timing(v51_first, n),
            "v51_early_operation": bool(0 <= v51_first < early_stop), "v51_lead": n - 1 - v51_first if v51_first >= 0 else None,
            "v48_first_alarm": v48_first, "v48_timing": timing(v48_first, n),
            "v48_early_operation": bool(0 <= v48_first < early_stop), "v48_lead": n - 1 - v48_first if v48_first >= 0 else None,
        })
    write_csv(root / "v52_machine_results.csv", rows)
    v51 = _summaries(rows, "v51")
    v48 = _summaries(rows, "v48")
    improvements = sum(row["v51_timing"] == "valid" and row["v48_timing"] in {"premature", "late", "missed"} for row in rows)
    regressions = sum(row["v48_timing"] == "valid" and row["v51_timing"] != "valid" for row in rows)
    checks = {
        "integrity_all_source_members_size_and_crc_match": bool(read_json(root / "v52_acquire_lock.json")["all_source_members_size_and_crc_match"]),
        "adapter_b12_exact_parity": bool(read_json(root / "v52_audit_lock.json")["adapter_parity"]["passed"]),
        "v51_premature_machines_max": v51["premature"] <= 1,
        "v51_valid_machines_min": v51["valid"] >= 3,
        "v51_preterminal_machines_min": v51["preterminal"] >= 3,
        "v51_early_operation_machines_max": v51["early_operation"] <= 1,
        "v51_premature_not_more_than_v48": v51["premature"] <= v48["premature"],
        "v51_valid_not_less_than_v48": v51["valid"] >= v48["valid"],
        "v51_preterminal_not_less_than_v48": v51["preterminal"] >= v48["preterminal"],
        "at_least_one_strict_timing_improvement_over_v48": improvements >= 1 and regressions == 0,
    }
    gate = {"passed": all(checks.values()), "checks": checks, "v51": v51, "v48": v48, "strict_improvements": improvements, "strict_regressions": regressions}
    report_lines = [
        "# V52 frozen V51 fresh-machine validation", "",
        "**" + ("GATE PASSED" if gate["passed"] else "GATE FAILED") + ".** This is a fresh-machine test within LDM-UPB, not a fresh-dataset confirmation.", "",
        "The policy was frozen before B01/B02/B04/B05 were downloaded. The B12 adapter reproduced all 767 frozen feature rows bit-for-bit. No model training, target adaptation, threshold change, or policy selection occurred.", "",
        "## Machine-level results", "",
        "| Machine | N | Proxy onset | Terminal | V51 first | V51 timing | V48 first | V48 timing |", "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        report_lines.append(f"| {row['bearing']} | {row['n']} | {row['proxy_onset_60pct']} | {row['terminal']} | {row['v51_first_alarm']} | {row['v51_timing']} | {row['v48_first_alarm']} | {row['v48_timing']} |")
    report_lines += [
        "", "## Aggregate", "",
        "| Policy | Valid | Premature | Late | Missed | Preterminal | Early-operation |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| Frozen V51 AE 1.05, holdoff 8, 6-of-7 | {v51['valid']} | {v51['premature']} | {v51['late']} | {v51['missed']} | {v51['preterminal']} | {v51['early_operation']} |",
        f"| Frozen V48 AE+PCA+SVDD, 4-of-7 | {v48['valid']} | {v48['premature']} | {v48['late']} | {v48['missed']} | {v48['preterminal']} | {v48['early_operation']} |",
        "", f"Strict V48-to-V51 timing improvements: {improvements}; regressions: {regressions}.", "",
        "## Gate checks", "",
    ]
    report_lines += [f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in checks.items()]
    report_lines += [
        "", "## Claim boundary", "",
        "The 60% and 90% boundaries are normalized-life proxies, not verified physical degradation onset. Four fresh machines cannot establish broad generalization, and this cohort contains no fully healthy run, so it does not estimate healthy-machine specificity.", "",
    ]
    report = root / "V52_RESULTS.md"
    report.write_text("\n".join(report_lines), encoding="utf-8")
    gate["report_sha256"] = sha256(report)
    write_json(root / "v52_gate.json", gate)
    print(report.read_text(encoding="utf-8"), flush=True)
    print(f"V52 GATE {'PASSED' if gate['passed'] else 'FAILED'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("audit", "acquire", "derive", "score"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = root_for(args.run_id)
    try:
        {"audit": audit_phase, "acquire": acquire_phase, "derive": derive_phase, "score": score_phase}[args.phase](root)
    except Exception as exc:  # preserve a concise phase failure in console transcripts
        print(f"V52 {args.phase.upper()} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
