"""Sealed prospective LDM-UPB B06/B13/B14/B15 cohort execution.

The workflow is deliberately split at two access boundaries.  No ZIP member
is opened before protocol/execution dependencies are sealed, and the score
phase cannot open the post-lock transition file until every selected device
has either a hash-locked V64/V67 prediction or an immutable transport-failure
lock.  Signal members are streamed directly from their ZIP parts.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
import zipfile
import zlib

import numpy as np
import requests

import external_dual_axis_timing as dual_axis
import raw_online_federated_v12 as v12
import raw_online_federated_v45_grid as v45grid
import raw_online_federated_v52 as v52
import raw_online_federated_v57_policy as v57
import raw_online_federated_v63_policy as v63policy
import raw_online_federated_v64_policy as v64policy
import raw_online_federated_v65_policy as v65policy
import raw_online_federated_v67_policy as v67policy


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "ldm_reserved_cohort_protocol.json"
PARENT_PREREG = HERE / "EXTERNAL_VALIDATION_PREREGISTRATION.json"
TRANSITIONS_AFTER_LOCK = HERE / "ldm_reserved_cohort_transitions_after_lock.json"
ZENODO_API = "https://zenodo.org/api/records/10868257"
DEFAULT_ARCHIVE_ROOT = Path(r"C:\Users\Dell\Desktop\masters\datasets\ldm_reserved_cohort")
MACHINES = ("B06", "B13", "B14", "B15")
CLIENT_IDS = {machine: f"ldm_upb::{machine}" for machine in MACHINES}

V63_ROOT = RESULTS / "raw_online_federated_v63_frozen1"
V64_ROOT = RESULTS / "raw_online_federated_v64_frozen1"
V65_ROOT = RESULTS / "raw_online_federated_v65_frozen1"

EXECUTION_DEPENDENCIES = (
    Path(__file__).resolve(), PROTOCOL, PARENT_PREREG,
    HERE / "external_dual_axis_timing.py",
    HERE / "learning_failure_experiments.py",
    HERE / "raw_online_federated_v12.py",
    HERE / "raw_online_federated_v45_grid.py",
    HERE / "raw_online_federated_v25_policy.py",
    HERE / "raw_online_federated_v27_detector.py",
    HERE / "raw_online_federated_v29_models.py",
    HERE / "raw_online_federated_v30_data.py",
    HERE / "raw_online_federated_v39_comm.py",
    HERE / "raw_online_federated_v41.py",
    HERE / "raw_online_federated_v41_data.py",
    HERE / "raw_online_federated_v41r1_data.py",
    HERE / "raw_online_federated_v42.py",
    HERE / "raw_online_federated_v42_calibration.py",
    HERE / "raw_online_federated_v42_design.py",
    HERE / "raw_online_federated_v42_score.py",
    HERE / "raw_online_federated_v43.py",
    HERE / "raw_online_federated_v43_cohort.py",
    HERE / "raw_online_federated_v43_comms.py",
    HERE / "raw_online_federated_v43_grid.py",
    HERE / "raw_online_federated_v43_stats.py",
    HERE / "raw_online_federated_v45_cohort.py",
    HERE / "raw_online_federated_v46_machine.py",
    HERE / "raw_online_federated_v48_ablation.py",
    HERE / "raw_online_federated_v51_policy.py",
    HERE / "raw_online_federated_v52.py",
    HERE / "raw_online_federated_v57_policy.py",
    HERE / "raw_online_federated_v62_policy.py",
    HERE / "raw_online_federated_v63_policy.py",
    HERE / "raw_online_federated_v64_policy.py",
    HERE / "raw_online_federated_v65_policy.py",
    HERE / "raw_online_federated_v67_policy.py",
    HERE / "run_full_simulation.py",
    HERE / "raw_online_federated_v41_protocol.DRAFT.json",
    HERE / "raw_online_federated_v43_protocol.json",
    HERE / "raw_online_federated_v48_protocol.json",
    HERE / "results" / "raw_online_federated_v41r1_confirm1" / "models" / "all_source.npz",
    V63_ROOT / "v63_development_lock.json",
    V64_ROOT / "v64_development_lock.json",
    V65_ROOT / "v65_development_lock.json",
)


class IntegrityError(RuntimeError):
    pass


class TransitionNotReadyError(IntegrityError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hashes(path: Path) -> dict[str, Any]:
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            size += len(block); sha.update(block); md5.update(block)
    return {"bytes": size, "md5": md5.hexdigest().lower(), "sha256": sha.hexdigest().upper()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, np.bool_): return bool(item)
        if isinstance(item, np.integer): return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise ValueError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must match [A-Za-z0-9_-]+")
    return RESULTS / f"ldm_reserved_cohort_external_{run_id}"


def specification() -> dict[str, Any]:
    spec = read_json(PROTOCOL)
    selection = spec["selection"]
    if tuple(selection["machines"]) != MACHINES:
        raise IntegrityError("reserved cohort identity changed")
    if tuple(selection["client_ids"]) != tuple(CLIENT_IDS[m] for m in MACHINES):
        raise IntegrityError("exact client IDs changed")
    entries = spec["archive_source_snapshot"]["machines"]
    parts = [part for machine in MACHINES for part in entries[machine]["parts"]]
    if len(parts) != 29 or sum(int(p["size"]) for p in parts) != 99_945_886_710:
        raise IntegrityError("official 29-part cohort manifest changed")
    if spec["parent_preregistration"]["sha256"].upper() != sha256(PARENT_PREREG):
        raise IntegrityError("parent preregistration changed")
    text = PROTOCOL.read_text(encoding="utf-8")
    forbidden = ("published_transition_index", "transition_values", "onset_index")
    if any(word in text for word in forbidden):
        raise IntegrityError("selection protocol contains a transition/onset value field")
    return spec


def parts_for(spec: dict[str, Any], machine: str) -> list[dict[str, Any]]:
    return list(spec["archive_source_snapshot"]["machines"][machine]["parts"])


def _assert_transition_absent() -> None:
    if TRANSITIONS_AFTER_LOCK.exists():
        raise IntegrityError("post-lock transition file exists before cohort prediction barrier")


def protocol_seal_phase(root: Path, archive_root: Path) -> None:
    _assert_transition_absent()
    spec = specification()
    root.mkdir(parents=True, exist_ok=True)
    seal = root / "protocol_seal_lock.json"
    if seal.exists(): raise IntegrityError(f"protocol already sealed: {seal}")
    if any(root.iterdir()): raise IntegrityError(f"results root is not empty: {root}")
    protected = (Path(__file__).resolve(), PROTOCOL, PARENT_PREREG)
    missing = [str(path) for path in protected if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    write_json(seal, {
        "schema": "ldm_reserved_cohort_protocol_seal_v1", "sealed_utc": _utc(),
        "machines": list(MACHINES), "client_ids": [CLIENT_IDS[m] for m in MACHINES],
        "archive_root": str(archive_root.resolve()),
        "official_parts": sum(len(parts_for(spec, m)) for m in MACHINES),
        "official_bytes": sum(int(p["size"]) for m in MACHINES for p in parts_for(spec, m)),
        "protected_sha256": {str(path): sha256(path) for path in protected},
        "signal_payload_opened": False, "transition_or_onset_opened": False,
        "transition_file_absent": True,
    })
    print(f"LDM COHORT PROTOCOL SEALED: {seal}", flush=True)


def require_protocol_seal(root: Path) -> dict[str, Any]:
    seal = root / "protocol_seal_lock.json"
    if not seal.is_file(): raise IntegrityError("run --phase protocol-seal first")
    lock = read_json(seal)
    for name, digest in lock["protected_sha256"].items():
        if sha256(Path(name)) != digest: raise IntegrityError(f"sealed dependency changed: {name}")
    specification()
    return lock


def _live_metadata() -> dict[str, dict[str, Any]]:
    response = requests.get(ZENODO_API, timeout=90)
    response.raise_for_status()
    output = {}
    for row in response.json()["files"]:
        output[str(row["key"])] = {
            "size": int(row["size"]),
            "md5": str(row["checksum"]).removeprefix("md5:").lower(),
            "url": str(row["links"]["self"]),
        }
    return output


def _download_one(part: dict[str, Any], remote: dict[str, Any], archive_root: Path) -> dict[str, Any]:
    target = archive_root / str(part["file"])
    expected_size, expected_md5 = int(part["size"]), str(part["md5"]).lower()
    if target.is_file():
        observed = _hashes(target)
        if observed["bytes"] == expected_size and observed["md5"] == expected_md5:
            return {"file": target.name, **observed, "downloaded": False}
        raise IntegrityError(f"existing final archive fails official size/MD5: {target}")
    temporary = target.with_suffix(target.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected_size:
        temporary.unlink(); offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requests.get(remote["url"], headers=headers, stream=True, timeout=(90, 600))
    if offset and response.status_code != 206:
        response.close(); temporary.unlink(); offset = 0
        response = requests.get(remote["url"], stream=True, timeout=(90, 600))
    response.raise_for_status()
    mode = "ab" if offset else "wb"
    with response, temporary.open(mode) as handle:
        for block in response.iter_content(chunk_size=8 * 1024 * 1024):
            if block: handle.write(block)
    observed = _hashes(temporary)
    if observed["bytes"] != expected_size or observed["md5"] != expected_md5:
        raise IntegrityError(f"download fails official size/MD5: {target.name}")
    temporary.replace(target)
    return {"file": target.name, **observed, "downloaded": True}


def download_phase(root: Path, archive_root: Path, workers: int = 3) -> None:
    require_protocol_seal(root); _assert_transition_absent()
    spec, remote = specification(), _live_metadata()
    jobs = []
    for machine in MACHINES:
        for part in parts_for(spec, machine):
            live = remote.get(str(part["file"]))
            if live is None or live["size"] != int(part["size"]) or live["md5"] != str(part["md5"]).lower():
                raise IntegrityError(f"live official metadata differs from frozen manifest: {part['file']}")
            jobs.append((machine, part, live))
    archive_root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "download_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"archives": {}}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 6))) as executor:
        futures = {executor.submit(_download_one, part, live, archive_root): (machine, part) for machine, part, live in jobs}
        for future in as_completed(futures):
            machine, part = futures[future]
            row = future.result(); row["machine"] = machine
            progress["archives"][str(part["file"])] = row
            write_json(progress_path, progress)
            print(f"LDM ARCHIVE READY {machine} {part['file']}: official size+MD5 and local SHA256 locked", flush=True)
    rows = [progress["archives"].get(str(part["file"])) for _, part, _ in jobs]
    if any(row is None for row in rows): raise IntegrityError("download did not lock all 29 parts")
    write_json(root / "download_lock.json", {
        "schema": "ldm_reserved_download_lock_v1", "completed_utc": _utc(),
        "archive_root": str(archive_root.resolve()), "archives": rows,
        "official_part_count": 29, "official_bytes": 99_945_886_710,
        "live_zenodo_metadata_matched_frozen_manifest": True,
        "all_official_size_md5_and_local_sha256_locked": True,
        "signal_payload_opened": False, "transition_or_onset_opened": False,
    })


def require_download(root: Path) -> dict[str, Any]:
    require_protocol_seal(root)
    path = root / "download_lock.json"
    if not path.is_file(): raise IntegrityError("run --phase download first")
    lock = read_json(path)
    if len(lock.get("archives", [])) != 29: raise IntegrityError("download lock is incomplete")
    return lock


def _download_rows(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["file"]): row for row in require_download(root)["archives"]}


def verify_archive(root: Path, archive_root: Path, part: dict[str, Any]) -> dict[str, Any]:
    """Recheck official size+MD5 and locked local SHA256 immediately before access."""
    locked = _download_rows(root).get(str(part["file"]))
    if locked is None: raise IntegrityError(f"archive not present in download lock: {part['file']}")
    path = archive_root / str(part["file"])
    if not path.is_file(): raise IntegrityError(f"archive missing: {path}")
    observed = _hashes(path)
    if (observed["bytes"] != int(part["size"]) or
            observed["md5"] != str(part["md5"]).lower() or
            observed["sha256"] != str(locked["sha256"]).upper()):
        raise IntegrityError(f"official/local archive integrity mismatch before access: {path.name}")
    return observed


def _member_rows(machine: str, filename: str, archive_root: Path) -> list[dict[str, Any]]:
    stem = Path(filename).stem
    pattern = re.compile(rf"^{re.escape(stem)}/vibrationData/data_{machine}_M(\d+)\.mat$", re.I)
    rows = []
    with zipfile.ZipFile(archive_root / filename, "r") as handle:
        for item in handle.infolist():
            match = pattern.fullmatch(item.filename)
            if match:
                rows.append({
                    "index": int(match.group(1)), "key": item.filename,
                    "size": int(item.file_size), "compressed_size": int(item.compress_size),
                    "crc32": int(item.CRC), "compression": int(item.compress_type),
                    "source_part": filename,
                })
    return rows


def validate_member_union(machine: str, rows: list[dict[str, Any]]) -> None:
    if not rows: raise IntegrityError(f"empty member union: {machine}")
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)) or indices != list(range(1, len(rows) + 1)):
        raise IntegrityError(f"member union is not unique/contiguous from M00001: {machine}")


def inventory_phase(root: Path, archive_root: Path) -> None:
    require_download(root); _assert_transition_absent(); spec = specification()
    records = []
    for machine in MACHINES:
        rows = []
        for part in parts_for(spec, machine):
            verify_archive(root, archive_root, part)
            rows.extend(_member_rows(machine, str(part["file"]), archive_root))
        rows.sort(key=lambda row: row["index"]); validate_member_union(machine, rows)
        path = root / "inventories" / f"{machine}.json"; write_json(path, rows)
        records.append({
            "machine": machine, "client_id": CLIENT_IDS[machine], "members": len(rows),
            "first": rows[0]["index"], "last": rows[-1]["index"],
            "member_bytes": sum(int(row["size"]) for row in rows),
            "inventory_sha256": sha256(path),
        })
        print(f"LDM INVENTORY {machine}: {len(rows)} unique contiguous members", flush=True)
    write_json(root / "inventory_lock.json", {
        "machines": records, "all_machine_unions_unique_and_contiguous": True,
        "whole_archives_verified_before_central_directory_access": True,
        "transition_or_onset_opened": False,
    })


def require_inventory(root: Path) -> dict[str, Any]:
    require_download(root)
    path = root / "inventory_lock.json"
    if not path.is_file(): raise IntegrityError("run --phase inventory first")
    lock = read_json(path)
    for row in lock["machines"]:
        inventory = root / "inventories" / f"{row['machine']}.json"
        if sha256(inventory) != row["inventory_sha256"]:
            raise IntegrityError(f"member inventory changed: {row['machine']}")
    return lock


def conformance_phase(root: Path, archive_root: Path) -> None:
    inventory = require_inventory(root); _assert_transition_absent(); spec = specification()
    rows = []
    supported = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
    for record in inventory["machines"]:
        machine = str(record["machine"])
        for part in parts_for(spec, machine): verify_archive(root, archive_root, part)
        members = read_json(root / "inventories" / f"{machine}.json")
        validate_member_union(machine, members)
        bad = sum(int(row["size"]) <= 0 or int(row["compression"]) not in supported for row in members)
        if bad: raise IntegrityError(f"unsupported/empty members for {machine}: {bad}")
        rows.append({
            "machine": machine, "client_id": CLIENT_IDS[machine], "members": len(members),
            "member_pattern_conforms": True, "compression_supported": True,
            "decoder": "raw_online_federated_v12._ldm_signal",
            "preprocessor": "raw_online_federated_v12.local_preprocess_current_acquisition",
            "signal_member_opened": False,
        })
    write_json(root / "conformance_lock.json", {
        "machines": rows, "decoder_mapping_inherited_only_from_burned_LDM_code": True,
        "no_reserved_signal_member_opened": True, "transition_or_onset_opened": False,
    })


def require_conformance(root: Path) -> dict[str, Any]:
    require_inventory(root)
    path = root / "conformance_lock.json"
    if not path.is_file(): raise IntegrityError("run --phase conformance first")
    return read_json(path)


def _frozen_candidates() -> tuple[v63policy.Candidate, v64policy.Candidate, v65policy.Candidate]:
    locks = [read_json(path) for path in EXECUTION_DEPENDENCIES[-3:]]
    if any(lock.get("status") != "development_passed_policy_frozen" for lock in locks):
        raise IntegrityError("V63/V64/V65 frozen dependency is not passed")
    c63 = v63policy.Candidate(**locks[0]["selected_candidate"])
    c64 = v64policy.Candidate(**locks[1]["selected_candidate"])
    c65 = v65policy.Candidate(**locks[2]["selected_candidate"])
    if c64.watch_reference_gap != 40: raise IntegrityError("V64 active candidate is not frozen gap 40")
    v67policy.Candidate().validate()
    return c63, c64, c65


def execution_seal_phase(root: Path) -> None:
    require_conformance(root); _assert_transition_absent()
    path = root / "execution_seal_lock.json"
    if path.exists(): raise IntegrityError("execution already sealed")
    missing = [str(source) for source in EXECUTION_DEPENDENCIES if not source.is_file()]
    if missing: raise FileNotFoundError(missing)
    c63, c64, c65 = _frozen_candidates()
    write_json(path, {
        "schema": "ldm_reserved_execution_seal_v1", "sealed_utc": _utc(),
        "machines": list(MACHINES), "client_ids": [CLIENT_IDS[m] for m in MACHINES],
        "dependencies": [{"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)} for source in EXECUTION_DEPENDENCIES],
        "v63_candidate": c63.__dict__, "v64_active_candidate": c64.__dict__,
        "v65_internal_base_candidate_for_v67": c65.__dict__, "v67_shadow_candidate": v67policy.Candidate().__dict__,
        "reserved_signal_payload_opened": False, "transition_or_onset_opened": False,
        "threshold_tuning": False, "rescue_allowed": False,
    })


def require_execution_seal(root: Path) -> dict[str, Any]:
    require_conformance(root)
    path = root / "execution_seal_lock.json"
    if not path.is_file(): raise IntegrityError("run --phase execution-seal first")
    lock = read_json(path)
    for row in lock["dependencies"]:
        source = Path(row["path"])
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise IntegrityError(f"execution dependency changed after seal: {source}")
    return lock


def _feature_path(root: Path, machine: str) -> Path:
    return root / "features" / f"ldm_upb__{machine}.npz"


def features_phase(root: Path, archive_root: Path, selected_machine: str | None = None) -> None:
    require_execution_seal(root); _assert_transition_absent(); spec = specification()
    machines = MACHINES if selected_machine is None else (selected_machine,)
    if any(machine not in MACHINES for machine in machines): raise ValueError("invalid reserved machine")
    for machine in machines:
        final = _feature_path(root, machine); lock_path = root / "feature_locks" / f"{machine}.json"
        if final.exists() and lock_path.exists():
            if read_json(lock_path)["feature_sha256"] != sha256(final): raise IntegrityError(f"feature changed: {machine}")
            print(f"LDM FEATURE {machine}: already locked", flush=True); continue
        members = read_json(root / "inventories" / f"{machine}.json")
        checkpoint = root / "feature_progress" / f"{machine}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as data: output = [x.copy() for x in np.asarray(data["features"], dtype=np.float32)]
        else: output = []
        current_part, handle = None, None
        try:
            for row in members[len(output):]:
                if row["source_part"] != current_part:
                    if handle is not None: handle.close()
                    part = next(p for p in parts_for(spec, machine) if p["file"] == row["source_part"])
                    verify_archive(root, archive_root, part)
                    handle = zipfile.ZipFile(archive_root / row["source_part"], "r"); current_part = row["source_part"]
                payload = handle.read(row["key"])
                if len(payload) != int(row["size"]) or (zlib.crc32(payload) & 0xFFFFFFFF) != int(row["crc32"]):
                    raise IntegrityError(f"member CRC mismatch at decode: {machine} {row['key']}")
                values = v12.local_preprocess_current_acquisition(v12._ldm_signal(payload))
                if values.shape != (100,) or not np.all(np.isfinite(values)):
                    raise IntegrityError(f"bad frozen feature vector: {machine} {row['index']}")
                output.append(values)
                if len(output) % 100 == 0 or len(output) == len(members):
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    temporary = checkpoint.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, features=np.asarray(output, dtype=np.float32))
                    temporary.replace(checkpoint)
                    print(f"LDM FEATURE {machine}: {len(output)}/{len(members)}", flush=True)
        finally:
            if handle is not None: handle.close()
        matrix = np.asarray(output, dtype=np.float32)
        if matrix.shape != (len(members), 100) or not np.all(np.isfinite(matrix)):
            raise IntegrityError(f"bad final feature matrix: {machine} {matrix.shape}")
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, client_id=np.asarray(CLIENT_IDS[machine]), acquisition_index=np.arange(len(matrix), dtype=np.int32), features=matrix)
        temporary.replace(final)
        write_json(lock_path, {
            "machine": machine, "client_id": CLIENT_IDS[machine], "n": len(matrix),
            "feature_sha256": sha256(final), "direct_zip_streaming": True,
            "all_members_crc32_verified_at_decode": True, "transition_or_onset_opened": False,
        })


def _first(values: np.ndarray) -> int:
    found = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(found[0]) if len(found) else -1


def _predict_one(root: Path, machine: str, execution: dict[str, Any]) -> dict[str, Any]:
    feature_path = _feature_path(root, machine); feature_lock = read_json(root / "feature_locks" / f"{machine}.json")
    if feature_lock["feature_sha256"] != sha256(feature_path): raise IntegrityError(f"feature changed: {machine}")
    with np.load(feature_path, allow_pickle=False) as data: values = np.asarray(data["features"], dtype=np.float32)
    bundle = v45grid.load_bundle("all_source")
    prepared = v52._prepare(values, CLIENT_IDS[machine], bundle)
    ae = v52.machine.fuse_machine([v52.ablation.subset_evidence(prepared, "AE")], "bearing_latch_then_any")
    reference = v52.machine.fuse_machine([v52.ablation.subset_evidence(prepared, "AE_PCA_SVDD")], "bearing_latch_then_any")
    ae_score, v48_alarm = np.asarray(ae["score"], dtype=np.float64), np.asarray(reference["alarm"], dtype=bool)
    trace = v57.evaluate_machine([values], ae_score, v48_alarm)
    c63 = v63policy.Candidate(**execution["v63_candidate"])
    c64 = v64policy.Candidate(**execution["v64_active_candidate"])
    c65 = v65policy.Candidate(**execution["v65_internal_base_candidate_for_v67"])
    alarm63, adaptive = v63policy.candidate_alarm(trace.impulse_trend, ae_score, trace.raw_watch_evidence, trace.final_alarm, trace.collapse_route, trace.clip_fraction, c63)
    alarm64 = v64policy.candidate_alarm(alarm63, trace.raw_watch_evidence, v48_alarm, trace.collapse_route, c64)
    alarm65 = v65policy.candidate_alarm(alarm64, trace.raw_watch_evidence, trace.collapse_route, trace.clip_fraction, c65)
    alarm67, early67, triad67, short67 = v67policy.candidate_alarm(alarm65, [values], ae_score, v48_alarm, trace.clip_fraction, v67policy.Candidate())
    path = root / "predictions" / f"ldm_upb__{machine}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, client_id=np.asarray(CLIENT_IDS[machine]), acquisition_index=np.arange(len(values), dtype=np.int32), ae_continuous_score=ae_score, v48_alarm=v48_alarm, impulse_trend=trace.impulse_trend, raw_watch_evidence=trace.raw_watch_evidence, v57_alarm=trace.final_alarm, v63_alarm=alarm63, v64_alarm=alarm64, v65_alarm=alarm65, v67_alarm=alarm67, v67_early=early67, v67_triad=triad67, v67_short_change=short67, clip_fraction=np.asarray(trace.clip_fraction), collapse_route=np.asarray(trace.collapse_route), v63_adaptive_threshold=np.asarray(np.nan if adaptive is None else adaptive))
    temporary.replace(path)
    finite = np.all(np.isfinite(ae_score)) and np.all(np.isfinite(trace.impulse_trend[v57.RECENT + v57.PRIOR - 1:]))
    lock = {
        "machine": machine, "client_id": CLIENT_IDS[machine], "n": len(values),
        "feature_sha256": feature_lock["feature_sha256"], "prediction_sha256": sha256(path),
        "v64_first_alarm": _first(alarm64), "v64_alarm_count": int(np.sum(alarm64)),
        "v67_first_alarm": _first(alarm67), "v67_alarm_count": int(np.sum(alarm67)),
        "v64_v67_disagreement_count": int(np.sum(alarm64 != alarm67)),
        "insufficient_history_count": min(len(values), v57.RECENT + v57.PRIOR - 1),
        "numerical_failure_count": 0 if finite else 1,
        "clip_fraction": float(trace.clip_fraction), "collapse_route": bool(trace.collapse_route),
        "prediction_locked_before_any_transition_or_onset": True,
    }
    write_json(root / "prediction_locks" / f"{machine}.json", lock)
    print(f"LDM PREDICT {machine}: V64={lock['v64_first_alarm']} V67={lock['v67_first_alarm']}", flush=True)
    return lock


def _transport_failure_path(root: Path, machine: str) -> Path:
    return root / "transport_failure_locks" / f"{machine}.json"


def _validated_prediction_lock(root: Path, machine: str) -> dict[str, Any] | None:
    lock_path, prediction = root / "prediction_locks" / f"{machine}.json", root / "predictions" / f"ldm_upb__{machine}.npz"
    if not lock_path.is_file(): return None
    lock = read_json(lock_path)
    if lock.get("client_id") != CLIENT_IDS[machine] or not prediction.is_file() or lock.get("prediction_sha256") != sha256(prediction):
        raise IntegrityError(f"invalid prediction lock: {machine}")
    return lock


def predict_all_phase(root: Path) -> None:
    execution = require_execution_seal(root); _assert_transition_absent()
    locks = []
    for machine in MACHINES:
        existing = _validated_prediction_lock(root, machine)
        if existing is not None: locks.append(existing); continue
        failure = _transport_failure_path(root, machine)
        if failure.is_file(): continue
        feature = _feature_path(root, machine)
        if not feature.is_file(): raise IntegrityError(f"features not locked for {machine}")
        locks.append(_predict_one(root, machine, execution))
    states = []
    for machine in MACHINES:
        prediction = _validated_prediction_lock(root, machine)
        failure_path = _transport_failure_path(root, machine)
        if prediction is not None:
            states.append({"machine": machine, "client_id": CLIENT_IDS[machine], "state": "prediction_locked", "lock_sha256": sha256(root / "prediction_locks" / f"{machine}.json"), "prediction_sha256": prediction["prediction_sha256"]})
        elif failure_path.is_file():
            states.append({"machine": machine, "client_id": CLIENT_IDS[machine], "state": "transport_failure_locked", "lock_sha256": sha256(failure_path)})
        else: raise IntegrityError(f"cohort barrier incomplete: {machine}")
    write_json(root / "cohort_prediction_barrier_lock.json", {
        "schema": "ldm_reserved_cohort_prediction_barrier_v1", "locked_utc": _utc(),
        "machines": states, "all_four_accounted_for": len(states) == 4,
        "predictions": sum(row["state"] == "prediction_locked" for row in states),
        "transport_failures": sum(row["state"] == "transport_failure_locked" for row in states),
        "transition_or_onset_opened": False, "transition_file_absent_when_barrier_locked": True,
    })


def require_barrier(root: Path) -> dict[str, Any]:
    """Validate the whole-cohort barrier without touching the transition path."""
    require_execution_seal(root)
    path = root / "cohort_prediction_barrier_lock.json"
    if not path.is_file(): raise IntegrityError("cohort prediction barrier is not locked")
    lock = read_json(path)
    rows = lock.get("machines", [])
    if len(rows) != 4 or {row.get("machine") for row in rows} != set(MACHINES):
        raise IntegrityError("cohort prediction barrier does not cover exactly all four devices")
    for row in rows:
        machine = row["machine"]
        if row["state"] == "prediction_locked":
            if _validated_prediction_lock(root, machine) is None: raise IntegrityError(f"prediction disappeared: {machine}")
            source = root / "prediction_locks" / f"{machine}.json"
        elif row["state"] == "transport_failure_locked": source = _transport_failure_path(root, machine)
        else: raise IntegrityError(f"invalid barrier state: {machine}")
        if not source.is_file() or sha256(source) != row["lock_sha256"]:
            raise IntegrityError(f"barrier-bound lock changed: {machine}")
    return lock


def classify_transition(first: int, n: int, onset: int, tolerance: float = 0.10) -> str:
    margin = int(math.floor(float(tolerance) * int(n)))
    if first < 0: return "missed"
    if first < onset - margin: return "premature"
    if first <= onset + margin: return "valid"
    return "late"


def score_phase(root: Path) -> None:
    barrier = require_barrier(root)  # deliberately before existence/read of transition file
    if not TRANSITIONS_AFTER_LOCK.is_file():
        raise TransitionNotReadyError("cohort predictions are locked; create the post-lock transition file bound to this barrier")
    timing = read_json(TRANSITIONS_AFTER_LOCK)
    if str(timing.get("cohort_barrier_lock_sha256", "")).upper() != sha256(root / "cohort_prediction_barrier_lock.json"):
        raise IntegrityError("transition file is not bound to this cohort barrier")
    transitions = timing.get("transitions")
    if not isinstance(transitions, dict) or set(transitions) != set(MACHINES):
        raise IntegrityError("post-lock transition file must report exactly all four selected machines")
    rows, primary = [], []
    spec = specification()
    for state in barrier["machines"]:
        machine, onset = state["machine"], int(transitions[state["machine"]])
        archive_parts = parts_for(spec, machine); archive_bytes = sum(int(p["size"]) for p in archive_parts)
        if state["state"] == "transport_failure_locked":
            rows.append({"machine": machine, "client_id": CLIENT_IDS[machine], "transport_status": "failed_locked", "archive_parts": len(archive_parts), "archive_bytes": archive_bytes, "published_transition": onset, "v64_timing_class": "transport_failure", "v67_timing_class": "transport_failure"})
            primary.append("missed"); continue
        lock = _validated_prediction_lock(root, machine); assert lock is not None
        n = int(lock["n"])
        if onset < 0 or onset >= n: raise IntegrityError(f"transition outside stream: {machine}")
        prediction = root / "predictions" / f"ldm_upb__{machine}.npz"
        with np.load(prediction, allow_pickle=False) as data:
            alarm64, alarm67 = np.asarray(data["v64_alarm"], bool), np.asarray(data["v67_alarm"], bool)
        first64, first67 = int(lock["v64_first_alarm"]), int(lock["v67_first_alarm"])
        class64, class67 = classify_transition(first64, n, onset), classify_transition(first67, n, onset)
        dual64, dual67 = dual_axis.evaluate(first64, onset, n), dual_axis.evaluate(first67, onset, n)
        row = {
            "machine": machine, "client_id": CLIENT_IDS[machine], "transport_status": "passed",
            "archive_parts": len(archive_parts), "archive_bytes": archive_bytes, "acquisitions": n,
            "feature_sha256": lock["feature_sha256"], "prediction_sha256": lock["prediction_sha256"],
            "published_transition": onset, "timing_tolerance": int(math.floor(.10 * n)),
            "v64_first_alarm": first64, "v64_alarm_count": int(np.sum(alarm64)), "v64_alarm_fraction": float(np.mean(alarm64)), "v64_timing_class": class64,
            "v67_first_alarm": first67, "v67_alarm_count": int(np.sum(alarm67)), "v67_alarm_fraction": float(np.mean(alarm67)), "v67_timing_class": class67,
            "v64_v67_disagreement_count": int(np.sum(alarm64 != alarm67)),
            "v64_degradation_axis": dual64["degradation_axis"], "v64_maintenance_axis": dual64["maintenance_axis"], "v64_predictive_timing": dual64["predictive_timing"],
            "v67_degradation_axis": dual67["degradation_axis"], "v67_maintenance_axis": dual67["maintenance_axis"], "v67_predictive_timing": dual67["predictive_timing"],
            "insufficient_history_count": lock["insufficient_history_count"], "numerical_failure_count": lock["numerical_failure_count"],
            "clip_fraction": lock["clip_fraction"], "collapse_route": lock["collapse_route"],
        }
        rows.append(row); primary.append(class67)
    gate_spec = read_json(PARENT_PREREG)["promotion_cohort"]["gate"]
    gate = {
        "premature_max": primary.count("premature") <= int(gate_spec["premature_max"]),
        "valid_min": primary.count("valid") >= int(gate_spec["valid_min"]),
        "late_plus_missed_max": primary.count("late") + primary.count("missed") <= int(gate_spec["late_plus_missed_max"]),
        "all_devices_reported": len(rows) == 4,
    }
    write_csv(root / "full_per_device_statistics.csv", rows)
    write_json(root / "score_lock.json", {
        "classification": "prospective_same_dataset_reserved_cohort", "active": "V64", "shadow": "V67",
        "v67_promotion_gate_passed": all(gate.values()), "gate_checks": gate,
        "v67_primary_counts": {name: primary.count(name) for name in ("valid", "premature", "late", "missed")},
        "devices_reported": len(rows), "barrier_lock_sha256": sha256(root / "cohort_prediction_barrier_lock.json"),
        "transition_file_sha256": sha256(TRANSITIONS_AFTER_LOCK), "statistics_sha256": sha256(root / "full_per_device_statistics.csv"),
        "threshold_or_policy_changed": False, "rescue_or_device_drop_allowed": False,
    })
    print(f"LDM COHORT SCORE: V67 promotion gate={'PASS' if all(gate.values()) else 'FAIL'} counts={primary}", flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("protocol-seal", "download", "inventory", "conformance", "execution-seal", "features", "predict-all", "score"))
    parser.add_argument("--run-id", default="frozen1")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--machine", choices=MACHINES)
    args = parser.parse_args(argv); root = root_for(args.run_id); archive_root = args.archive_root.resolve()
    try:
        if args.phase == "protocol-seal": protocol_seal_phase(root, archive_root)
        elif args.phase == "download": download_phase(root, archive_root, args.download_workers)
        elif args.phase == "inventory": inventory_phase(root, archive_root)
        elif args.phase == "conformance": conformance_phase(root, archive_root)
        elif args.phase == "execution-seal": execution_seal_phase(root)
        elif args.phase == "features": features_phase(root, archive_root, args.machine)
        elif args.phase == "predict-all": predict_all_phase(root)
        else: score_phase(root)
    except TransitionNotReadyError as exc:
        print(f"LDM COHORT TRANSITION STOP: {exc}", flush=True)
    except Exception as exc:
        print(f"LDM COHORT FAILED: {exc}", file=sys.stderr, flush=True); raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
