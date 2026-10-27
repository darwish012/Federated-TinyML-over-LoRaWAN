"""Static-only validation. Never import/execute the runner or open target files.

The sole executable runner fragment is its isolated argument-parser definition,
explicitly permitted for this amendment. All analysis definitions remain AST only.
"""
import argparse
import ast
import contextlib
import encodings.unicode_escape  # Used lazily by ast.unparse, not target decoding.
import hashlib
import io
import json
import locale  # Preload argparse's lazy standard-library dependencies before guard.
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parent


def deny_external_access(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        path = Path(args[0]).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError(f"Stage-1 validation forbids external open: {path}")
    if event.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.spawn")):
        raise RuntimeError(f"Stage-1 validation forbids: {event}")


sys.addaudithook(deny_external_access)
source = (ROOT / "run_unseen.py").read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
assert "parse_args" in functions
parser_node = functions["parse_args"]
# Only argparse and Path are available; no runner globals, main, file functions,
# imports, model code or dataset adapters enter this isolated namespace.
parser_tree = ast.Module(body=[parser_node], type_ignores=[])
namespace = {"argparse": argparse, "Path": Path, "__doc__": "Isolated parser dry run"}
exec(compile(parser_tree, "<isolated argument parser only>", "exec"), namespace)
parse = namespace["parse_args"]
for mode in ("--metadata-only", "--run"):
    args = parse(["--protocol", "protocol.json", mode])
    assert args.protocol == Path("protocol.json")
    assert args.metadata_only == (mode == "--metadata-only")
    assert args.run == (mode == "--run")
for invalid in ([], ["--protocol", "protocol.json"],
                ["--protocol", "protocol.json", "--run", "--metadata-only"]):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parse(invalid)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("Invalid CLI was accepted")
print("PASS: isolated argument parsing (2 valid modes, 3 invalid cases); main not executed")


def calls(name):
    return [ast.unparse(n.func) for n in ast.walk(functions[name]) if isinstance(n, ast.Call)]


# Static dependency/flow checks, not a simulation of analysis.
assert calls("parse_args") == ["argparse.ArgumentParser", "parser.add_argument",
                              "parser.add_mutually_exclusive_group", "action.add_argument",
                              "action.add_argument", "parser.parse_args"]
assert not set(calls("make_speed_window")) & {"make_snd_partitions", "mods['frozen_v19_analytic_onset'].estimate"}
assert "np.percentile" in calls("make_speed_window")
assert any(isinstance(n, ast.keyword) and n.arg == "method"
           and isinstance(n.value, ast.Constant) and n.value.value == "linear"
           for n in ast.walk(functions["make_speed_window"]))
assert "read_snd_speeds" not in calls("snd_features")
assert "mods['data_cache'].physical_matrix" in calls("score_unit")
assert "mods['cadence'].dense_flags" in calls("score_unit")
assert "mods['wider_models'].score" in calls("score_unit")
main_source = ast.unparse(functions["main"])
assert main_source.index("time_metadata(") < main_source.index("check_data_integrity(")
assert main_source.index("check_data_integrity(") < main_source.index("frozen_imports(")
assert main_source.index("ANALYSIS_STARTED.json") < main_source.index("process_snd_unit(")
assert "variable_names=['vibration']" in main_source
assert source.index("## No-filter sensitivity") < source.index("## Partition and onset audit") < source.index("## Limitations")
assert "Strict successes /6" in source and "Controls silent /5" in source
assert "not applicable (no speed column)" in source
assert "first_alarm_in_fault_period" in source and "365.25" in source
print("PASS: static arm inheritance, model/scaler/cadence imports, gate order, HSS selection and report structure")

p = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
old = json.loads((ROOT / "protocol_v1.json").read_text(encoding="utf-8"))
assert p["version"] == 2 and p["amendment"]["number"] == 1
for key in ("models", "training", "features", "alarm", "target_files", "seeds"):
    assert p[key] == old[key], key
assert p["snd"]["identity_gate"] == old["snd"]["identity_gate"]
assert p["snd"]["units"] == old["snd"]["units"]
assert p["partitions"] == old["partitions"]
assert p["onset"]["no_onset"] == old["onset"]["no_onset"]
assert p["snd"]["arms"] == ["authors_rms_gt_0p5g", "variable_speed_operating_window", "no_operating_filter"]
assert p["amendment"]["expected_output_rows"] == {"primary_snd_including_sensitivity": 54, "secondary_event2": 6, "hss": 6, "total": 66}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for rel, expected in p["frozen_files_sha256"].items():
    path = (ROOT / rel).resolve()
    assert path.is_relative_to(ROOT)
    assert digest(path) == expected, rel
assert digest(ROOT / "download_sha256.txt") == p["download_manifest_sha256"]
for rel, expected in old["frozen_files_sha256"].items():
    if rel.startswith("frozen/") or rel == "source_snapshot_manifest.json":
        assert digest(ROOT / rel) == expected, rel
assert digest(ROOT / "run_unseen_v1.py") == old["frozen_files_sha256"]["run_unseen.py"]
checks = {}
for line in (ROOT / "PROTOCOL_SHA256.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        expected, rel = line.split(None, 1)
        checks[rel.lstrip(" *")] = expected
assert set(checks) == {"PROTOCOL.md", "protocol.json", "PROTOCOL_v1.md", "protocol_v1.json"}
for rel, expected in checks.items():
    assert digest(ROOT / rel) == expected, rel
assert checks["PROTOCOL_v1.md"] == "b34745734954a655680afc37c575a905fd4cd7a7906a961c90985d5ea27206a6"
assert checks["protocol_v1.json"] == "5b4ce87ac9c87e95df754267c79544293bfbef31477469e2880705355f05539a"
print(f"PASS: v1 preservation, unchanged scientific core and {len(p['frozen_files_sha256'])} artifact seals plus 4 protocol seals")
print("No runner module/main, target file, model, pipeline import or analysis path executed")
