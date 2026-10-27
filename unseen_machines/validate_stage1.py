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
old = json.loads((ROOT / "protocol_v2.json").read_text(encoding="utf-8"))
v1 = json.loads((ROOT / "protocol_v1.json").read_text(encoding="utf-8"))
assert p["version"] == 3 and p["amendment"]["number"] == 2
for key in ("models", "training", "features", "alarm", "target_files", "seeds", "hss", "outcomes",
            "partitions", "execution_commands", "outputs", "download_manifest_sha256"):
    assert p[key] == old[key], key
for key in ("identity_gate", "units", "arms", "operating_window", "rms_gate_g", "rms_definition",
            "bearing_replacement_years", "gearbox_replacement_years", "replacement_rule"):
    assert p["snd"][key] == old["snd"][key], key
for key in old["onset"]:
    if key not in ("unit", "no_onset", "limitations"):
        assert p["onset"][key] == old["onset"][key], key
assert p["snd"]["arms"] == ["authors_rms_gt_0p5g", "variable_speed_operating_window", "no_operating_filter"]
assert p["amendment"]["expected_output_rows"] == {"primary_snd_including_sensitivity": 54, "secondary_event2": 6, "hss": 6, "total": 66}
assert p["amendment"]["report_order"] == old["amendment"]["report_order"]
assert p["output_root"] == r"D:\external_unseen\results_v3"
documented = p["onset"]["documented_onsets"]
assert len(documented) == 1 and documented[0]["dataset"] == "SND"
assert documented[0]["turbine"] == 5 and documented[0]["event"] == "event1" and documented[0]["years"] == 1.0

# Compare immutable implementation paths as AST only; no function execution.
old_source = (ROOT / "run_unseen_v2.py").read_text(encoding="utf-8")
old_functions = {n.name: n for n in ast.parse(old_source).body if isinstance(n, ast.FunctionDef)}
for name in ("main", "parse_args", "time_metadata", "frozen_imports", "check_data_integrity",
             "load_models", "read_snd_speeds", "make_speed_window"):
    assert ast.dump(functions[name]) == ast.dump(old_functions[name]), name
# Scoring math, scaler, persistence, HSS and lead days are unchanged. Only onset
# provenance columns have changed; reconstruct v2's single legacy assignment.
score_v3 = ast.unparse(functions["score_unit"])
score_v3 = score_v3.replace("result['first_alarm_before_onset'] = bool(first_index < len(ti) + len(ci) + len(hi)) if first_index is not None else None\n            result['first_alarm_before_estimated_onset'] = result['first_alarm_before_onset'] if extra.get('onset_method') == 'historical_proxy' else None",
                          "result['first_alarm_before_estimated_onset'] = bool(first_index < len(ti) + len(ci) + len(hi)) if first_index is not None else None")
assert score_v3 == ast.unparse(old_functions["score_unit"])
partition_source = ast.unparse(functions["make_snd_partitions"])
assert partition_source.index("documented_onset(") < partition_source.index(".estimate(")
assert "np.count_nonzero(data['times'][positions] < documented['years'])" in partition_source
proxy_call = next(n for n in ast.walk(functions["make_snd_partitions"])
                  if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".estimate"))
branches = [n for n in ast.walk(functions["make_snd_partitions"]) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "documented is not None"]
assert len(branches) == 1
assert any(proxy_call is child for statement in branches[0].orelse for child in ast.walk(statement))
assert not any(proxy_call is child for statement in branches[0].body for child in ast.walk(statement))
feature_source = ast.unparse(functions["snd_features"])
assert "unit['turbine'] == 5 and documented_onset(unit, p) is None" in feature_source
process_source = ast.unparse(functions["process_snd_unit"])
assert "isinstance(exc, NoProxyOnset) and event == 'event2'" in process_source
assert "for affected_arm in arms if no_proxy else (arm,):" in process_source
assert "onset['onset_years'] if onset and onset['method'] == 'documented'" in process_source
assert "not evaluable (no proxy onset)" in ast.unparse(functions["record_not_evaluable"])
verify_source = ast.unparse(functions["verify_protocol"])
assert "Path(p['output_root']).resolve() != root / 'results_v3'" in verify_source
assert "NoProxyOnset" not in main_source  # no broad top-level no-onset exception
disclosure_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "AMENDMENT_2_DISCLOSURE" for t in n.targets))
disclosure = ast.literal_eval(disclosure_node.value)
assert p["amendment"]["disclosure"] == disclosure
assert disclosure in (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")
assert "AMENDMENT_2_DISCLOSURE" in ast.unparse(functions["report"])
print("PASS: static documented/proxy branching, event2-only continuation, v3 output guard, disclosure and unchanged v2 core")


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
for version, archived in ((1, v1), (2, old)):
    assert digest(ROOT / f"run_unseen_v{version}.py") == archived["frozen_files_sha256"]["run_unseen.py"]
assert digest(ROOT / "STAGE1_VALIDATION_v2.md") == old["frozen_files_sha256"]["STAGE1_VALIDATION.md"]
assert digest(ROOT / "validate_stage1_v2.py") == old["frozen_files_sha256"]["validate_stage1.py"]
assert digest(ROOT / "PROTOCOL_v2.md") == p["amendment"]["parent_protocol_md_sha256"]
assert digest(ROOT / "protocol_v2.json") == p["amendment"]["parent_protocol_json_sha256"]
checks = {}
for line in (ROOT / "PROTOCOL_SHA256.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        expected, rel = line.split(None, 1)
        checks[rel.lstrip(" *")] = expected
assert set(checks) == {"PROTOCOL.md", "protocol.json", "PROTOCOL_v1.md", "protocol_v1.json", "PROTOCOL_v2.md", "protocol_v2.json"}
for rel, expected in checks.items():
    assert digest(ROOT / rel) == expected, rel
assert checks["PROTOCOL_v1.md"] == "b34745734954a655680afc37c575a905fd4cd7a7906a961c90985d5ea27206a6"
assert checks["protocol_v1.json"] == "5b4ce87ac9c87e95df754267c79544293bfbef31477469e2880705355f05539a"
assert checks["PROTOCOL_v2.md"] == "c6cbae027aabdabb0090031f8d51bdc2260b111a80ee4dfd94e11916387cda21"
assert checks["protocol_v2.json"] == "221bd1e462dabf46f5012a23eb7f38fbdd747e7fd156ae9a14353a1da65d489c"
for version in (1, 2):
    for line in (ROOT / f"PROTOCOL_SHA256_v{version}.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, rel = line.split(None, 1)
            rel = rel.lstrip(" *")
            if rel in ("PROTOCOL.md", "protocol.json"):
                rel = rel.replace(".md", f"_v{version}.md").replace(".json", f"_v{version}.json")
            assert digest(ROOT / rel) == expected, (version, rel)
assert (ROOT / "DEVIATIONS.md").read_bytes().endswith(b"\n")
print(f"PASS: byte-identical v1/v2 archives, unchanged scientific core and {len(p['frozen_files_sha256'])} artifact seals plus 6 protocol seals")
print("No runner module/main, target file, model, pipeline import or analysis path executed")
