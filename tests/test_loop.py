"""The loop driver: stopping rules, killing and resuming, and the Lean spot-check hook."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from core import loop
from core.lean import LeanError
from core.util import REPO_ROOT

TESTS = Path(__file__).resolve().parent
SLOW_RECIPE = '''```python
import time

import helpers


def make_proof(weights, info, knob):
    time.sleep(1.5)  # slow enough to be killed half way through a round
    return helpers.proof(helpers.full_tree(info["input_space"]), notes=str(knob))
```'''


def response(recipe: str, note: str) -> str:
    return (
        f"CLAIM: {note}\n\nWHY IT HELPS: no.\n\nPREDICTION: full accuracy.\n\nKNOB: unused.\n\n"
        f"RECIPE:\n{recipe}\n\nNOTES: {note}\n"
    )


def write_config(tmp_path, model, name, **over) -> Path:
    cfg = {
        "run_id": name,
        "model": str(model),
        "runs_dir": str(tmp_path / "runs"),
        "knob_values": [0, 1],
        "sandbox": "process",
        "rounds_max": 5,
        **over,
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def comparable(records):
    keep = ("status", "certified", "length", "length_by_rule", "proof_sha256", "uncertified_by_group", "rejected_summary", "error")
    out = {}
    for r in records:
        rid = r["record_id"].split("/", 1)[1]
        out[rid] = {k: r.get(k) for k in keep}
    return out


def test_patience_stops_a_run_that_stopped_gaining(trained_stub, tmp_path):
    brute = (TESTS / "stub_model" / "baselines" / "01_brute_force.md").read_text()
    folder = tmp_path / "responses"
    folder.mkdir()
    for i in range(4):
        (folder / f"{i}.md").write_text(brute)
    logs = []
    ctx = loop.start(write_config(tmp_path, trained_stub, "patience", patience=2, fake_responses=[str(folder)]), log=logs.append)
    loop.drive(ctx)
    assert ctx.run.done_rounds() == [0, 1, 2]
    assert loop.rounds_without_gain(ctx.run, ctx.entries, "Q") == 2
    assert any("patience 2" in line for line in logs)


def test_a_run_killed_mid_round_resumes_with_identical_results(trained_stub, tmp_path):
    folder = tmp_path / "responses"
    folder.mkdir()
    for i in range(3):
        (folder / f"{i}.md").write_text(response(SLOW_RECIPE, f"slow {i}"))
    over = {"fake_responses": [str(folder)], "workers": 1, "knob_values": [0, 0.5, 1], "rounds_max": 3}
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    def run(args):
        return subprocess.Popen([sys.executable, "-m", "core.loop", *args], cwd=REPO_ROOT, env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    ref = run(["run", "--config", str(write_config(tmp_path, trained_stub, "reference", **over))])
    killed = run(["run", "--config", str(write_config(tmp_path, trained_stub, "killed", **over))])
    attempt = tmp_path / "runs" / "killed" / "round_002" / "attempt_0"
    deadline = time.time() + 600
    while time.time() < deadline:
        if (attempt / "proofs").is_dir() and any((attempt / "proofs").iterdir()) and not (attempt / "results.json").exists():
            break
        assert killed.poll() is None, killed.stderr.read().decode()
        time.sleep(0.05)
    os.killpg(killed.pid, signal.SIGKILL)  # the loop, its recipes and its checker, all at once
    killed.wait()
    assert not (attempt.parent / "DONE").exists() and (attempt / "response.md").exists()

    resumed = run(["resume", "--run", str(tmp_path / "runs" / "killed")])
    assert resumed.wait(timeout=900) == 0, resumed.stderr.read().decode()
    assert ref.wait(timeout=900) == 0, ref.stderr.read().decode()
    from core.archive import RunDir

    a = RunDir(tmp_path / "runs", "reference").read_archive()
    b = RunDir(tmp_path / "runs", "killed").read_archive()
    assert len(a) == len(b) > 0
    assert comparable(a) == comparable(b)
    assert RunDir(tmp_path / "runs", "killed").done_rounds() == [0, 1, 2, 3]


FAKE_LEAN = """
import gzip, json, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
from core.check.checker import check_proof
from core.model_folder import load_model_folder
proof = json.loads(Path(sys.argv[1]).read_text())
folder = load_model_folder(Path(sys.argv[2]).parent.parent)
n = check_proof(proof, folder, proof["network"])["certified"]
print(json.dumps({{"certified": n + {offset}}}))
"""


def lean_model(trained_stub, tmp_path, offset: int) -> Path:
    model = tmp_path / f"lean-{offset}"
    shutil.copytree(trained_stub, model, ignore=shutil.ignore_patterns("__pycache__"))
    script = tmp_path / f"fake_lean_{offset}.py"
    script.write_text(FAKE_LEAN.format(repo=str(REPO_ROOT), offset=offset))
    cfg = yaml.safe_load((model / "config.yaml").read_text())
    cfg["lean_check"] = [sys.executable, str(script), "{proof}", "{weights}"]
    (model / "config.yaml").write_text(yaml.safe_dump(cfg))
    return model


def test_lean_spot_checks_agree_or_halt_the_run(trained_stub, tmp_path):
    sym = (TESTS / "stub_model" / "baselines" / "02_symmetry.md").read_text()
    folder = tmp_path / "responses"
    folder.mkdir()
    (folder / "0.md").write_text(sym.replace("Stub baseline 2 of 2.", "Symmetry, written by the agent."))
    # A Lean checker that agrees: the spot checks are logged.
    model = lean_model(trained_stub, tmp_path, 0)
    cfg = write_config(tmp_path, model, "lean-ok", lean_spot_check=True, baselines=False, fake_responses=[str(folder)], rounds_max=1)
    ctx = loop.start(cfg, log=lambda *a: None)
    loop.drive(ctx)
    lines = (ctx.run.path / "lean.jsonl").read_text().splitlines()
    assert lines and all('"agree": true' in ln for ln in lines)
    # One that disagrees: the run halts.
    model = lean_model(trained_stub, tmp_path, 1)
    cfg = write_config(tmp_path, model, "lean-bad", lean_spot_check=True, baselines=False, fake_responses=[str(folder)], rounds_max=1)
    ctx = loop.start(cfg, log=lambda *a: None)
    with pytest.raises(LeanError, match="disagree"):
        loop.drive(ctx)


def test_lean_spot_checks_need_a_lean_checker(trained_stub, tmp_path):
    with pytest.raises(loop.LoopError, match="no Lean checker"):
        loop.start(write_config(tmp_path, trained_stub, "no-lean", lean_spot_check=True), log=lambda *a: None)


def test_brute_force_only_round_zero_hides_the_other_baselines(trained_stub, tmp_path):
    from core.agent.build_prompt import build_prompt
    from core.report import build_report

    cfg = write_config(tmp_path, trained_stub, "bf-only", baselines=["brute_force"], fake_responses=[], rounds_max=1)
    ctx = loop.start(cfg, log=lambda *a: None)
    loop.drive(ctx)
    sources = [m["source"] for m in ctx.run.read(0, "meta.json")["attempts"]]
    assert sources == ["01_brute_force.md"]
    p = build_prompt(ctx.cfg, ctx.folder, ctx.entries, ctx.run, 1)
    assert "01_brute_force" in p.part2 and "symmetry" not in p.part2.split("BEST RECIPES")[1].split("YOUR LAST")[0].lower().replace("symmetry: ", "")
    assert "02_symmetry" not in p.text
    # Q starts at 0: brute force alone scores nothing, so every gain is the agent's own.
    from core.scoring import summarize

    s = summarize(ctx.run.read_archive(), ctx.entries)
    assert all(v["Q_median"] == 0.0 for v in s.values())
    # The full-baseline run can be overlaid as a reference in the report.
    full = loop.start(write_config(tmp_path, trained_stub, "all-baselines", fake_responses=[], rounds_max=1), log=lambda *a: None)
    loop.drive(full)
    report = build_report(ctx.run.path, full.run.path)
    text = report.read_text()
    assert "Reference proofs (never shown to the agent)" in text and "02_symmetry" in text


def test_unknown_baseline_name_is_an_error(trained_stub, tmp_path):
    with pytest.raises(loop.LoopError, match="no baseline named"):
        loop.start(write_config(tmp_path, trained_stub, "bad-name", baselines=["contraction"]), log=lambda *a: None)
