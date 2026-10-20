"""The open lane in the loop: the MCP server's protocol, and a fake-backend run that submits the scored
brute-force example through the Lean project's acceptance service and records its receipt.

The run test needs the Lean project built next to this repository (or at COMPACT_PROOF_LEAN) and takes a
few minutes; it is skipped otherwise.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from core import loop
from core.open_lane.mcp_server import serve
from core.open_lane.session import Session
from core.run_config import merge
from core.util import REPO_ROOT

LEAN = Path(os.environ.get("COMPACT_PROOF_LEAN", REPO_ROOT.parent / "auto-research-compact-proof-in-lean"))
HAVE_LEAN = (LEAN / "scripts" / "accept.py").is_file() and (LEAN / ".lake" / "build" / "bin" / "acceptcheck").is_file()


def base_config(tmp_path: Path, scripts: list[str], **extra) -> dict:
    return {
        "run_id": "open-lane-test", "model": "tests/max2_tiny", "backend": "fake", "rule_set": ["single", "skip"],
        "prompt_template": "open_lane_v1", "baselines": False, "rounds_max": max(1, len(scripts)), "patience": 10,
        "runs_dir": str(tmp_path / "runs"), "sandbox": "auto", "workers": 1,
        "open_lane": {"enabled": True, "lean_project": str(LEAN), "work_dir": str(tmp_path / "work"),
                      "fake_scripts": scripts, "compile_calls": 3, "submissions": 3, "python_calls": 2, "stall_rounds": 1},
        **extra,
    }


def test_the_mcp_server_speaks_the_protocol(tmp_path):
    cfg = merge(base_config(tmp_path, [], open_lane={"enabled": False}))
    cfg["open_lane"].update(lean_project=str(LEAN) if HAVE_LEAN else str(tmp_path), work_dir=str(tmp_path / "work"))
    entries = [{"id": "max2_tiny-v8-s0", "setting": "v8", "n_inputs": 64}]
    session = Session.create(tmp_path / "attempt" / "open_lane", cfg, "tests/max2_tiny", entries, tmp_path / "run")
    (tmp_path / "docs").mkdir(exist_ok=True)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_library", "arguments": {"query": "numCorrect"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "lean_check", "arguments": {"path": "x"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nothing"},
    ]
    out = io.StringIO()
    serve(session, io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18" and "tools" in replies[0]["result"]["capabilities"]
    assert {t["name"] for t in replies[1]["result"]["tools"]} == {"lean_check", "run_python", "search_library", "submit_candidate"}
    assert replies[2]["result"]["isError"] is False
    assert replies[3]["result"]["isError"] is True and "needs" in replies[3]["result"]["content"][0]["text"]
    assert "error" in replies[4]
    log = [json.loads(l) for l in (session.path / "tools.jsonl").read_text().splitlines()]
    assert [l["tool"] for l in log] == ["search_library"]  # a malformed call is refused before it is a tool call


def test_the_mcp_server_runs_as_the_cli_would_start_it(tmp_path):
    """The server as a subprocess, started with the command the Claude Code backend writes into mcp.json."""
    import subprocess

    from core.open_lane.backends import ClaudeCodeToolBackend

    cfg = merge(base_config(tmp_path, [], open_lane={"enabled": False}))
    cfg["open_lane"].update(lean_project=str(LEAN) if HAVE_LEAN else str(tmp_path), work_dir=str(tmp_path / "work"))
    entries = [{"id": "max2_tiny-v8-s0", "setting": "v8", "n_inputs": 64}]
    where = tmp_path / "attempt_0"
    session = Session.create(where / "open_lane", cfg, "tests/max2_tiny", entries, tmp_path / "run")
    backend = ClaudeCodeToolBackend(cfg, runner=lambda *a, **k: None, check_login=False)
    backend.bind(cfg, "tests/max2_tiny", entries, tmp_path / "run")
    mcp = json.loads(backend.mcp_config(session, where).read_text())["mcpServers"]["cpl"]
    cmd = backend.command_line(where / "mcp.json")
    assert "--mcp-config" in cmd and "--allowedTools" in cmd and "--max-turns" in cmd and "--tools" in cmd
    assert cmd[cmd.index("--allowedTools") + 1].split(",") == [f"mcp__cpl__{t}" for t in ("lean_check", "run_python", "search_library", "submit_candidate")]
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    r = subprocess.run([mcp["command"]] + mcp["args"], input="\n".join(json.dumps(q) for q in requests) + "\n",
                       capture_output=True, text=True, timeout=120, env={**os.environ, **mcp["env"]})
    assert r.returncode == 0, r.stderr
    replies = [json.loads(line) for line in r.stdout.splitlines()]
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert len(replies[1]["result"]["tools"]) == 4


def test_the_lean_lock_is_checked(tmp_path):
    """preflight refuses a Lean project at another commit than lean-checker.lock names."""
    import subprocess

    if not HAVE_LEAN:
        pytest.skip("no Lean project")
    lock = json.loads((REPO_ROOT / "lean-checker.lock").read_text())
    head = subprocess.run(["git", "-C", str(LEAN), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head == lock["commit"]:
        assert loop.check_lean_lock(LEAN)["commit"] == head
    else:
        with pytest.raises(loop.LoopError, match="lean-checker.lock"):
            loop.check_lean_lock(LEAN)
    with pytest.raises(loop.LoopError):
        loop.check_lean_lock(tmp_path)


def test_budgets_persist_across_sessions(tmp_path):
    cfg = merge(base_config(tmp_path, [], open_lane={"enabled": False}))
    cfg["open_lane"].update(lean_project=str(tmp_path), work_dir=str(tmp_path / "work"), compile_calls=1)
    entries = [{"id": "max2_tiny-v8-s0", "setting": "v8", "n_inputs": 64}]
    s = Session.create(tmp_path / "a" / "open_lane", cfg, "tests/max2_tiny", entries, tmp_path / "run")
    s._spend("lean_check")
    again = Session.create(tmp_path / "a" / "open_lane", cfg, "tests/max2_tiny", entries, tmp_path / "run")
    assert again.remaining()["lean_check"] == 0
    with pytest.raises(Exception, match="budget"):
        again._spend("lean_check")


@pytest.mark.skipif(not HAVE_LEAN, reason="the Lean project with a built acceptcheck is not next to this repository")
def test_a_fake_session_submits_the_scored_example_and_the_receipt_becomes_a_record(tmp_path):
    scripts = [str(REPO_ROOT / "tests" / "open_lane" / "scored_brute.json"), str(REPO_ROOT / "tests" / "open_lane" / "nothing.json")]
    cfg_path = tmp_path / "run.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(base_config(tmp_path, scripts)))
    lines = []
    ctx = loop.start(cfg_path, log=lines.append)
    loop.drive(ctx)
    records = ctx.run.read_archive()
    ok = [r for r in records if r["status"] == "ok"]
    assert len(ok) == 1 and ok[0]["lane"] == "open" and ok[0]["certified"] == 64 and ok[0]["length"] == 4112
    assert ok[0]["checked_by"] == "lean" and ok[0]["network"] == "max2_tiny-v8-s0" and ok[0]["round"] == 1
    assert "reference_warning" not in ok[0]
    lemma = [r for r in records if r["status"] == "lemma"]
    assert len(lemma) == 1 and "countP" in lemma[0]["statement"] and lemma[0]["certified"] is None
    assert [r["status"] for r in records if r["round"] == 2] == ["no_submission"]
    meta = ctx.run.read(1, "attempt_0/response_meta.json")
    assert meta["submissions"] == 2 and meta["tools_used"] == {"lean_check": 1, "submit_candidate": 2, "run_python": 1}
    log = [json.loads(l) for l in (ctx.run.round_path(1) / "attempt_0" / "open_lane" / "tools.jsonl").read_text().splitlines()]
    assert [l["tool"] for l in log] == ["search_library", "run_python", "lean_check", "submit_candidate", "submit_candidate"]
    assert "M (8, 8)" in log[1]["outcome"]["output"] and log[2]["outcome"]["ok"]
    # the prompt of round 2 shows the receipt, the lemma, the accepted sources and the notes
    prompt = ctx.run.read(2, "prompt.md")
    assert "length 4,112" in prompt and "countSingles" in prompt and "column sums" in prompt and "auxiliary lemma accepted" in prompt
    assert "{COUNTED}" not in prompt and "def qadd" in prompt and "Correct" in prompt
    summary = loop.status_text(ctx.run)
    assert "Q" in summary
    # resume: nothing is redone, nothing is added
    before = len(records)
    ctx2 = loop.resume(ctx.run.path, log=lines.append)
    loop.drive(ctx2)
    assert len(ctx2.run.read_archive()) == before
