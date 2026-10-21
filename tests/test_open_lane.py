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
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "docs/COUNTED.md", "start": 1, "lines": 5}}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "../etc/passwd"}}},
    ]
    out = io.StringIO()
    serve(session, io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 6, 7]
    if HAVE_LEAN:
        assert replies[5]["result"]["isError"] is False and "1: # The counted language" in replies[5]["result"]["content"][0]["text"]
    assert replies[6]["result"]["isError"] is True
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18" and "tools" in replies[0]["result"]["capabilities"]
    assert {t["name"] for t in replies[1]["result"]["tools"]} == {"lean_check", "run_python", "search_library", "read_file", "submit_candidate"}
    assert replies[2]["result"]["isError"] is False
    assert replies[3]["result"]["isError"] is True and "needs" in replies[3]["result"]["content"][0]["text"]
    assert "error" in replies[4]
    log = [json.loads(l) for l in (session.path / "tools.jsonl").read_text().splitlines()]
    assert [l["tool"] for l in log] == ["search_library", "read_file", "read_file"]  # a malformed call is refused before it is a tool call


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
    assert cmd[cmd.index("--allowedTools") + 1].split(",") == [f"mcp__cpl__{t}" for t in ("lean_check", "run_python", "search_library", "read_file", "submit_candidate")]
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    r = subprocess.run([mcp["command"]] + mcp["args"], input="\n".join(json.dumps(q) for q in requests) + "\n",
                       capture_output=True, text=True, timeout=120, env={**os.environ, **mcp["env"]})
    assert r.returncode == 0, r.stderr
    replies = [json.loads(line) for line in r.stdout.splitlines()]
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert len(replies[1]["result"]["tools"]) == 5


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
    # patience for the open lane: round 1 accepted new artifacts, round 2 none
    assert loop.rounds_without_new_artifact(ctx.run) == 1
    # the final evaluation refuses an open-lane run until frozen transfer exists
    from core import final

    with pytest.raises(final.FinalError, match="frozen"):
        final.evaluate(ctx.run.path, log=lines.append)
    # the run recorded the Lean service's identity, and every receipt carries the same digests
    started = json.loads((ctx.run.path / "versions.json").read_text())
    ident = started["lean_checker"]
    assert set(ident) == set(loop.LEAN_IDENTITY_KEYS) and ident["commit"] == json.loads(loop.LEAN_LOCK.read_text())["commit"]
    digests = ok[0]["receipt"]["digests"]
    assert all(digests[k] == ident[k] for k in loop.LEAN_IDENTITY_KEYS)
    # resume refuses a service that is not the one the run started with, whatever today's lock says
    import pytest as _pytest

    real_identity, real_lock = loop.lean_identity, loop.LEAN_LOCK
    try:
        loop.lean_identity = lambda project: {**ident, "policy": "0000000000000000"}
        with _pytest.raises(loop.LoopError, match="policy"):
            loop.resume(ctx.run.path, log=lines.append)
        loop.lean_identity = real_identity
        moved = tmp_path / "moved.lock"
        moved.write_text(json.dumps({**json.loads(real_lock.read_text()), "commit": "f" * 40}))
        loop.LEAN_LOCK = moved
        with _pytest.raises(loop.LoopError, match="moved"):
            loop.resume(ctx.run.path, log=lines.append)
    finally:
        loop.lean_identity, loop.LEAN_LOCK = real_identity, real_lock
    # resume: nothing is redone, nothing is added
    before = len(records)
    ctx2 = loop.resume(ctx.run.path, log=lines.append)
    loop.drive(ctx2)
    assert len(ctx2.run.read_archive()) == before


def test_patience_counts_claims_not_files(tmp_path):
    """Progress detection for the open lane, with no Lean: a run folder with synthetic archive records.
    Round 1 accepts a scored proof and a lemma; round 2 resubmits both from files with a changed comment
    (new digests, same claims): not progress; round 3 proves a second theorem already present in the same
    unchanged file (same digests, new statement): progress; round 4 improves the length: progress; round 5
    repeats round 4's claim under another theorem name, adds a verified-only proof of a count that has a
    scored one, and a rejection: not progress."""
    from core.archive import RunDir

    run = RunDir(tmp_path / "runs", "patience-test")
    run.create({"run_id": "patience-test"})
    base = {"run_id": "patience-test", "lane": "open", "network": "n1", "attempt": 0}

    def rec(k, i, **f):
        return {**base, "round": k, "record_id": f"patience-test/r{k:03d}/a0/n1/open{i:03d}",
                "receipt": {"digests": {"modules": f.pop("digest")}}, **f}

    rounds = {
        1: [rec(1, 0, status="ok", certified=23, length=642, theorem="A.performance", digest="d1"),
            rec(1, 1, status="lemma", statement="∀ x, p x", theorem="A.lemma1", digest="d1")],
        2: [rec(2, 0, status="ok", certified=23, length=642, theorem="A.performance", digest="d2"),
            rec(2, 1, status="lemma", statement="∀ x,  p x", theorem="A.lemma1", digest="d2")],
        3: [rec(3, 0, status="lemma", statement="∀ x, q x", theorem="A.lemma2", digest="d2")],
        4: [rec(4, 0, status="ok", certified=23, length=600, theorem="A.performance", digest="d4")],
        5: [rec(5, 0, status="ok", certified=23, length=600, theorem="A.performance2", digest="d5"),
            rec(5, 1, status="verified", certified=23, theorem="A.performance3", digest="d5"),
            rec(5, 2, status="rejected", theorem="A.performance4", digest="d6")],
    }
    expected = {1: 0, 2: 1, 3: 0, 4: 0, 5: 1}
    for k in range(1, 6):
        run.round_path(k).mkdir(parents=True)
        run.append_archive(rounds[k])
        run.mark_done(k)
        assert loop.rounds_without_new_artifact(run) == expected[k], k
    assert loop.claim_identity(rounds[5][2]) is None


def test_the_tool_backend_reports_the_counters_the_tools_saved(tmp_path):
    """The tools run in the MCP server's process and save their counters to the session; the backend's
    metadata must come from that saved state, not from the session it loaded before the CLI ran. A fake
    CLI that bumps the saved counters (as the server would) and answers; no Lean, no model."""
    from core.open_lane.backends import ClaudeCodeToolBackend
    from tests.test_claude_code import result_json

    cfg = merge(base_config(tmp_path, [], backend="claude_code", agent_model="m"))
    where = tmp_path / "runs" / "r" / "round_001" / "attempt_0"
    where.mkdir(parents=True)

    def cli(cmd, prompt_text, env, cwd, timeout_s):
        state_path = where / "open_lane" / "session.json"
        state = json.loads(state_path.read_text())
        state["used"] = {"lean_check": 2, "submit_candidate": 1, "run_python": 3}
        state_path.write_text(json.dumps(state))

        class Done:
            stdout, stderr, returncode = result_json("CLAIM: nothing"), "", 0

        return Done()

    b = ClaudeCodeToolBackend(cfg, log=lambda *a: None, runner=cli, sleep=lambda s: None, environ={"PATH": "/usr/bin"})
    b.bind(cfg, "max2_tiny", [{"id": "max2_tiny-v8-s0", "setting": "v8", "n_inputs": 64}], where.parent.parent)
    from core.agent.build_prompt import Prompt

    r = b.respond(Prompt("one", "two", {}), 0, where=where)
    assert r.meta["tools_used"] == {"lean_check": 2, "submit_candidate": 1, "run_python": 3}


def test_a_receipt_is_evidence_only_with_the_whole_identity_of_the_service_the_run_started_with():
    """No Lean: the receipt check alone. An accepted receipt must carry all five digests of the service
    (commit, toolchain, trusted sources, policy, the acceptcheck binary) and each must equal what the run
    recorded; a missing field or a changed binary halts the run; a receipt rejected before its digests were
    complete is checked on what it has."""
    from core.open_lane.records import check_receipt_identity

    ident = {"commit": "c" * 40, "toolchain": "lean:aaaa", "lean_project": "bbbb", "policy": "cccc", "acceptcheck": "dddd"}
    accepted = {"status": "accepted", "receipt_path": "r.json", "digests": {**ident, "modules": {"Candidate.X": "ee"}}}
    check_receipt_identity(accepted, ident)
    for k in ident:
        incomplete = {**accepted, "digests": {kk: v for kk, v in accepted["digests"].items() if kk != k}}
        with pytest.raises(RuntimeError, match=k):
            check_receipt_identity(incomplete, ident)
    rebuilt = {**accepted, "digests": {**accepted["digests"], "acceptcheck": "ffff"}}
    with pytest.raises(RuntimeError, match="acceptcheck"):
        check_receipt_identity(rebuilt, ident)
    with pytest.raises(RuntimeError, match="no Lean checker identity"):
        check_receipt_identity(accepted, None)
    rejected_early = {"status": "rejected", "receipt_path": "r.json", "digests": {"commit": ident["commit"], "toolchain": ident["toolchain"]}}
    check_receipt_identity(rejected_early, ident)
    rejected_elsewhere = {**rejected_early, "digests": {**rejected_early["digests"], "toolchain": "lean:zzzz"}}
    with pytest.raises(RuntimeError, match="toolchain"):
        check_receipt_identity(rejected_elsewhere, ident)


def test_the_identity_command_must_succeed_and_report_every_field():
    """No Lean: what the pipeline accepts from `accept.py identity` at the start of a run."""
    full = {"commit": "c" * 40, "dirty": False, "toolchain": "lean:aaaa", "lean_project": "bbbb", "policy": "cccc", "acceptcheck": "dddd"}
    assert loop.parse_identity(0, json.dumps(full), "", "P") == {k: full[k] for k in loop.LEAN_IDENTITY_KEYS}
    with pytest.raises(loop.LoopError, match="exit 1"):
        loop.parse_identity(1, json.dumps(full), "boom", "P")
    with pytest.raises(loop.LoopError, match="did not report"):
        loop.parse_identity(0, "not json", "", "P")
    with pytest.raises(loop.LoopError, match="acceptcheck"):
        loop.parse_identity(0, json.dumps({**full, "acceptcheck": None}), "", "P")
    with pytest.raises(loop.LoopError, match="uncommitted"):
        loop.parse_identity(0, json.dumps({**full, "dirty": True}), "", "P")
