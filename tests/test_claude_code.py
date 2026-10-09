"""The Claude Code backend, against a fake CLI. No test here runs the real CLI or calls a model."""

import json
import subprocess

import pytest
import yaml

from core import loop
from core.agent.backends import BackendError
from core.agent.backends.api import StopRun
from core.agent.backends.claude_code import ClaudeCodeBackend
from core.agent.build_prompt import Prompt
from core.run_config import merge
from tests.test_api import RESPONSE_TEXT

PROMPT = Prompt("PART ONE", "PART TWO", {})


def result_json(text=RESPONSE_TEXT, is_error=False, **extra):
    d = {
        "type": "result",
        "subtype": "success" if not is_error else "error_during_execution",
        "is_error": is_error,
        "result": text,
        "session_id": "sess_123",
        "total_cost_usd": 0.42,
        "num_turns": 1,
        "usage": {"input_tokens": 5000, "output_tokens": 20000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 3600},
    }
    d.update(extra)
    return json.dumps(d)


class Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeCli:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, cmd, prompt_text, env, cwd, timeout_s):
        self.calls.append({"cmd": cmd, "prompt": prompt_text, "env": env, "cwd": cwd, "timeout": timeout_s})
        if not self.outcomes:
            raise AssertionError("an unexpected CLI run")
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def config(**over):
    return merge({"run_id": "cc-test", "model": "models/max2", "backend": "claude_code", **over})


def backend(outcomes, environ=None, **over):
    sleeps, logs = [], []
    cli = FakeCli(outcomes)
    b = ClaudeCodeBackend(config(**over), log=logs.append, runner=cli, sleep=sleeps.append, environ=environ or {"PATH": "/usr/bin"})
    return b, cli, sleeps, logs


def test_the_cli_is_run_as_a_plain_model_call():
    b, cli, _, _ = backend([Completed(result_json())], effort="high")
    r = b.respond(PROMPT, 0)
    call = cli.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["claude", "-p"]
    for flag in ("--bare", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools at all
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "claude-fable-5-1" and cmd[cmd.index("--effort") + 1] == "high"
    assert call["prompt"] == PROMPT.text  # the whole prompt, on stdin
    assert call["cwd"] and "cpl-claude-code-" in call["cwd"]
    assert r.text == RESPONSE_TEXT


def test_no_api_key_ever_reaches_the_cli():
    env = {"PATH": "/usr/bin", "HOME": "/home/x", "ANTHROPIC_API_KEY": "sk-should-not-leak", "CPL_ANTHROPIC_KEY": "sk-nor-this", "ANTHROPIC_AUTH_TOKEN": "t"}
    b, cli, _, _ = backend([Completed(result_json())], environ=env)
    b.respond(PROMPT, 0)
    seen = cli.calls[0]["env"]
    assert "ANTHROPIC_API_KEY" not in seen and "CPL_ANTHROPIC_KEY" not in seen and "ANTHROPIC_AUTH_TOKEN" not in seen
    assert seen["HOME"] == "/home/x"  # the login lives under HOME, so that stays


def test_response_is_saved_raw_and_tokens_recorded(tmp_path):
    b, _, _, _ = backend([Completed(result_json())])
    r = b.respond(PROMPT, 0, tmp_path)
    raw = json.loads((tmp_path / "response_raw.json").read_text())
    assert raw["session_id"] == "sess_123"
    m = r.meta
    assert m["backend"] == "claude_code" and m["input_tokens"] == 5000 and m["output_tokens"] == 20000
    assert m["cost_usd"] == 0.0 and m["api_equivalent_usd"] == 0.42 and m["stop_reason"] == "end_turn"


def test_usage_limits_are_retried_with_growing_waits():
    limited = Completed(result_json("You have hit your usage limit. Try again later.", is_error=True), returncode=1)
    b, cli, sleeps, logs = backend([limited, limited, Completed(result_json())], api_retries={"tries": 5, "first_wait_s": 60, "max_wait_s": 1800, "timeout_s": 600})
    r = b.respond(PROMPT, 0)
    assert r.text == RESPONSE_TEXT and len(cli.calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] * 1.5 and sleeps[0] >= 60
    assert len(r.meta["failed_tries"]) == 2 and "usage or rate limit" in logs[0]


def test_a_login_problem_stops_the_run_at_once():
    b, cli, sleeps, _ = backend([Completed("", stderr="Error: not logged in. Please run claude login", returncode=1)])
    with pytest.raises(StopRun, match="login"):
        b.respond(PROMPT, 0)
    assert sleeps == [] and len(cli.calls) == 1


def test_endless_failures_stop_after_the_configured_tries():
    junk = Completed("not json at all", returncode=2)
    b, _, sleeps, _ = backend([junk] * 3, api_retries={"tries": 3, "first_wait_s": 1, "max_wait_s": 4, "timeout_s": 600})
    with pytest.raises(StopRun, match="after 3 tries"):
        b.respond(PROMPT, 0)
    assert len(sleeps) == 2


def test_a_hung_cli_is_retried():
    hung = subprocess.TimeoutExpired(cmd="claude", timeout=1)
    b, _, sleeps, _ = backend([hung, Completed(result_json())])
    assert b.respond(PROMPT, 0).text == RESPONSE_TEXT and len(sleeps) == 1


def test_missing_cli_is_reported_before_the_run_starts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(BackendError, match="not installed"):
        ClaudeCodeBackend(config(claude_code_command="claude-nowhere"), environ={"PATH": "/usr/bin"})


# through the loop, still with the fake CLI ---------------------------------------------
def test_a_subscription_run_records_zero_cost_and_resumes(trained_stub, tmp_path, monkeypatch):
    limited = Completed(result_json("usage limit reached", is_error=True), returncode=1)
    cli = FakeCli([Completed(result_json()), limited, limited, limited])
    monkeypatch.setattr(loop, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(
        loop, "ClaudeCodeBackend",
        lambda cfg, log=print: ClaudeCodeBackend(cfg, log=log, runner=cli, sleep=lambda s: None, environ={"PATH": "/usr/bin"}),
    )
    cfg = {
        "run_id": "cc-run", "model": str(trained_stub), "runs_dir": str(tmp_path / "runs"), "backend": "claude_code",
        "knob_values": [0], "sandbox": "process", "rounds_max": 3,
        "api_retries": {"tries": 3, "first_wait_s": 1, "max_wait_s": 2, "timeout_s": 60},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    ctx = loop.start(path, log=lambda *a: None)
    with pytest.raises(StopRun, match="after 3 tries"):
        loop.drive(ctx)
    assert ctx.run.done_rounds() == [0, 1]
    meta = ctx.run.read(1, "meta.json")
    assert meta["cost_usd"] == 0.0 and meta["tokens"] == {"input": 5000, "output": 20000}
    assert ctx.run.read(1, "attempt_0/response_meta.json")["api_equivalent_usd"] == 0.42
    # The limit lifts: resume finishes without re-asking for round 1.
    cli.outcomes = [Completed(result_json()), Completed(result_json())]
    ctx2 = loop.resume(ctx.run.path, log=lambda *a: None)
    loop.drive(ctx2)
    assert ctx2.run.done_rounds() == [0, 1, 2, 3] and len(cli.calls) == 6
