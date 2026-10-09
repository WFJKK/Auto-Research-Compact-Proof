"""The Claude Code backend: one headless run of the official Claude Code CLI per attempt.

This lets a run draw on a Claude subscription (Pro or Max) through the user's
own Claude Code login instead of billing an API key. Each attempt runs

    claude -p --bare --tools "" --output-format json --no-session-persistence
           --model <agent_model> --effort <effort> --system-prompt <short>

with the prompt on stdin and an empty temporary folder as the working
directory, so the CLI behaves as a plain model call: no tools, no files, no
hooks, no project settings. The JSON result gives the response text and the
tokens used; the CLI's own dollar figure is recorded as api_equivalent_usd,
while cost_usd stays 0 because nothing is billed per call.

The CLI's environment has ANTHROPIC_API_KEY and the pipeline's key variable
removed, so it cannot fall back to billing an API key. Usage-limit and
overload errors are retried with growing waits (a subscription's window can
take a while to reset); a login problem stops the run, which `resume`
continues later.

Whether a subscription may be used this way is governed by Anthropic's terms
for Claude Code, which have changed over time: check the current policy for
your account before relying on it.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import BackendError, Response
from .api import StopRun

SYSTEM_PROMPT = (
    "You are answering a research prompt. You have no tools and no files. Answer the message directly and "
    "completely, in exactly the format it asks for, and nothing else."
)
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "overloaded", "too many requests", "capacity", "529", "429")
LOGIN_WORDS = ("not logged in", "login", "authenticat", "unauthorized", "invalid api key", "401", "403")
STDERR_TAIL = 2000


def _run_cli(cmd, prompt_text, env, cwd, timeout_s):
    """subprocess.run, kept separate so tests can replace it."""
    return subprocess.run(
        cmd, input=prompt_text, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout_s
    )


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(self, cfg: dict, log=print, runner=None, sleep=time.sleep, environ=None, check_login: bool = True):
        self.cfg = cfg
        self.log = log
        self.run = runner or _run_cli
        self.sleep = sleep
        self.retries = cfg["api_retries"]
        self.command = cfg["claude_code_command"]
        env = dict(os.environ if environ is None else environ)
        for var in KEY_VARIABLES + (cfg["api_key_env"],):
            env.pop(var, None)  # the subscription login, never an API key
        self.env = env
        if runner is None:
            if shutil.which(self.command) is None:
                raise BackendError(f"the Claude Code CLI ({self.command!r}) is not installed or not on PATH")
            if check_login:
                self._check_login()

    def _check_login(self) -> None:
        try:
            r = subprocess.run([self.command, "auth", "status"], capture_output=True, text=True, env=self.env, timeout=60)
            status = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise BackendError(f"could not check the Claude Code login: {exc}") from exc
        if not status.get("loggedIn"):
            raise BackendError("Claude Code is not logged in; run `claude` once and sign in with your subscription")

    def available(self, index: int) -> bool:
        return True

    def command_line(self) -> list[str]:
        cmd = [self.command, "-p", "--bare", "--tools", "", "--output-format", "json", "--no-session-persistence"]
        cmd += ["--system-prompt", SYSTEM_PROMPT, "--model", self.cfg["agent_model"]]
        if self.cfg["effort"] is not None:
            cmd += ["--effort", self.cfg["effort"]]
        return cmd

    def _wait(self, attempt: int) -> float:
        wait = min(self.retries["max_wait_s"], self.retries["first_wait_s"] * 2**attempt)
        return wait * (1 + 0.1 * random.random())

    @staticmethod
    def _classify(text: str) -> tuple[bool, str]:
        """(retry?, description) for an error the CLI reported."""
        low = text.lower()
        if any(w in low for w in LOGIN_WORDS) and not any(w in low for w in LIMIT_WORDS):
            return False, f"Claude Code login problem: {text[:300]}"
        if any(w in low for w in LIMIT_WORDS):
            return True, f"usage or rate limit: {text[:300]}"
        return True, f"Claude Code error: {text[:300]}"

    def respond(self, prompt, index: int, where=None) -> Response:
        cmd = self.command_line()
        t0 = time.monotonic()
        failures = []
        tries = int(self.retries["tries"])
        for attempt in range(tries):
            with tempfile.TemporaryDirectory(prefix="cpl-claude-code-") as cwd:
                try:
                    r = self.run(cmd, prompt.text, self.env, cwd, float(self.retries["timeout_s"]))
                except subprocess.TimeoutExpired:
                    retry, what = True, f"Claude Code did not answer within {self.retries['timeout_s']:g} s"
                    r = None
                except OSError as exc:
                    raise StopRun(f"could not run the Claude Code CLI: {exc}") from exc
            if r is not None:
                out = (r.stdout or "").strip()
                data = None
                if out:
                    try:
                        data = json.loads(out)
                    except ValueError:
                        data = None
                if r.returncode == 0 and isinstance(data, dict) and not data.get("is_error") and data.get("result") is not None:
                    break
                message = ""
                if isinstance(data, dict):
                    message = str(data.get("result") or data.get("error") or data.get("subtype") or "")
                message = message or (r.stderr or "")[-STDERR_TAIL:] or out[-STDERR_TAIL:] or f"exit code {r.returncode}"
                retry, what = self._classify(message)
            failures.append(what)
            if not retry:
                raise StopRun(f"Claude Code stopped the run: {what}")
            if attempt + 1 >= tries:
                raise StopRun(f"Claude Code kept failing after {tries} tries; last: {what}. Resume later with `python -m core.loop resume`.")
            wait = self._wait(attempt)
            self.log(f"  Claude Code: {what}; trying again in {wait:.0f} s")
            self.sleep(wait)

        if where is not None:
            Path(where).mkdir(parents=True, exist_ok=True)
            tmp = Path(where) / "response_raw.json.tmp"
            tmp.write_text(json.dumps(data))
            tmp.replace(Path(where) / "response_raw.json")
        usage = data.get("usage") or {}
        meta = {
            "backend": self.name,
            "source": data.get("session_id"),
            "model": self.cfg["agent_model"],
            "stop_reason": data.get("stop_reason") or ("end_turn" if not data.get("is_error") else "error"),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cost_usd": 0.0,  # a subscription is not billed per call
            "api_equivalent_usd": data.get("total_cost_usd"),
            "num_turns": data.get("num_turns"),
            "request": {"command": cmd[:1] + cmd[1:9]},
            "failed_tries": failures,
            "seconds": round(time.monotonic() - t0, 1),
        }
        return Response(text=str(data["result"]), meta=meta)
