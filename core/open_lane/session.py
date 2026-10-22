"""One open-lane session: the agent's workspace, its four tools, their budgets and their log.

A session lives in an attempt's folder:

    attempt_j/open_lane/
        session.json          what the tools need (paths, networks, budgets) and what was used so far
        tools.jsonl           every tool call, appended as it happens: tool, arguments, outcome, seconds
        workspace/            the agent's Lean modules (Candidate/<Name>.lean) and their compiled output
        submissions/<n>/      one folder per submit_candidate call: the candidate as submitted, receipt.json

The tools:

- lean_check(path, source, network, claimed_count, certificate): save a module in the workspace and compile
  it against the protected module of that network and claim (`accept.py compile` of the Lean project, in its
  box), returning the compiler's messages. Counts against the compile budget.
- run_python(code, network): run code in the recipe sandbox with that network's weights and info, returning
  what it printed. Same limits as a recipe.
- search_library(query, max_results): search the Lean project's sources and docs (the allowed package only).
- read_file(path, start, lines): read a stretch of one of those files, numbered.
- submit_candidate(network, claimed_count, theorem, modules, scored, kind): hand the workspace's modules to
  the acceptance service (`accept.py submit`); returns the receipt. Counts against the submission budget.

The service is the Lean project's; this file is its client and never decides anything about a proof.

Budgets persist in session.json, so a session that is started again (a resumed run) continues them rather
than starting afresh, and its earlier submissions and their receipts stay.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..runner import Limits, run_recipe
from ..sandbox import Sandbox
from ..util import REPO_ROOT

MAX_TEXT = 20000  # characters of compiler output, program output or search results returned to the agent


def excerpt(text: str, limit: int = MAX_TEXT) -> str:
    """Compiler output cut to the limit: the first errors and the last lines (one later error, a `decide`
    that cannot reduce, can be thousands of lines, and the first error is the one to fix)."""
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4
    return text[:head] + "\n[...]\n" + text[-tail:]
MAX_SOURCE = 400_000  # characters of one module
MODULE_PATH = re.compile(r"^Candidate/([A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.lean$")
SUBMIT_TIMEOUT_S = 4 * 3600
COMPILE_TIMEOUT_S = 3600


class ToolError(ValueError):
    """A tool refused the call: the message goes back to the agent."""


class Session:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state_path = self.path / "session.json"
        self.state = json.loads(self.state_path.read_text())
        self.workspace = self.path / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "out").mkdir(exist_ok=True)
        self.lean_project = Path(self.state["lean_project"])

    # creation ------------------------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, cfg: dict, model: str, entries: list[dict], run_path: Path) -> "Session":
        """Make (or reopen) the session folder of an attempt. Budgets carry over if it exists."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "session.json"
        if state_path.exists():
            return cls(path)
        ol = cfg["open_lane"]
        work = ol["work_dir"] or str(run_path / "open_lane_work")
        state = {
            "lean_project": str(Path(ol["lean_project"]).expanduser().resolve()),
            "work_dir": str(Path(work).expanduser().resolve()),
            "repo": str(REPO_ROOT),
            "model": model,
            "networks": {e["id"]: {"setting": e["setting"], "n_inputs": e["n_inputs"]} for e in entries},
            "budgets": {"lean_check": int(ol["compile_calls"]), "submit_candidate": int(ol["submissions"]),
                        "run_python": int(ol["python_calls"])},
            "used": {"lean_check": 0, "submit_candidate": 0, "run_python": 0},
            "limits": {"time_limit_s": cfg["time_limit_s"], "memory_limit_mb": cfg["memory_limit_mb"],
                       "max_proof_mb": cfg["max_proof_mb"], "threads_per_recipe": cfg["threads_per_recipe"]},
            "sandbox": cfg["sandbox"],
            "sandbox_dir": cfg["sandbox_dir"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        tmp = path / "session.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(state_path)
        return cls(path)

    def reload(self) -> None:
        """Re-read the saved state: the tools run in the MCP server's process, which saves its counters here."""
        self.state = json.loads(self.state_path.read_text())

    def _save(self) -> None:
        tmp = self.path / "session.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def _log(self, tool: str, args: dict, outcome: dict, seconds: float) -> None:
        short = {k: (v if not isinstance(v, str) or len(v) < 2000 else v[:2000] + f"... ({len(v)} chars)") for k, v in args.items()}
        line = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool": tool, "args": short, "outcome": outcome, "seconds": round(seconds, 2)}
        with open(self.path / "tools.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")

    def _spend(self, tool: str) -> None:
        used, budget = self.state["used"][tool], self.state["budgets"][tool]
        if used >= budget:
            raise ToolError(f"the {tool} budget of this session ({budget} calls) is used up")
        self.state["used"][tool] = used + 1
        self._save()

    def remaining(self) -> dict:
        return {t: self.state["budgets"][t] - self.state["used"][t] for t in self.state["budgets"]}

    # the service, called as a program: the pipeline is a client of it, never a second service ---------
    def _service(self, args: list[str], timeout_s: float) -> dict:
        cmd = [sys.executable, "-B", str(self.lean_project / "scripts" / "accept.py")] + args + \
              ["--repo", self.state["repo"], "--work", self.state["work_dir"], "--mode", "hardened"]
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "ok": False, "error": f"the service did not answer within {timeout_s:g} s"}
        try:
            return json.loads(r.stdout)
        except ValueError:
            return {"status": "error", "ok": False, "error": f"the service failed: {(r.stderr or r.stdout)[-1500:]}"}

    def _network(self, network: str) -> dict:
        if network not in self.state["networks"]:
            raise ToolError(f"{network!r} is not one of this run's networks: {sorted(self.state['networks'])}")
        return self.state["networks"][network]

    # tools -------------------------------------------------------------------------------------
    def lean_check(self, path: str, source: str, network: str, claimed_count: int, certificate: list | None = None) -> dict:
        t0 = time.monotonic()
        args = {"path": path, "source": source, "network": network, "claimed_count": claimed_count, "certificate": certificate}
        try:
            if not isinstance(path, str) or not MODULE_PATH.match(path):
                raise ToolError("path must be Candidate/<Name>.lean (identifiers only)")
            if not isinstance(source, str) or len(source) > MAX_SOURCE:
                raise ToolError(f"source must be a string of at most {MAX_SOURCE} characters")
            self._network(network)
            if isinstance(claimed_count, bool) or not isinstance(claimed_count, int) or claimed_count < 0:
                raise ToolError("claimed_count must be a natural number")
            self._spend("lean_check")
            src = self.workspace / path
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(source)
            cmd = ["compile", str(self.workspace), path, "--model", self.state["model"], "--network", network, "--claimed", str(claimed_count)]
            if certificate:
                cpath = self.path / "certificate.json"
                cpath.write_text(json.dumps(certificate))
                cmd += ["--certificate", str(cpath)]
            res = self._service(cmd, COMPILE_TIMEOUT_S)
            if "error" in res and not res.get("messages"):
                raise ToolError(str(res["error"]))
            text = (res.get("messages") or "").strip()
            out = {"ok": bool(res.get("ok")), "timed_out": bool(res.get("timed_out")), "module": res.get("module"),
                   "protected_module": res.get("protected_module"), "messages": excerpt(text) if text else "(no messages)",
                   "seconds": res.get("seconds"), "remaining": self.remaining()}
            self._log("lean_check", args, {"ok": out["ok"], "messages": out["messages"][:500]}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("lean_check", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def run_python(self, code: str, network: str) -> dict:
        t0 = time.monotonic()
        args = {"code": code, "network": network}
        try:
            if not isinstance(code, str) or len(code) > MAX_SOURCE:
                raise ToolError("code must be a string")
            self._network(network)
            self._spend("run_python")
            from ..model_folder import load_model_folder
            from ..run_config import resolve
            from ..zoo import load_weights, network_entry

            folder = load_model_folder(resolve(self.state["model"]))
            entry = network_entry(folder, network)
            weights = load_weights(folder, network)
            info = {"sizes": folder.sizes(entry["setting"]), "input_space": folder.input_space(entry["setting"])}
            lim = self.state["limits"]
            limits = Limits.from_config({"time_limit_s": lim["time_limit_s"], "memory_limit_mb": lim["memory_limit_mb"],
                                         "max_proof_mb": lim["max_proof_mb"], "threads_per_recipe": lim["threads_per_recipe"]})
            recipe = WRAPPER.replace("@@CODE@@", repr(code))
            from ..sandbox import choose, protected_paths

            sb = choose({"sandbox": self.state["sandbox"], "sandbox_image": "auto-research-compact-proof-sandbox:1"},
                        protected_paths(None, folder, self.path.parent))
            ex = run_recipe(recipe, folder.source(), weights, info, 0.0, limits, self.state["sandbox_dir"], sandbox=sb)
            if ex.status == "ok" and isinstance(ex.proof, dict):
                output = str(ex.proof.get("stdout", ""))
            else:
                output = f"({ex.status}) {ex.error or ''}"
            out = {"status": ex.status, "output": output[-MAX_TEXT:], "runtime_s": ex.runtime_s, "remaining": self.remaining()}
            self._log("run_python", args, {"status": ex.status, "output": output[:500]}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("run_python", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    # the examples the prompt already shows (the verified and the scored brute force) and nothing else under
    # examples/: the other examples are earlier proofs of the pilot's own networks (the agent's pilot proof,
    # its hand port, review fixtures), and a pilot that could read them would show copying, not research
    READABLE = ("CompactCheck/", "docs/", "examples/open_lane/max2_tiny_brute/", "examples/open_lane/max2_tiny_scored/",
                "accept/", "README.md", "lean-toolchain", "lakefile.toml")
    MAX_READ_LINES = 400

    def read_file(self, path: str, start: int = 1, lines: int = 200) -> dict:
        """A file of the Lean project (its sources, docs, the two examples of the prompt and the allowlist),
        by line range."""
        t0 = time.monotonic()
        args = {"path": path, "start": start, "lines": lines}
        try:
            if not isinstance(path, str) or ".." in path or path.startswith("/") or not path.startswith(self.READABLE):
                raise ToolError(f"path must be inside the Lean project's {', '.join(self.READABLE)}")
            p = self.lean_project / path
            if p.is_symlink() or not p.is_file():
                raise ToolError(f"no file {path} in the Lean project")
            all_lines = p.read_text().splitlines()
            start = max(1, int(start or 1))
            n = max(1, min(int(lines or 200), self.MAX_READ_LINES))
            chunk = all_lines[start - 1:start - 1 + n]
            text = "\n".join(f"{start + i}: {ln}" for i, ln in enumerate(chunk))
            out = {"path": path, "total_lines": len(all_lines), "start": start, "end": start + len(chunk) - 1,
                   "text": text[:MAX_TEXT], "truncated": len(chunk) < len(all_lines) - start + 1 or len(text) > MAX_TEXT}
            self._log("read_file", args, {"lines": len(chunk)}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("read_file", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def search_library(self, query: str, max_results: int = 40) -> dict:
        t0 = time.monotonic()
        args = {"query": query, "max_results": max_results}
        try:
            if not isinstance(query, str) or not query.strip():
                raise ToolError("query must be a non-empty string (a regular expression)")
            try:
                rx = re.compile(query)
            except re.error as exc:
                raise ToolError(f"not a regular expression: {exc}")
            n = max(1, min(int(max_results or 40), 200))
            hits = []
            files = sorted((self.lean_project / "CompactCheck").glob("*.lean")) + sorted((self.lean_project / "docs").glob("*.md"))
            for f in files:
                try:
                    lines = f.read_text().splitlines()
                except OSError:
                    continue
                for i, line in enumerate(lines, 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(self.lean_project)}:{i}: {line.rstrip()[:240]}")
                        if len(hits) >= n:
                            break
                if len(hits) >= n:
                    break
            text = "\n".join(hits) if hits else "(no match)"
            out = {"matches": len(hits), "text": text[:MAX_TEXT], "truncated": len(hits) >= n}
            self._log("search_library", args, {"matches": len(hits)}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("search_library", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def submit_candidate(self, network: str, claimed_count: int, theorem: str, modules: list, scored: dict | None = None,
                         kind: str = "performance") -> dict:
        t0 = time.monotonic()
        args = {"network": network, "claimed_count": claimed_count, "theorem": theorem, "modules": modules, "scored": scored, "kind": kind}
        try:
            self._network(network)
            if kind not in ("performance", "auxiliary"):
                raise ToolError("kind must be performance (a proof of Target) or auxiliary (a lemma)")
            if not isinstance(modules, list) or not modules or not all(isinstance(m, str) and MODULE_PATH.match(m) for m in modules):
                raise ToolError("modules must be a list of Candidate/<Name>.lean paths saved with lean_check")
            for m in modules:
                if not (self.workspace / m).is_file():
                    raise ToolError(f"{m} is not in the workspace; save it with lean_check first")
            if not isinstance(theorem, str) or not theorem.startswith("Candidate."):
                raise ToolError("theorem must be the full name of the theorem proving Target, under Candidate.")
            self._spend("submit_candidate")
            n = len(list((self.path / "submissions").glob("*"))) if (self.path / "submissions").is_dir() else 0
            cdir = self.path / "submissions" / f"{n:03d}"
            (cdir / "Candidate").mkdir(parents=True)
            for m in modules:
                dst = cdir / m
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((self.workspace / m).read_bytes())
            manifest = {"schema": "open-lane-candidate-v1", "candidate_id": f"{self.path.parent.name}-s{n:03d}", "kind": kind,
                        "model": self.state["model"], "network": network, "claimed_count": claimed_count,
                        "theorem": theorem, "modules": modules}
            if scored is not None:
                manifest["scored"] = scored
            (cdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            # hardened: Landlock where it exists, else process mode, which the receipt records as weak
            receipt = self._service(["submit", str(cdir)], SUBMIT_TIMEOUT_S)
            if "status" not in receipt:
                receipt = {"status": "error", "reason": receipt.get("error", "no receipt")}
            if receipt.get("status") == "timeout" and "reason" not in receipt:
                receipt["reason"] = receipt.get("error")
            receipt["submission"] = n
            (cdir / "receipt.json").write_text(json.dumps(receipt, indent=1))
            out = {k: receipt.get(k) for k in ("status", "reason", "claim_kind", "certified_count", "domain_size", "claimed_count",
                                                "network", "statement", "axioms", "score_eligible", "execution_assurance", "scoring", "seconds")}
            out["submission"] = n
            out["remaining"] = self.remaining()
            scoring = receipt.get("scoring") or {}
            self._log("submit_candidate", args, {"status": receipt.get("status"), "scoring": scoring.get("status"),
                                                 "reason": (receipt.get("reason") or scoring.get("reason") or "")[:300],
                                                 "length": scoring.get("length"), "certified": receipt.get("certified_count")}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("submit_candidate", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def receipts(self) -> list[dict]:
        out = []
        sub = self.path / "submissions"
        if sub.is_dir():
            for d in sorted(sub.iterdir()):
                p = d / "receipt.json"
                if p.is_file():
                    try:
                        out.append(json.loads(p.read_text()))
                    except ValueError:
                        out.append({"status": "error", "reason": f"unreadable receipt {p}", "submission": int(d.name)})
        return out

    # the tool table -------------------------------------------------------------------------
    TOOLS = {
        "lean_check": {
            "description": "Save a Lean module in your workspace and compile it against the protected module of a network "
                           "and a claimed count (Protected.N_<network>_c<claimed>, which the service builds; with a "
                           "certificate the name carries its hash). Returns the compiler's messages (errors, goals, "
                           "#eval output). Each call counts against the compile budget.",
            "schema": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Candidate/<Name>.lean"},
                "source": {"type": "string", "description": "the module's full source"},
                "network": {"type": "string"},
                "claimed_count": {"type": "integer"},
                "certificate": {"type": "array", "items": {}, "description": "optional: numbers (integers, floats or [p, q]) for a scored proof"}},
                "required": ["path", "source", "network", "claimed_count"]},
        },
        "run_python": {
            "description": "Run Python code in the recipe sandbox with `weights` (a dict of numpy arrays, the network's "
                           "state_dict), `info` (sizes and input_space) and `np` in scope; returns what it printed. "
                           "No network access; the recipe time and memory limits apply.",
            "schema": {"type": "object", "properties": {"code": {"type": "string"}, "network": {"type": "string"}},
                       "required": ["code", "network"]},
        },
        "search_library": {
            "description": "Search the Lean project's sources (CompactCheck/*.lean) and docs with a regular expression, "
                           "matched against each line's text (not against file names or line numbers); returns the "
                           "matching lines as file:line: text. To read a stretch of a file, use read_file.",
            "schema": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                       "required": ["query"]},
        },
        "read_file": {
            "description": "Read lines of a file of the Lean project: its sources (CompactCheck/<Name>.lean), docs "
                           "(docs/COUNTED.md, docs/OPEN_LANE.md), the two examples of your prompt "
                           "(examples/open_lane/max2_tiny_brute/..., examples/open_lane/max2_tiny_scored/...) and the "
                           "allowlist (accept/allowlist.txt); no other example. Returns up to 400 numbered lines from `start`.",
            "schema": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"},
                                                        "lines": {"type": "integer"}},
                       "required": ["path"]},
        },
        "submit_candidate": {
            "description": "Submit modules from your workspace to the acceptance service as a proof that the network "
                           "answers at least claimed_count inputs correctly (theorem : Target). Returns the receipt: "
                           "status accepted or rejected with the reason, the axioms, and, with `scored`, the scoring "
                           "block (status scored or refused, the bill and the length). Each call counts against the "
                           "submission budget.",
            "schema": {"type": "object", "properties": {
                "network": {"type": "string"}, "claimed_count": {"type": "integer"},
                "theorem": {"type": "string", "description": "Candidate.<...>.<name> proving Target"},
                "modules": {"type": "array", "items": {"type": "string"}, "description": "paths saved with lean_check, imports first"},
                "scored": {"type": "object", "description": "optional: {checker, sound, instance, certificate, claimed_bill}"},
                "kind": {"type": "string", "enum": ["performance", "auxiliary"],
                         "description": "performance (default): theorem proves Target; auxiliary: theorem is a lemma with any statement, accepted with no count and no score, listed to you in later rounds"}},
                "required": ["network", "claimed_count", "theorem", "modules"]},
        },
    }

    def call(self, tool: str, arguments: dict) -> dict:
        if tool not in self.TOOLS:
            raise ToolError(f"no tool {tool!r}; the tools are {sorted(self.TOOLS)}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        fn = getattr(self, tool)
        required = self.TOOLS[tool]["schema"]["required"]
        missing = [k for k in required if k not in arguments]
        if missing:
            raise ToolError(f"{tool} needs {missing}")
        allowed = set(self.TOOLS[tool]["schema"]["properties"])
        extra = sorted(set(arguments) - allowed)
        if extra:
            raise ToolError(f"{tool} does not take {extra}")
        return fn(**arguments)


WRAPPER = '''"""The open lane's run_python: the agent's code as a recipe, its output as the proof."""
import io, contextlib, traceback
import numpy as np

CODE = @@CODE@@


def make_proof(weights, info, knob):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(CODE, "<run_python>", "exec"), {"weights": weights, "info": info, "np": np, "__name__": "__main__"})
        except BaseException:
            buf.write(traceback.format_exc())
    return {"stdout": buf.getvalue()[-20000:]}
'''
