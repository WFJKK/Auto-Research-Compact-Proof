"""One open-lane session: the agent's workspace, its four tools, their budgets and their log.

A session lives in an attempt's folder:

    attempt_j/open_lane/
        session.json          what the tools need (paths, networks, budgets) and what was used so far
        tools.jsonl           every tool call, appended as it happens: tool, arguments, outcome, seconds
        workspace/            the agent's Lean modules (Candidate/<Name>.lean) and their compiled output
        submissions/<n>/      one folder per submit_candidate call: the candidate as submitted, receipt.json

The tools:

- lean_check(path, source, network, claimed_count, certificate): save a module in the workspace and compile
  it against the protected module of that network and claim (built and cached by the service), returning the
  compiler's messages. Counts against the compile budget.
- run_python(code, network): run code in the recipe sandbox with that network's weights and info, returning
  what it printed. Same limits as a recipe.
- search_library(query, max_results): search the Lean project's sources and docs (the allowed package only).
- submit_candidate(network, claimed_count, theorem, modules, scored): hand the workspace's modules to the
  acceptance service; returns the receipt. Counts against the submission budget.

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
MAX_SOURCE = 400_000  # characters of one module
MODULE_PATH = re.compile(r"^Candidate/([A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.lean$")
SUBMIT_TIMEOUT_S = 4 * 3600


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
        self._accept = None

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

    # the service's helpers ------------------------------------------------------------------
    @property
    def accept(self):
        """scripts/accept.py of the Lean project, imported once."""
        if self._accept is None:
            scripts = self.lean_project / "scripts"
            sys.path.insert(0, str(scripts))
            try:
                import accept  # noqa: F401
            finally:
                sys.path.pop(0)
            accept._repo = Path(self.state["repo"])
            self._accept = accept
        return self._accept

    def _network(self, network: str) -> dict:
        if network not in self.state["networks"]:
            raise ToolError(f"{network!r} is not one of this run's networks: {sorted(self.state['networks'])}")
        return self.state["networks"][network]

    def _protected(self, network: str, claimed_count: int, certificate) -> dict:
        """The protected module for a network, claim and certificate, built by the service (cached)."""
        self._network(network)
        if isinstance(claimed_count, bool) or not isinstance(claimed_count, int) or claimed_count < 0:
            raise ToolError("claimed_count must be a natural number")
        accept = self.accept
        work = Path(self.state["work_dir"])
        try:
            net_json = accept.export_network(Path(self.state["repo"]), self.state["model"], network, work)
            return accept.build_protected(net_json, claimed_count, work, accept.lean_binary(), accept.lean_path(),
                                          accept.box_mode("hardened"), certificate or None)
        except (accept.Reject, accept.Unsupported) as exc:
            raise ToolError(str(exc))
        except accept.Infra as exc:
            raise ToolError(f"the service could not build the protected module: {exc}")

    # tools -------------------------------------------------------------------------------------
    def lean_check(self, path: str, source: str, network: str, claimed_count: int, certificate: list | None = None) -> dict:
        t0 = time.monotonic()
        args = {"path": path, "source": source, "network": network, "claimed_count": claimed_count, "certificate": certificate}
        try:
            if not isinstance(path, str) or not MODULE_PATH.match(path):
                raise ToolError("path must be Candidate/<Name>.lean (identifiers only)")
            if not isinstance(source, str) or len(source) > MAX_SOURCE:
                raise ToolError(f"source must be a string of at most {MAX_SOURCE} characters")
            self._spend("lean_check")
            prot = self._protected(network, claimed_count, certificate)
            accept = self.accept
            src = self.workspace / path
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(source)
            module = "Candidate." + path[len("Candidate/"):-len(".lean")].replace("/", ".")
            lean, lp = accept.lean_binary(), accept.lean_path() + [prot["lean_path_dir"], str(self.workspace / "out")]
            res = accept.compile_module(self.workspace, self.workspace, module, src, lean, lp, accept.box_mode("hardened"))
            text = (res["stdout"] + res["stderr"]).strip()
            out = {"ok": res["rc"] == 0 and not res["timed_out"], "timed_out": res["timed_out"], "module": module,
                   "protected_module": prot["module"], "messages": text[-MAX_TEXT:] if text else "(no messages)",
                   "seconds": round(res["seconds"], 1), "remaining": self.remaining()}
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

    def submit_candidate(self, network: str, claimed_count: int, theorem: str, modules: list, scored: dict | None = None) -> dict:
        t0 = time.monotonic()
        args = {"network": network, "claimed_count": claimed_count, "theorem": theorem, "modules": modules, "scored": scored}
        try:
            self._network(network)
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
            manifest = {"schema": "open-lane-candidate-v1", "candidate_id": f"{self.path.parent.name}-s{n:03d}",
                        "model": self.state["model"], "network": network, "claimed_count": claimed_count,
                        "theorem": theorem, "modules": modules}
            if scored is not None:
                manifest["scored"] = scored
            (cdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            # hardened: Landlock where it exists, else process mode, which the receipt records as weak
            cmd = [sys.executable, "-B", str(self.lean_project / "scripts" / "accept.py"), "submit", str(cdir),
                   "--repo", self.state["repo"], "--work", self.state["work_dir"], "--mode", "hardened"]
            env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT_S, env=env)
            except subprocess.TimeoutExpired:
                receipt = {"status": "timeout", "reason": f"the service did not answer within {SUBMIT_TIMEOUT_S} s"}
            else:
                try:
                    receipt = json.loads(r.stdout)
                except ValueError:
                    receipt = {"status": "error", "reason": f"the service failed: {(r.stderr or r.stdout)[-1500:]}"}
            receipt["submission"] = n
            (cdir / "receipt.json").write_text(json.dumps(receipt, indent=1))
            out = {k: receipt.get(k) for k in ("status", "reason", "certified_count", "domain_size", "claimed_count", "network",
                                                "axioms", "scoring", "seconds")}
            out["submission"] = n
            out["remaining"] = self.remaining()
            self._log("submit_candidate", args, {"status": receipt.get("status"), "scoring": (receipt.get("scoring") or {}).get("status"),
                                                 "reason": (receipt.get("reason") or "")[:300]}, time.monotonic() - t0)
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
            "description": "Search the Lean project's sources (CompactCheck/*.lean) and docs with a regular expression; "
                           "returns matching lines as file:line: text.",
            "schema": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                       "required": ["query"]},
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
                "scored": {"type": "object", "description": "optional: {checker, sound, instance, certificate, claimed_bill}"}},
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
