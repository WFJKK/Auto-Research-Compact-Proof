"""The checker in its own clean process.

The loop starts one worker per attempt (core.check.worker.CheckerProcess):

    python -I -B -X pycache_prefix=<empty folder> -c <bootstrap> <model folder> <hashes>

-I ignores the environment and the user's site folder, -B writes no bytecode,
and the empty pycache prefix means compiled files lying next to the sources,
planted or not, are never read: every module is compiled from its source. The
worker gets no API key and no other environment. It first checks that the
trusted files have the hashes the run started with, then answers one request
per line on stdin,

    {"network": id, "proof": "<path to a .json.gz>", "rule_set": [...]}

with one JSON line on stdout: {"result": ..., "diagnosis": ...} or
{"error": ...}. It hashes the trusted files again before every check, and an
error stops the run.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_PROOF_BYTES = 1 << 30
STDERR_TAIL = 4000


class CheckerError(RuntimeError):
    pass


# the worker side ------------------------------------------------------------------------
def serve(folder_path: str, expected: dict) -> None:
    # Keep stdout for the protocol; anything else printed goes to stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.dont_write_bytecode = True

    from ..diagnostics import diagnose, float_margins
    from ..model_folder import load_model_folder
    from ..versions import changed
    from ..zoo import correct_mask, network_entry
    from .checker import check_proof

    folder = load_model_folder(folder_path)
    float_view: dict = {}

    def verify():
        bad = changed(expected, folder)
        if bad:
            raise CheckerError(f"trusted files changed since the run started: {', '.join(bad)}")

    try:
        verify()
    except CheckerError as exc:
        proto.write(json.dumps({"error": str(exc)}) + "\n")
        return
    proto.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        try:
            req = json.loads(line)
            verify()
            with gzip.open(req["proof"], "rb") as f:
                data = f.read(MAX_PROOF_BYTES + 1)
            if len(data) > MAX_PROOF_BYTES:
                raise CheckerError("the proof file is too large")
            nid = req["network"]
            result = check_proof(json.loads(data), folder, nid, rule_set=tuple(req["rule_set"]), detail=True)
            diagnosis = None
            if result["status"] == "ok":
                if nid not in float_view:
                    float_view[nid] = (float_margins(folder, nid), correct_mask(folder, nid))
                diagnosis = diagnose(folder, network_entry(folder, nid), result, *float_view[nid])
            result.pop("mask", None)
            result.pop("outcomes", None)
            reply = {"result": result, "diagnosis": diagnosis}
        except Exception as exc:  # reported to the driver, which halts the run
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        proto.write(json.dumps(reply) + "\n")


def main() -> None:
    serve(sys.argv[1], json.loads(sys.argv[2]))


# the driver side ------------------------------------------------------------------------
class CheckerProcess:
    """A checker worker for one model folder, started from read-only code with a clean slate."""

    def __init__(self, folder_path: str | Path, expected: dict):
        self.pycache = tempfile.mkdtemp(prefix="cpl-no-pycache-")
        self.log = tempfile.TemporaryFile()
        bootstrap = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); from core.check.worker import main; main()"
        cmd = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={self.pycache}", "-c", bootstrap]
        cmd += [str(folder_path), json.dumps(expected)]
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, text=True, cwd=self.pycache
        )
        hello = self._reply()
        if not hello.get("ready"):
            self.close()
            raise CheckerError(hello.get("error", "the checker did not start"))

    def _stderr(self) -> str:
        try:
            self.log.seek(0)
            return self.log.read().decode("utf-8", errors="replace")[-STDERR_TAIL:]
        except OSError:
            return ""

    def _reply(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait(timeout=30)
            raise CheckerError(f"the checker process stopped (exit code {self.proc.returncode}): {self._stderr()}")
        return json.loads(line)

    def check(self, network_id: str, proof_path: str | Path, rule_set) -> tuple[dict, dict | None]:
        self.proc.stdin.write(json.dumps({"network": network_id, "proof": str(proof_path), "rule_set": list(rule_set)}) + "\n")
        self.proc.stdin.flush()
        reply = self._reply()
        if "error" in reply:
            raise CheckerError(reply["error"])
        return reply["result"], reply["diagnosis"]

    def close(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        finally:
            self.log.close()
            shutil.rmtree(self.pycache, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
