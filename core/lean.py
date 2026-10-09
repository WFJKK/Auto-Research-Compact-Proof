"""Lean spot checks: when a recipe enters the frontier, check a proof file or two in Lean as well.

A model folder opts in with `lean_check` in its config.yaml: the command that
checks one proof file, as a list of strings in which {proof} stands for a JSON
proof file and {weights} for the network's .npz weights. It runs in the model
folder's lean/ folder (or the model folder), and its last line of output is a
JSON object with "certified": how many inputs the Lean checker certifies. If
Lean and the Python checker disagree, the run halts and says where.

Every spot check is appended to the run's lean.jsonl. No model has a Lean
checker yet, so every result is still labelled Python-checked.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
from pathlib import Path

from .archive import knob_label
from .zoo import network_entry, weights_path

SPOT_CHECKS = 2
TIMEOUT_S = 3600


class LeanError(RuntimeError):
    pass


def lean_command(folder) -> list[str] | None:
    cmd = folder.config.get("lean_check")
    return [str(c) for c in cmd] if cmd else None


def run_lean(folder, network_id: str, proof_path: Path) -> int:
    cmd = lean_command(folder)
    if cmd is None:
        raise LeanError(f"{folder.name} has no Lean checker")
    weights = weights_path(folder, network_entry(folder, network_id))
    args = [c.replace("{proof}", str(proof_path)).replace("{weights}", str(weights)) for c in cmd]
    cwd = folder.path / "lean" if (folder.path / "lean").is_dir() else folder.path
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeanError(f"the Lean checker did not run: {exc}") from exc
    if r.returncode != 0:
        raise LeanError(f"the Lean checker failed (exit code {r.returncode}): {r.stderr[-1000:]}")
    try:
        return int(json.loads(r.stdout.strip().splitlines()[-1])["certified"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise LeanError(f"the Lean checker's output has no certified count: {r.stdout[-300:]!r}") from exc


def spot_check(ctx, k: int, records: list[dict]) -> list[dict]:
    """Check up to SPOT_CHECKS of these new frontier records in Lean; halt on any disagreement."""
    results = []
    for rec in records[:SPOT_CHECKS]:
        gz = ctx.run.round_path(k) / f"attempt_{rec['attempt']}" / "proofs" / f"{rec['network']}__{knob_label(rec['knob'])}.json.gz"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_bytes(gzip.decompress(gz.read_bytes()))
            lean = run_lean(ctx.folder, rec["network"], path)
        results.append({"record_id": rec["record_id"], "python": rec["certified"], "lean": lean, "agree": lean == rec["certified"]})
    if results:
        with open(ctx.run.path / "lean.jsonl", "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        ctx.log(f"  Lean spot check: {sum(r['agree'] for r in results)} of {len(results)} agree")
    bad = [r for r in results if not r["agree"]]
    if bad:
        raise LeanError(
            "Lean and the Python checker disagree, so the run halts: "
            + "; ".join(f"{r['record_id']}: Python {r['python']}, Lean {r['lean']}" for r in bad)
        )
    return results
