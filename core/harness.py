"""Runs one recipe inside a sandbox folder. Not trusted: it shares its process with the recipe.

core.runner copies this file into a fresh sandbox folder next to recipe.py,
model.py, helpers.py, weights.npz and input.json, and starts it with
`python -I -B harness.py`. It sets its own resource limits before the recipe is
imported, calls make_proof(weights, info, knob), and writes out/proof.json.

Its exit code is only a hint to the runner, which trusts nothing written here:
0 the proof was written, 3 the recipe raised, 4 the proof is not a JSON object
within the size limit.

This file uses only the standard library and numpy, and imports nothing from
the core.
"""

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAX_ERROR_CHARS = 4000

EXIT_OK, EXIT_RAISED, EXIT_BAD_OUTPUT = 0, 3, 4


def _limit(name, value):
    try:
        import resource
    except ImportError:  # not available on this platform; the runner still watches from outside
        return
    if value is None or not hasattr(resource, name):
        return
    which = getattr(resource, name)
    try:
        _, hard = resource.getrlimit(which)
        v = int(value)
        if hard != resource.RLIM_INFINITY:
            v = min(v, hard)
        resource.setrlimit(which, (v, v))
    except (ValueError, OSError):
        pass


def _finish(code, message=None):
    if message is not None:
        with open(os.path.join(OUT, "error.txt"), "w") as f:
            f.write(message[-MAX_ERROR_CHARS:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _plain(obj):
    """Numpy numbers and arrays in a proof become JSON numbers and lists."""
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"a {type(obj).__name__} is not JSON-serialisable")


def main():
    with open(os.path.join(HERE, "input.json")) as f:
        job = json.load(f)
    limits = job["limits"]
    _limit("RLIMIT_DATA", limits.get("data_bytes"))
    _limit("RLIMIT_FSIZE", limits.get("file_bytes"))
    _limit("RLIMIT_CPU", limits.get("cpu_s"))
    _limit("RLIMIT_CORE", 0)
    sys.path.insert(0, HERE)  # -I leaves the script's folder off the path

    import numpy as np

    with np.load(os.path.join(HERE, "weights.npz"), allow_pickle=False) as z:
        weights = {k: z[k] for k in z.files}
    try:
        import recipe

        proof = recipe.make_proof(weights, job["info"], job["knob"])
    except BaseException:  # includes SystemExit and KeyboardInterrupt from the recipe
        _finish(EXIT_RAISED, traceback.format_exc())
    if not isinstance(proof, dict):
        _finish(EXIT_BAD_OUTPUT, f"make_proof returned a {type(proof).__name__}, not a dict")
    try:
        text = json.dumps(proof, allow_nan=False, default=_plain, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        _finish(EXIT_BAD_OUTPUT, f"the proof is not valid JSON data: {exc}")
    data = text.encode()
    if len(data) > limits["max_proof_bytes"]:
        _finish(EXIT_BAD_OUTPUT, f"the proof file has {len(data)} bytes, above the limit of {limits['max_proof_bytes']}")
    tmp = os.path.join(OUT, "proof.json.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, os.path.join(OUT, "proof.json"))
    _finish(EXIT_OK)


if __name__ == "__main__":
    main()
