"""The four attention model folders (Gross et al.'s one-head transformer on max and min of 2 and 4 tokens):
they load, the general evaluator accepts the model, exact single checks agree with the float network, interval
leaves are sound, and brute force on the smallest setting certifies what the manifest says at length B."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from core.check.checker import brute_force_proof, check_proof
from core.check.cost import Counter
from core.check.trace import build_any, make_program
from core.model_folder import load_model_folder
from core.zoo import load_module, load_weights, networks

REPO = Path(__file__).resolve().parent.parent
FOLDERS = ["attn-max4", "attn-min4", "attn-max2", "attn-min2"]


@pytest.fixture(scope="module", params=FOLDERS)
def attn(request):
    return load_model_folder(REPO / "models" / request.param)


def _first(folder):
    return networks(folder)[0]


def test_folder_loads_with_the_paper_architecture_and_a_trained_zoo(attn):
    e = _first(attn)
    sizes = attn.sizes(e["setting"])
    assert set(sizes) == {"n_ctx", "d_vocab", "d_model", "d_head"} and sizes["d_model"] == sizes["d_head"]
    assert e["E"] and e["B"] and e["real_accuracy"] > 0.99
    names = set(load_weights(attn, e["id"]))
    assert names == {"E", "P", "WQ", "WK", "WV", "WO", "U"}  # the paper's parameters, nothing else


def test_single_checks_agree_with_the_float_network(attn):
    e = _first(attn)
    sizes = attn.sizes(e["setting"])
    space = attn.input_space(e["setting"])
    prog = make_program(build_any(attn.build(e["setting"])), load_weights(attn, e["id"]), Counter(), space)
    module = load_module(attn, e["id"])
    rng = np.random.default_rng(1)
    toks = rng.integers(0, sizes["d_vocab"], size=(200, sizes["n_ctx"]))
    with torch.no_grad():
        logits = module(attn.encode(toks, e["setting"]).to(next(module.parameters()).dtype)).numpy()
    labels = attn.task.label(toks, sizes)
    for t, z, lab in zip(toks, logits, labels):
        ok, worst, _ = prog.run_interval([(int(v), int(v)) for v in t], int(lab), Counter())
        margin = z[lab] - np.delete(z, lab).max()
        assert float(worst) <= margin + 1e-4 and ok == (margin > 1e-4 or (ok and margin > 0))


def test_interval_leaves_are_sound_against_the_float_network(attn):
    e = _first(attn)
    sizes = attn.sizes(e["setting"])
    space = attn.input_space(e["setting"])
    prog = make_program(build_any(attn.build(e["setting"])), load_weights(attn, e["id"]), Counter(), space)
    module = load_module(attn, e["id"])
    rng = np.random.default_rng(2)
    V, n = sizes["d_vocab"], sizes["n_ctx"]
    certified = 0
    for _ in range(40):
        ranges = []
        for _p in range(n):
            lo = int(rng.integers(0, V))
            hi = min(V - 1, lo + int(rng.integers(0, 4)))
            ranges.append((lo, hi))
        lab = attn.task.constant_label(ranges, sizes)
        if lab is None:
            continue
        toks = np.array(list(itertools.product(*[range(lo, hi + 1) for lo, hi in ranges])))
        with torch.no_grad():
            z = module(attn.encode(toks, e["setting"]).to(next(module.parameters()).dtype)).numpy()
        true = (z[:, lab] - np.delete(z, lab, axis=1).max(axis=1)).min()
        good, worst, _ = prog.run_interval(ranges, int(lab), Counter())
        assert float(worst) <= true + 1e-4
        certified += good
    assert certified > 0


def test_brute_force_certifies_the_manifest_count_at_length_b():
    folder = load_model_folder(REPO / "models" / "attn-max4")
    e = next(x for x in networks(folder) if x["setting"] == "v16")
    r = check_proof(brute_force_proof(folder, e["id"]), folder, e["id"])
    assert r["status"] == "ok" and r["certified"] == e["real_correct"] and r["length"] == e["B"]
