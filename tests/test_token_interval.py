"""The token_interval rule (core/check/tokens.py) on the four attention model folders: its bounds are
sound against the float network on random leaves, exact up to rounding when one position varies, far
tighter than interval leaves, and its token-only rows are charged once per proof. Linear networks refuse
it, and a hand-written proof built from it certifies a whole network well below brute force."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from core.check.checker import check_proof
from core.check.cost import Counter
from core.check.proof import SCHEMA
from core.check.trace import build_any, make_program
from core.model_folder import load_model_folder
from core.zoo import load_module, load_weights, networks

REPO = Path(__file__).resolve().parent.parent
FOLDERS = ["attn-max4", "attn-min4", "attn-max2", "attn-min2"]
RULES = ("single", "interval", "token_interval", "skip")


def setup(name, setting=None):
    folder = load_model_folder(REPO / "models" / name)
    e = next(x for x in networks(folder) if setting is None or x["setting"] == setting)
    space = folder.input_space(e["setting"])
    prog = make_program(build_any(folder.build(e["setting"])), load_weights(folder, e["id"]), Counter(), space)
    return folder, e, prog, load_module(folder, e["id"])


def true_margin(folder, e, module, ranges, label):
    toks = np.array(list(itertools.product(*[range(lo, hi + 1) for lo, hi in ranges])))
    with torch.no_grad():
        z = module(folder.encode(toks, e["setting"]).to(next(module.parameters()).dtype)).numpy()
    return float((z[:, label] - np.delete(z, label, axis=1).max(axis=1)).min())


def random_leaves(folder, e, rng, n, one_varying):
    sizes = folder.sizes(e["setting"])
    V, P = sizes["d_vocab"], sizes["n_ctx"]
    out = []
    while len(out) < n:
        ranges = []
        free = int(rng.integers(0, P))
        for p in range(P):
            lo = int(rng.integers(0, V))
            width = int(rng.integers(0, V // 2)) if (p == free or not one_varying) else 0
            if not one_varying:
                width = int(rng.integers(0, 4))
            ranges.append((lo, min(V - 1, lo + width)))
        label = folder.task.constant_label(ranges, sizes)
        if label is not None:
            out.append((ranges, int(label)))
    return out


@pytest.mark.parametrize("name", FOLDERS)
def test_bounds_are_sound_and_exact_when_one_position_varies(name):
    folder, e, prog, module = setup(name)
    rng = np.random.default_rng(7)
    for ranges, label in random_leaves(folder, e, rng, 25, one_varying=True):
        ok, bound, _ = prog.run_token_interval(ranges, label, Counter())
        true = true_margin(folder, e, module, ranges, label)
        assert float(bound) <= true + 1e-4  # sound
        assert float(bound) >= true - 1e-3  # and tight: the smallest margin over the leaf's inputs
        assert ok == (float(bound) > 0)


@pytest.mark.parametrize("name", ["attn-max4", "attn-min4"])
def test_bounds_are_sound_when_several_positions_vary(name):
    folder, e, prog, module = setup(name)
    rng = np.random.default_rng(8)
    for ranges, label in random_leaves(folder, e, rng, 25, one_varying=False):
        ok, bound, _ = prog.run_token_interval(ranges, label, Counter())
        assert float(bound) <= true_margin(folder, e, module, ranges, label) + 1e-4


def test_far_tighter_than_interval_and_cheaper_than_singles():
    folder, e, prog, module = setup("attn-max2", "v64")
    ranges, label = [(0, 30), (40, 40)], 40
    c_tok, c_int = Counter(), Counter()
    tok = prog.run_token_interval(ranges, label, c_tok)
    box = prog.run_interval(ranges, label, c_int)
    assert tok[0] and not box[0] and float(box[1]) < 0
    per_single = (e["B"] - e["E"]) // folder.n_inputs(e["setting"])
    # the strip's 31 inputs for well under their singles; the token rows (tables) are shared by the whole proof
    assert c_tok.by_rule["token_interval"] < 31 * per_single / 5


def test_token_rows_are_charged_once_per_proof():
    folder, e, prog, module = setup("attn-max2", "v64")
    first, again = Counter(), Counter()
    prog.run_token_interval([(0, 20), (30, 30)], 30, first)
    prog.run_token_interval([(0, 20), (30, 30)], 30, again)
    assert first.by_rule["tables"] > 0 and again.by_rule.get("tables", 0) == 0
    assert again.by_rule["token_interval"] == first.by_rule["token_interval"]  # the per-leaf work is paid every time


def test_linear_networks_refuse_the_rule():
    folder = load_model_folder(REPO / "models" / "max2")
    e = networks(folder)[0]
    space = folder.input_space(e["setting"])
    V = space[0]
    leaf = {"split": {"position": 1, "cuts": [1]}, "children": [{"leaf": "token_interval"}, {"leaf": "skip"}]}
    tree = {"split": {"position": 0, "cuts": [1]}, "children": [leaf, {"leaf": "skip"}]}
    r = check_proof({"schema": SCHEMA, "network": e["id"], "tree": tree}, folder, e["id"], rule_set=RULES)
    assert V > 1 and r["status"] == "rejected" and "beyond linear maps" in r["reason"]


def test_the_rule_must_be_allowed_in_the_run():
    folder, e, prog, module = setup("attn-max2", "v64")
    proof = {"schema": SCHEMA, "network": e["id"], "tree": {"leaf": "token_interval"}}
    r = check_proof(proof, folder, e["id"], rule_set=("single", "interval", "skip"))
    assert r["status"] == "rejected" and "not allowed" in r["reason"]


def strip_proof(nid, V):
    """Per token a at position 0: the inputs (a, b) with b <= a share the label a, one token_interval leaf;
    the rest (b > a) are token_interval leaves of one input each."""
    children = []
    for a in range(V):
        if a == V - 1:
            children.append({"leaf": "token_interval"})
            continue
        kids = [{"leaf": "token_interval"}] + [{"leaf": "token_interval"} for _ in range(a + 1, V)]
        children.append({"split": {"position": 1, "cuts": list(range(a + 1, V))}, "children": kids})
    tree = {"split": {"position": 0, "cuts": list(range(1, V))}, "children": children}
    return {"schema": SCHEMA, "network": nid, "tree": tree}


def test_a_hand_written_strip_proof_certifies_a_network_well_below_brute_force():
    folder = load_model_folder(REPO / "models" / "attn-max2")
    e = next(x for x in networks(folder) if x["setting"] == "v64")
    V = folder.sizes("v64")["d_vocab"]
    r = check_proof(strip_proof(e["id"], V), folder, e["id"], rule_set=RULES)
    assert r["status"] == "ok" and r["certified"] == e["real_correct"]
    assert r["length"] < e["B"] / 5
