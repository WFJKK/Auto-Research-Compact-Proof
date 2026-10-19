"""A proof that uses token_convex for max (or min) of n tokens, built with the checker as an oracle.

    python results/convex-rule-2026-10-19/peel.py <model folder> <network id> [--max-leaves-per-box K]

The tree first fixes the query (the last position). Below it, the "peeling" decomposition: with every other
position allowed 0..M, either position 0 holds M (a box: position 0 = M, the rest 0..M), or it is below M and
position 1 holds M (position 0 in 0..M-1, position 1 = M, the rest 0..M), and so on; when every position is
below M, the same again with M - 1. Each box has the constant label M (the largest token), so the tree has
about (n - 1) V boxes per query instead of V^(n-1) inputs. For min tasks the token order is reversed.

Each box is offered to token_convex; a box it cannot certify is split by peeling the top token off its widest
varying position (the near-maximal tokens are the ones that compete for attention), up to K leaves per box;
what is left after that is skipped. This uses the checker to decide splits, which a recipe cannot; a recipe
would compute the same bound in floats. The work of the search is not part of the proof; the checker counts
only the final tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from core.check.checker import check_proof  # noqa: E402
from core.check.cost import Counter  # noqa: E402
from core.check.plugins import use_candidate  # noqa: E402
from core.check.proof import SCHEMA  # noqa: E402
from core.check.trace import build_any, make_program  # noqa: E402
from core.model_folder import load_model_folder  # noqa: E402
from core.zoo import load_weights, network_entry  # noqa: E402

RULE = "token_convex"


class Builder:
    def __init__(self, folder, nid, max_leaves, rule):
        self.folder, self.nid, self.rule = folder, nid, rule
        e = network_entry(folder, nid)
        self.setting = e["setting"]
        self.sizes = folder.sizes(self.setting)
        self.V, self.n = self.sizes["d_vocab"], self.sizes["n_ctx"]
        self.prog = make_program(build_any(folder.build(self.setting)), load_weights(folder, nid), Counter(), folder.input_space(self.setting))
        self.max_leaves = max_leaves
        self.reverse = "min" in folder.name
        self.stats = {"boxes": 0, "leaves": 0, "skipped": 0, "oracle_calls": 0}

    # tokens in the task's order: for min tasks, token t plays the role of V - 1 - t
    def tok(self, t):
        return self.V - 1 - t if self.reverse else t

    def rng(self, lo, hi):
        """The range of task-order tokens lo..hi as real tokens (lo, hi)."""
        a, b = self.tok(lo), self.tok(hi)
        return (min(a, b), max(a, b))

    def certified(self, ranges):
        label = self.folder.task.constant_label(ranges, self.sizes)
        assert label is not None, ranges
        self.stats["oracle_calls"] += 1
        ok, _, _ = self.rule.check_leaf(self.prog, ranges, int(label), Counter())
        return ok

    def split(self, p, ranges, lo_child, hi_child, cut_real):
        """A split of position p's real range at cut_real; children given in task order (low part, high part)."""
        # in real token order the children are [lo, cut-1] and [cut, hi]
        if not self.reverse:
            return {"split": {"position": p, "cuts": [cut_real]}, "children": [lo_child, hi_child]}
        return {"split": {"position": p, "cuts": [cut_real]}, "children": [hi_child, lo_child]}

    def box(self, ranges_task):
        """A subtree for a box given as task-order ranges, refined until certified or out of leaves."""
        self.stats["boxes"] += 1
        budget = [self.max_leaves]
        return self.refine(ranges_task, budget)

    def refine(self, rt, budget):
        real = [self.rng(lo, hi) for lo, hi in rt]
        if budget[0] <= 0:
            self.stats["skipped"] += 1
            return {"leaf": "skip"}
        if self.certified(real):
            budget[0] -= 1
            self.stats["leaves"] += 1
            return {"leaf": RULE}
        varying = [p for p, (lo, hi) in enumerate(rt) if hi > lo]
        if not varying:
            budget[0] -= 1
            self.stats["skipped"] += 1
            return {"leaf": "skip"}
        p = max(varying, key=lambda q: (rt[q][1] - rt[q][0], -q))
        lo, hi = rt[p]
        low = list(rt)
        low[p] = (lo, hi - 1)
        top = list(rt)
        top[p] = (hi, hi)
        cut_real = self.tok(hi) if not self.reverse else self.tok(hi) + 1
        # real order: for max tasks the top token hi is the upper child; for min tasks tok(hi) is the lowest real token
        a = self.refine(low, budget)
        b = self.refine(top, budget)
        return self.split(p, rt, a, b, cut_real)

    def peel(self, q, M):
        """Positions 0..n-2 each allowed task tokens 0..M, query (task token) q at position n-1."""
        k = self.n - 1
        if M <= q:
            rt = [(0, M)] * k + [(q, q)]
            return self.box(rt)

        def level(i):
            # positions < i are below M, position i is M or below, positions > i are 0..M
            if i == k:
                return self.peel(q, M - 1)
            at_m = [(0, M - 1)] * i + [(M, M)] + [(0, M)] * (k - 1 - i) + [(q, q)]
            node_m = self.box(at_m)
            node_below = level(i + 1)
            cut_real = self.tok(M) if not self.reverse else self.tok(M) + 1
            return self.split(i, None, node_below, node_m, cut_real)

        return level(0)

    def tree(self):
        V = self.V
        children = [None] * V
        for q in range(V):
            children[self.tok(q)] = self.peel(q, V - 1)
        return {"split": {"position": self.n - 1, "cuts": list(range(1, V))}, "children": children}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("network")
    ap.add_argument("--max-leaves-per-box", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rule = use_candidate(REPO / "rules" / RULE)
    folder = load_model_folder(REPO / "models" / args.folder)
    t0 = time.time()
    b = Builder(folder, args.network, args.max_leaves_per_box, rule)
    proof = {"schema": SCHEMA, "network": args.network, "tree": b.tree()}
    t1 = time.time()
    res = check_proof(proof, folder, args.network, rule_set=("single", "token_interval", RULE, "skip"))
    t2 = time.time()
    e = network_entry(folder, args.network)
    out = {
        "network": args.network, "status": res["status"], "reason": res["reason"],
        "certified": res["certified"], "real_correct": e["real_correct"], "n_inputs": res["n_inputs"],
        "length": res["length"], "length_by_rule": res["length_by_rule"], "B": e["B"], "E": e["E"],
        "B_over_length": e["B"] / res["length"], "leaves": res["leaves"], "build": b.stats,
        "seconds_build": round(t1 - t0, 1), "seconds_check": round(t2 - t1, 1),
    }
    print(json.dumps(out))
    if args.out:
        Path(args.out).write_text(json.dumps(proof))


if __name__ == "__main__":
    main()
