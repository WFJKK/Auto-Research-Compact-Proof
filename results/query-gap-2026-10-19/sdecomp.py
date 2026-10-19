"""A proof for max (or min) of n tokens with leaves where the query varies, built with the checker as an oracle.

    python results/query-gap-2026-10-19/sdecomp.py <model folder> <network id> [--max-leaves-per-box K] [--rule DIR]

The decomposition by the set S of positions holding the largest token M: for each M and each nonempty set S,
one box with the positions in S at M and every other position (the query included) below M. These boxes
partition the inputs with largest token M, and the tree reaches them by splitting each position in turn at M
(below M, or M); when every position is below M, the same again with M - 1. So the tree has 2^n - 1 boxes per
M, about 15 V at n = 4, against about (n - 1) V^2 / 2 for the peeling tree with the query fixed
(results/convex-rule-2026-10-19/peel.py). For min tasks the token order is reversed.

Each box is offered to the rule; a box it cannot certify is split by peeling the top token off its widest
varying position (the near-maximal tokens compete for attention), up to K leaves per box, and what is left is
skipped. With --depth D, a box whose leaf costs more than --dear (default 50 V) is also split where that is
cheaper: peeling the top token off any of its varying positions, both halves solved the same way, D levels deep.
The checker decides the splits here, which a recipe cannot; a recipe would compute the same bounds in floats.
The search's work is not part of the proof: the checker counts only the final tree.
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
    def __init__(self, folder, nid, max_leaves, rule, depth=0, dear=None):
        self.folder, self.nid, self.rule = folder, nid, rule
        self.depth = depth  # with depth > 0, a box dearer than `dear` is split where its parts cost less (cheapest)
        self.solved = {}
        e = network_entry(folder, nid)
        self.setting = e["setting"]
        self.sizes = folder.sizes(self.setting)
        self.V, self.n = self.sizes["d_vocab"], self.sizes["n_ctx"]
        self.dear = dear if dear is not None else 50 * self.V
        self.prog = make_program(build_any(folder.build(self.setting)), load_weights(folder, nid), Counter(), folder.input_space(self.setting))
        self.max_leaves = max_leaves
        self.reverse = "min" in folder.name
        self.stats = {"boxes": 0, "leaves": 0, "skipped": 0, "oracle_calls": 0}
        self.box_costs = {}  # id of a box's subtree -> its per-leaf charges (tables are shared, so not counted)

    def tok(self, t):
        """Task-order token t as a real token (for min tasks, t plays the role of V - 1 - t)."""
        return self.V - 1 - t if self.reverse else t

    def rng(self, lo, hi):
        a, b = self.tok(lo), self.tok(hi)
        return (min(a, b), max(a, b))

    def leaf_cost(self, ranges):
        """The leaf's per-leaf charge if the rule certifies it, else None."""
        label = self.folder.task.constant_label(ranges, self.sizes)
        assert label is not None, ranges
        self.stats["oracle_calls"] += 1
        c = Counter()
        ok, _, _ = self.rule.check_leaf(self.prog, ranges, int(label), c)
        return c.by_rule.get(RULE, 0) + 2 if ok else None  # + the leaf's tree node and its rule name

    def split(self, p, lo_child, hi_child, cut_real):
        """A split of position p at cut_real; children given in task order (low part, high part)."""
        if not self.reverse:
            return {"split": {"position": p, "cuts": [cut_real]}, "children": [lo_child, hi_child]}
        return {"split": {"position": p, "cuts": [cut_real]}, "children": [hi_child, lo_child]}

    def cut(self, t):
        """The real cut that separates task tokens below t from t itself."""
        return self.tok(t) if not self.reverse else self.tok(t) + 1

    def box(self, rt):
        self.stats["boxes"] += 1
        budget = [self.max_leaves]
        cost = [0]
        node = self.refine(rt, budget, cost)
        node = dict(node)
        node["_box"] = True
        self.box_costs[id(node)] = cost[0]
        return node

    def cheapest(self, rt, depth):
        """(subtree, its charges) certifying the task-order box rt as cheaply as found: the box as one leaf, or
        a split peeling the top token off one of its varying positions with both halves solved the same way
        (while depth allows), or None if neither certifies. Only boxes dearer than self.dear are split."""
        key = (tuple(rt), depth)
        if key in self.solved:
            return self.solved[key]
        c = self.leaf_cost([self.rng(lo, hi) for lo, hi in rt])
        best = None if c is None else ({"leaf": RULE}, c)
        if depth > 0 and (c is None or c > self.dear):
            for p in [q for q, (lo, hi) in enumerate(rt) if hi > lo]:
                lo, hi = rt[p]
                low, top = list(rt), list(rt)
                low[p], top[p] = (lo, hi - 1), (hi, hi)
                a = self.cheapest(low, depth - 1)
                b = self.cheapest(top, depth - 1) if a is not None else None
                if a is not None and b is not None and (best is None or a[1] + b[1] + 3 < best[1]):
                    best = (self.split(p, a[0], b[0], self.cut(hi)), a[1] + b[1] + 3)
        self.solved[key] = best
        return best

    def refine(self, rt, budget, cost):
        real = [self.rng(lo, hi) for lo, hi in rt]
        if budget[0] <= 0:
            self.stats["skipped"] += 1
            return {"leaf": "skip"}
        if self.depth > 0:
            found = self.cheapest(rt, self.depth)
            if found is not None:
                budget[0] -= 1
                cost[0] += found[1]
                self.stats["leaves"] += 1
                return found[0]
        c = self.leaf_cost(real)
        if c is not None:
            budget[0] -= 1
            cost[0] += c
            self.stats["leaves"] += 1
            return {"leaf": RULE}
        varying = [p for p, (lo, hi) in enumerate(rt) if hi > lo]
        if not varying:
            budget[0] -= 1
            self.stats["skipped"] += 1
            return {"leaf": "skip"}
        p = max(varying, key=lambda q: (rt[q][1] - rt[q][0], -q))
        lo, hi = rt[p]
        low, top = list(rt), list(rt)
        low[p], top[p] = (lo, hi - 1), (hi, hi)
        cost[0] += 3  # the split's node and its cut
        a = self.refine(low, budget, cost)
        b = self.refine(top, budget, cost)
        return self.split(p, a, b, self.cut(hi))

    def node(self, M, i, decided, has_m):
        """Positions before i decided (each at M or below it), positions from i on allowed 0..M (task order)."""
        if M == 0:
            return self.box([(0, 0)] * self.n)
        if i == self.n:
            return self.box(decided) if has_m else self.node(M - 1, 0, [], False)
        low = self.node(M, i + 1, decided + [(0, M - 1)], has_m)
        high = self.node(M, i + 1, decided + [(M, M)], True)
        return self.split(i, low, high, self.cut(M))

    def tree(self):
        return self.node(self.V - 1, 0, [], False)


def strip(node):
    """The tree without the builder's markers (a proof file has only split and leaf keys)."""
    if "split" in node:
        return {"split": node["split"], "children": [strip(c) for c in node["children"]]}
    return {k: v for k, v in node.items() if k in ("leaf",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("network")
    ap.add_argument("--max-leaves-per-box", type=int, default=64)
    ap.add_argument("--depth", type=int, default=0, help="split dear boxes where their parts cost less, this deep")
    ap.add_argument("--dear", type=int, default=None, help="leaf charge above which a box is split if cheaper")
    ap.add_argument("--rule", default=str(REPO / "rules" / RULE), help="the rule's folder (a candidate, say)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rule = use_candidate(Path(args.rule))
    folder = load_model_folder(REPO / "models" / args.folder)
    t0 = time.time()
    b = Builder(folder, args.network, args.max_leaves_per_box, rule, args.depth, args.dear)
    proof = {"schema": SCHEMA, "network": args.network, "tree": strip(b.tree())}
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
