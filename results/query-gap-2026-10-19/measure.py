"""What leaves with a varying query buy: S-decomposition proofs on the development networks, scored like the agent's
and Gross et al.'s.

    python results/query-gap-2026-10-19/measure.py <folder:setting> [network ...] [--rule DIR] [--knobs ...]

Per network: the proof of sdecomp.py (the checker as the oracle that decides splits), then partial versions that keep
the boxes with the most inputs per unit of their own leaves' charges until a share of the correct inputs is reached and
skip the rest (so a few points trace the curve for Q, as an agent's knob would), each checked by check_proof with
token_convex allowed. Q, Q_grid and cost of finishing come from core.scoring with brute force at B as the last point,
as for every other proof here. Results go to measure_<folder>_<setting>.json, updated after each network (resumable).
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
sys.path.insert(0, str(HERE))

from core.check.checker import check_proof  # noqa: E402
from core.check.plugins import use_candidate  # noqa: E402
from core.check.proof import SCHEMA  # noqa: E402
from core.model_folder import load_model_folder  # noqa: E402
from core.scoring import network_metrics  # noqa: E402
from core.zoo import network_entry, networks  # noqa: E402

import sdecomp  # noqa: E402

RULES = ("single", "token_interval", "token_convex", "skip")
KNOBS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0)


def box_subtrees(tree, space):
    """Every subtree the builder marked as a box, with its size (inputs)."""
    out = []

    def visit(node, ranges):
        if node.get("_box"):
            size = 1
            for lo, hi in ranges:
                size *= hi - lo + 1
            out.append((node, size))
            return
        if "split" in node:
            p, cuts = node["split"]["position"], node["split"]["cuts"]
            lo, hi = ranges[p]
            bounds = [lo] + cuts + [hi + 1]
            for i, child in enumerate(node["children"]):
                sub = list(ranges)
                sub[p] = (bounds[i], bounds[i + 1] - 1)
                visit(child, sub)

    visit(tree, [(0, v - 1) for v in space])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("networks", nargs="*")
    ap.add_argument("--knobs", default=",".join(str(k) for k in KNOBS))
    ap.add_argument("--max-leaves-per-box", type=int, default=64)
    ap.add_argument("--depth", type=int, default=0, help="sdecomp.py's --depth: split dear boxes where cheaper")
    ap.add_argument("--dear", type=int, default=None, help="sdecomp.py's --dear")
    ap.add_argument("--rule", default=str(REPO / "rules" / "token_convex"))
    ap.add_argument("--tag", default="", help="suffix of the results file (measure_<folder>_<setting><tag>.json)")
    args = ap.parse_args()
    knobs = [float(k) for k in args.knobs.split(",")]
    name, setting = args.which.split(":")
    rule = use_candidate(Path(args.rule))
    folder = load_model_folder(REPO / "models" / name)
    space = folder.input_space(setting)
    ids = args.networks or [e["id"] for e in networks(folder, setting=setting)]
    path = HERE / f"measure_{name}_{setting}{args.tag}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for nid in ids:
        if nid in done:
            print(f"{nid}: done before", flush=True)
            continue
        e = network_entry(folder, nid)
        t0 = time.time()
        b = sdecomp.Builder(folder, nid, args.max_leaves_per_box, rule, args.depth, args.dear)
        tree = b.tree()
        build_s = time.time() - t0
        boxes = box_subtrees(tree, space)
        cost = [max(1, b.box_costs.get(id(node), 1)) for node, _ in boxes]
        certified = [not all(leaf == "skip" for leaf in _leaves(node)) for node, _ in boxes]
        order = sorted(range(len(boxes)), key=lambda i: -boxes[i][1] / cost[i])
        points = []
        for k in knobs:
            target = k * e["real_correct"]
            keep, total = set(), 0
            for i in order:
                if total >= target and k < 1:
                    break
                keep.add(i)
                total += boxes[i][1] if certified[i] else 0
            saved = []
            for i, (node, _) in enumerate(boxes):
                if i not in keep:
                    saved.append((node, dict(node)))
                    node.clear()
                    node.update({"leaf": "skip"})
            proof = {"schema": SCHEMA, "network": nid, "tree": sdecomp.strip(tree)}
            res = check_proof(proof, folder, nid, rule_set=RULES)
            for node, old in saved:
                node.clear()
                node.update(old)
            if res["status"] != "ok":
                raise SystemExit(f"{nid}: proof rejected: {res['reason']}")
            points.append({"knob": k, "length": res["length"], "certified": res["certified"], "n_inputs": res["n_inputs"],
                           "length_by_rule": res["length_by_rule"], "leaves": res["leaves"]})
        frontier_points = [{"length": p["length"], "certified": p["certified"], "n_inputs": p["n_inputs"]} for p in points]
        frontier_points.append({"length": e["B"], "certified": e["real_correct"], "n_inputs": e["n_inputs"]})
        m = network_metrics(frontier_points, e)
        full = points[-1]
        done[nid] = {
            "points": points, "Q": m["Q"], "Q_grid": m["Q_grid"], "cost_of_finishing": m["cost_of_finishing"],
            "shortest_full": m["shortest_full"], "B": e["B"], "E": e["E"], "real_correct": e["real_correct"],
            "full_B_over_length": e["B"] / full["length"], "full_share_of_correct": full["certified"] / e["real_correct"],
            "boxes": len(boxes), "build": b.stats, "seconds": round(time.time() - t0, 1), "seconds_build": round(build_s, 1),
            "rule": str(args.rule), "depth": args.depth, "dear": b.dear,
        }
        path.write_text(json.dumps(done, indent=1))
        print(f"{nid}: full proof B/{e['B'] / full['length']:.0f} certifying {full['certified'] / e['real_correct']:.4f} of correct; "
              f"Q {m['Q']:.4f}, Q_grid {m['Q_grid']:.4f}, cost of finishing {m['cost_of_finishing']:.1f} ({time.time() - t0:.0f} s)", flush=True)


def _leaves(node):
    if "split" in node:
        for c in node["children"]:
            yield from _leaves(c)
    else:
        yield node.get("leaf")


if __name__ == "__main__":
    main()
