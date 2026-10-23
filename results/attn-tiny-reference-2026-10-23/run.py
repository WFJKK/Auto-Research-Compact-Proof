"""The hand reference proofs on the tiny attention networks, scored as the agent's proofs are.

    python results/attn-tiny-reference-2026-10-23/run.py [--gbmi <checkout of JasonGross/guarantees-based-mechanistic-interpretability>]

For the development network of each tiny attention model (tests/attn_{max2,min2,max4,min4}_tiny), the bar an
open-lane pilot on it has to clear:

- brute force: the legacy lane's proof file (B, checked by the Python checker) and the open lane's (the single
  charge per input times the inputs, plus E: the open lane has no tree or read charge);
- the token_convex peeling proof (results/convex-rule-2026-10-19/peel.py, the legacy lane's best attention
  proof), full and partial (the boxes with the most inputs per leaf first, the rest skipped), checked by the
  Python checker with token_convex allowed;
- Gross et al.'s cubic proof (results/gross-cubic-2026-10-16/run_cubic.py: their code unchanged, in float64 and
  not checked by our checker, its length counted in our units), when their checkout is given.

Q, Q_exact, Q_grid and cost of finishing come from core.scoring with brute force at B as the last point and the
single rule's exact charge per input as the fallback cost, as in a run. Results go to results.json next to this
file, updated after each network (resumable).
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
sys.path.insert(0, str(REPO / "results" / "convex-rule-2026-10-19"))
sys.path.insert(0, str(REPO / "results" / "gross-cubic-2026-10-16"))

from core import helpers  # noqa: E402
from core.check.checker import check_proof  # noqa: E402
from core.check.plugins import use_candidate  # noqa: E402
from core.check.proof import SCHEMA  # noqa: E402
from core.model_folder import load_model_folder  # noqa: E402
from core.scoring import network_metrics  # noqa: E402
from core.zoo import load_weights, networks  # noqa: E402

import measure  # noqa: E402  (results/convex-rule-2026-10-19: box_subtrees, strip)
import peel  # noqa: E402

MODELS = ["tests/attn_max2_tiny", "tests/attn_min2_tiny", "tests/attn_max4_tiny", "tests/attn_min4_tiny"]
RULES = ("single", "token_interval", "token_convex", "skip")
KNOBS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def metrics(points: list[dict], e: dict, brute: dict) -> dict:
    pts = [{"length": p["length"], "certified": p["certified"], "n_inputs": p["n_inputs"]} for p in points]
    pts.append({"length": e["B"], "certified": e["real_correct"], "n_inputs": e["n_inputs"]})
    m = network_metrics(pts, e, records=[brute])
    return {k: m[k] for k in ("Q", "Q_exact", "Q_grid", "cost_of_finishing", "shortest_full")}


def brute_force(folder, e) -> tuple[dict, dict]:
    proof = helpers.proof(helpers.full_tree(folder.input_space(e["setting"])))
    proof["network"] = e["id"]
    res = check_proof(proof, folder, e["id"], rule_set=("single", "skip"))
    if res["status"] != "ok":
        raise SystemExit(f"{e['id']}: brute force rejected: {res['reason']}")
    record = {"network": e["id"], "round": 0, "status": "ok", "length_by_rule": res["length_by_rule"],
              "certified": res["certified"]}
    single = res["length_by_rule"]["single"]
    out = {"legacy_length": res["length"], "legacy_by_rule": res["length_by_rule"], "certified": res["certified"],
           "single_per_input": single / e["n_inputs"], "open_lane_length": single + e["E"]}
    return out, record


def peeling(folder, e, rule, record) -> dict:
    t0 = time.time()
    b = peel.Builder(folder, e["id"], 64, rule)
    orig_box = b.box

    def marked(rt, orig_box=orig_box):
        node = dict(orig_box(rt))
        node["_box"] = True
        return node

    b.box = marked
    tree = b.tree()
    space = folder.input_space(e["setting"])
    boxes = measure.box_subtrees(tree, space)

    def n_leaves(node):
        return 1 if "split" not in node else sum(n_leaves(c) for c in node["children"])

    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][1] / n_leaves(boxes[i][0]))
    points = []
    for k in KNOBS:
        target = k * e["real_correct"]
        keep, total = set(), 0
        for i in order:
            if total >= target and k < 1:
                break
            keep.add(i)
            total += boxes[i][1]
        saved = []
        for i, (node, _) in enumerate(boxes):
            if i not in keep:
                saved.append((node, dict(node)))
                node.clear()
                node.update({"leaf": "skip"})
        proof = {"schema": SCHEMA, "network": e["id"], "tree": measure.strip(tree)}
        res = check_proof(proof, folder, e["id"], rule_set=RULES)
        for node, old in saved:
            node.clear()
            node.update(old)
        if res["status"] != "ok":
            raise SystemExit(f"{e['id']}: peeling proof rejected: {res['reason']}")
        points.append({"knob": k, "length": res["length"], "certified": res["certified"], "n_inputs": res["n_inputs"],
                       "length_by_rule": res["length_by_rule"]})
    full = points[-1]
    return {"points": points, "full_length": full["length"], "full_certified": full["certified"], "boxes": len(boxes),
            **metrics(points, e, record), "seconds": round(time.time() - t0, 1)}


def cubic(folder, e, their, record) -> dict:
    import run_cubic

    s = folder.sizes(e["setting"])
    V, n, d, dh = s["d_vocab"], s["n_ctx"], s["d_model"], s["d_head"]
    m = run_cubic.paths(load_weights(folder, e["id"]), "min" in folder.name)
    extreme = their["compute_extreme_softmaxed_right_attention_cubic_simple"](
        EQKE=m["EQKE"], EQKP=m["EQKP"], attn_scale=m["attn_scale"], pbar=run_cubic._NoBar())
    wrong = their["compute_largest_wrong_logit_cubic"](extreme, EUPU=m["EUPU"], EVOU=m["EVOU"], PVOU=m["PVOU"])
    certified = int(their["count_correct_sequences_cubic"](wrong))
    if certified > e["real_correct"]:
        raise SystemExit(f"{e['id']}: their bound {certified} exceeds the true count {e['real_correct']}")
    L = run_cubic.length_in_our_units(V, n, d, dh, e["E"])
    point = {"length": L, "certified": certified, "n_inputs": e["n_inputs"]}
    return {"length": L, "certified": certified, **metrics([point], e, record)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gbmi", type=Path, help="a checkout of JasonGross/guarantees-based-mechanistic-interpretability")
    args = ap.parse_args()
    rule = use_candidate(REPO / "rules" / "token_convex")
    their = None
    if args.gbmi:
        import run_cubic

        their = run_cubic.load_their_cubic(args.gbmi)
    path = HERE / "results.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for model in MODELS:
        folder = load_model_folder(REPO / model)
        for e in networks(folder):
            if e["split"] != "dev":
                continue
            row = done.get(e["id"], {"model": model, "setting": e["setting"], "n_inputs": e["n_inputs"], "E": e["E"],
                                     "B": e["B"], "real_correct": e["real_correct"]})
            brute, record = brute_force(folder, e)
            row["brute_force"] = brute
            if "peeling" not in row:
                row["peeling"] = peeling(folder, e, rule, record)
            if their is not None and "cubic" not in row:
                row["cubic"] = cubic(folder, e, their, record)
            done[e["id"]] = row
            path.write_text(json.dumps(done, indent=1))
            p, c = row["peeling"], row.get("cubic")
            print(f"{e['id']}: B {e['B']:,}; open-lane brute force {brute['open_lane_length']:,}; peeling full "
                  f"{p['full_length']:,} ({p['full_certified']}/{e['n_inputs']}), Q_exact {p['Q_exact']:.4f}, "
                  f"cost of finishing {p['cost_of_finishing']:.2f}"
                  + (f"; cubic {c['length']:,} ({c['certified']}/{e['n_inputs']}), Q_exact {c['Q_exact']:.4f}, "
                     f"cost of finishing {c['cost_of_finishing']:.2f}" if c else ""), flush=True)


if __name__ == "__main__":
    main()
