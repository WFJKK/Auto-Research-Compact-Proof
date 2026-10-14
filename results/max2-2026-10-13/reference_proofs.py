"""Reference proofs for the max2 write-up: what the known strategies score on the development networks.

    python -B results/max2-2026-10-13/reference_proofs.py

Writes reference_proofs.json next to this file. Two kinds of numbers:

- Checked: baselines 02 (symmetry), 03 (the cubic proof) and 04 (the cubic proof with a
  budget knob, at knob 0, 0.25, 0.5, 0.75 and 1) run through the checker under the current
  cost model. Brute force is B and the network's real accuracy, from the zoo manifest.
- Estimated: three strategies the checker's rules cannot express. Which inputs they certify is
  computed in float64 from the weights; their lengths are estimates on the checker's scale
  (reading the weights and the proof, symmetry, the contraction as the checker charges it,
  then the charges below, which are guesses, not checked).
  - convexity_loss: the convexity bound of the max-of-2 tutorial, per input. An input counts as
    certified when its bound on the loss is below ln 2. Extra: one log-sum-exp per doubled
    input, 2V + 2 each, and 3 per sorted pair.
  - convexity_accuracy: the same argument applied to the margin: (a, b) is certified when
    lead_b(a) + lead_b(b) > 0, with lead_b(t) = M[b,t] - max over j != b of M[j,t]. Extra: the
    two largest entries of every column, 2V each, and 4 per sorted pair.
  - hybrid: convexity_accuracy where it holds and the cubic proof's exact check elsewhere,
    per run of consecutive tokens a in each group b: 3n + 3 for a run it certifies, 2Vn + 2V
    for a run checked exactly. Certifies every input the network gets right.
"""

import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))  # the recipes import helpers

from core.check.checker import check_proof  # noqa: E402
from core.check.cost import VERSION as COST_VERSION  # noqa: E402
from core.model_folder import load_model_folder  # noqa: E402
from core.scoring import cost_of_finishing, q_score  # noqa: E402
from core.zoo import load_weights  # noqa: E402

FOLDER = load_model_folder(str(ROOT / "models" / "max2"))
MANIFEST = json.loads((ROOT / "models" / "max2" / "zoo" / "manifest.json").read_text())
DEV = [e for e in MANIFEST["networks"] if e.get("split") == "dev"]
KNOBS = (0.0, 0.25, 0.5, 0.75, 1.0)


def recipe(name):
    src = (ROOT / "models" / "max2" / "baselines" / f"{name}.md").read_text()
    ns = {}
    exec(re.search(r"```python\n(.*?)```", src, re.S).group(1), ns)
    return ns["make_proof"]


def checked(make_proof, nid, w, V, knob):
    proof = make_proof(w, {"input_space": [V, V]}, knob)
    proof["network"] = nid
    r = check_proof(proof, FOLDER, nid)
    if r["status"] != "ok":
        raise SystemExit(f"{nid}: rejected: {r.get('reason')}")
    return {"length": r["length"], "certified": r["certified"], "length_by_rule": r["length_by_rule"]}


def lse(z):
    m = z.max(axis=0)
    return m + np.log(np.exp(z - m).sum(axis=0))


def estimates(w, V, base):
    """Certified counts (both orders of an off-diagonal pair) and estimated lengths."""
    M = np.asarray(w["unembedding.weight"], np.float64) @ np.asarray(w["linear.weight"], np.float64) @ np.asarray(w["embedding.weight"], np.float64)
    pairs = V * (V + 1) // 2
    # the tutorial's convexity bound, per input
    Z = 2 * M
    Ld = lse(Z)[:, None] - Z.T  # Ld[a, y]: the doubled input (a, a) scored against label y
    c_loss = sum((1 if a == b else 2) for b in range(V) for a in range(b + 1) if 0.5 * Ld[a, b] + 0.5 * Ld[b, b] < np.log(2))
    L_loss = base + V * (2 * V + 2) + 3 * V * (V - 1) // 2 + V
    # convexity for accuracy
    order = np.argsort(-M, axis=0)
    top1, top2 = M[order[0], np.arange(V)], M[order[1], np.arange(V)]

    def lead(l, t):
        return M[l, t] - (top2[t] if order[0, t] == l else top1[t])

    ok = {b: [lead(b, a) + lead(b, b) > 0 for a in range(b + 1)] for b in range(V)}
    c_acc = sum((1 if a == b else 2) for b in range(V) for a in range(b + 1) if ok[b][a])
    L_acc = base + 2 * V * V + 4 * pairs
    # hybrid: the bound where it holds, the exact check elsewhere, by runs of consecutive a
    L_hyb = base + 2 * V * V
    for b in range(V):
        a = 0
        while a <= b:
            s = a
            while a <= b and ok[b][a] == ok[b][s]:
                a += 1
            n = a - s
            L_hyb += (3 * n + 3) if ok[b][s] else (2 * V * n + 2 * V)
    return {
        "convexity_loss": {"length": int(L_loss), "certified": int(c_loss)},
        "convexity_accuracy": {"length": int(L_acc), "certified": int(c_acc)},
        "hybrid": {"length": int(L_hyb)},
    }


def pt(length, certified, n):
    return {"length": length, "certified": certified, "n_inputs": n}


def main():
    sym, cub, knob = recipe("02_symmetry"), recipe("03_contract_interval"), recipe("04_contract_interval_budget")
    per_net = {}
    for e in DEV:
        nid, E, B, real = e["id"], e["E"], e["B"], e["real_correct"]
        w = load_weights(FOLDER, nid)
        V = w["embedding.weight"].shape[1]
        n = V * V
        r = {"setting": e["setting"], "V": V, "E": E, "B": B, "real_correct": real, "n_inputs": n}
        r["brute_force"] = {"length": B, "certified": real}
        r["symmetry"] = checked(sym, nid, w, V, 1.0)
        r["cubic"] = checked(cub, nid, w, V, 1.0)
        r["cubic_knob"] = [dict(knob=k, **checked(knob, nid, w, V, k)) for k in KNOBS]
        lb = r["cubic"]["length_by_rule"]
        base = lb["extract"] + lb["read"] + lb["symmetry"] + lb["contract"] + 200
        r.update(estimates(w, V, base))
        r["hybrid"]["certified"] = real
        P = lambda d: pt(d["length"], d["certified"], n)  # noqa: E731
        sets = {
            "brute_force": [P(r["brute_force"])],
            "symmetry": [P(r["symmetry"])],
            "cubic": [P(r["cubic"])],
            "cubic_knob": [P(d) for d in r["cubic_knob"]],
            "convexity_loss": [P(r["convexity_loss"])],
            "convexity_accuracy": [P(r["convexity_accuracy"])],
            "convexity_accuracy+cubic": [P(r["convexity_accuracy"]), P(r["cubic"])],
            "hybrid": [P(r["hybrid"])],
            "convexity_accuracy+hybrid": [P(r["convexity_accuracy"]), P(r["hybrid"])],
        }
        r["scores"] = {}
        for name, pts in sets.items():
            pts = sorted(pts, key=lambda p: p["length"])
            frontier, best = [], -1
            for p in pts:
                if p["certified"] > best and p["certified"] > 0:
                    frontier.append(p)
                    best = p["certified"]
            r["scores"][name] = {"Q": q_score(frontier, E, B), "cost_of_finishing": cost_of_finishing(frontier, real, B)}
        for d in [r["symmetry"], r["cubic"], *r["cubic_knob"]]:
            d.pop("length_by_rule", None)
        per_net[nid] = r
        print(nid, {k: round(v["Q"], 3) for k, v in r["scores"].items()}, flush=True)

    medians = {}
    for st in sorted({r["setting"] for r in per_net.values()}):
        rs = [r for r in per_net.values() if r["setting"] == st]
        med = lambda xs: statistics.median(xs)  # noqa: E731
        m = {"networks": len(rs)}
        for name in ("brute_force", "symmetry", "cubic", "convexity_loss", "convexity_accuracy", "hybrid"):
            m[name] = {"length": med(r[name]["length"] for r in rs), "accuracy": med(r[name]["certified"] / r["n_inputs"] for r in rs)}
        m["cubic_knob"] = [
            {"knob": k, "length": med(r["cubic_knob"][i]["length"] for r in rs), "accuracy": med(r["cubic_knob"][i]["certified"] / r["n_inputs"] for r in rs)}
            for i, k in enumerate(KNOBS)
        ]
        m["scores"] = {name: {s: med(r["scores"][name][s] for r in rs) for s in ("Q", "cost_of_finishing")} for name in rs[0]["scores"]}
        medians[st] = m
    out = {
        "cost_model": COST_VERSION,
        "checked": ["brute_force", "symmetry", "cubic", "cubic_knob"],
        "estimated": ["convexity_loss", "convexity_accuracy", "hybrid", "convexity_accuracy+cubic", "convexity_accuracy+hybrid"],
        "medians": medians,
        "networks": per_net,
    }
    path = Path(__file__).with_name("reference_proofs.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
