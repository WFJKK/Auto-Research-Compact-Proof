"""One table: the agent, Gross et al.'s cubic proof, the best of all their proofs, and token_convex proofs.

    python results/convex-rule-2026-10-19/table.py

Medians over each setting's development networks. The agent's numbers are its four-round Q runs with
token_interval (results/gross-cubic-2026-10-16/README.md); Gross et al.'s come from cubic_results*.json and
all_proofs_*.json in results/gross-cubic-2026-10-16; token_convex's from measure_*.json here.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROSS = HERE.parent / "gross-cubic-2026-10-16"

# The agent (Fable 5.1, effort high, 4 rounds, scored on Q, with token_interval): median Q, Q_grid, cost of
# finishing and the range of its shortest full proofs, from results/gross-cubic-2026-10-16/README.md.
AGENT = {
    ("attn-max4", "v16"): (0.481, 0.481, 27, "B/23 to B/30"),
    ("attn-max4", "v24"): (0.481, 0.452, 56, "B/53 to B/76"),
    ("attn-min4", "v16"): (0.476, 0.471, 39, "B/30 to B/47"),
    ("attn-min4", "v24"): (0.495, 0.470, 99, "B/86 to B/124"),
}


def med(xs):
    return statistics.median(xs) if xs else None


def cubic(folder, setting):
    out = {}
    for p in GROSS.glob("cubic_results*.json"):
        out.update(json.loads(p.read_text()))
    rows = [v for k, v in out.items() if k.startswith(f"{folder}-{setting}-")]
    if not rows:
        return None
    return {"Q": med([r["Q"] for r in rows]), "Q_grid": med([r["Q_grid"] for r in rows]),
            "cof": med([r["cost_of_finishing"] for r in rows]), "full": med([r["B_over_length"] for r in rows]),
            "acc": med([r["normalised"] for r in rows]), "n": len(rows)}


def gross_best(folder, setting):
    p = GROSS / f"all_proofs_{folder}_{setting}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    s = [v["scores"] for v in d.values()]
    return {"Q": med([x["Q"] for x in s]), "Q_grid": med([x["Q_grid"] for x in s]), "cof": med([x["cost_of_finishing"] for x in s]), "n": len(s)}


def convex(folder, setting):
    p = HERE / f"measure_{folder}_{setting}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    v = list(d.values())
    return {"Q": med([x["Q"] for x in v]), "Q_grid": med([x["Q_grid"] for x in v]), "cof": med([x["cost_of_finishing"] for x in v]),
            "full": med([x["full_B_over_length"] for x in v]), "acc": med([x["full_share_of_correct"] for x in v]), "n": len(v)}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    lines = ["| Task, size | Proof | Full proof | Accuracy | Q | Q_grid | Cost of finishing |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for (folder, setting), (q, qg, cof, full) in AGENT.items():
        task = f"{folder.split('-')[1].replace('max', 'max of ').replace('min', 'min of ')}, {setting}"
        lines.append(f"| {task} | agent (token_interval) | {full} | 1 | {q:.3f} | {qg:.3f} | {cof} |")
        c = cubic(folder, setting)
        if c:
            lines.append(f"| | Gross cubic | B/{c['full']:.0f} | {c['acc']:.3f} | {fmt(c['Q'])} | {fmt(c['Q_grid'])} | {c['cof']:.0f} |")
        g = gross_best(folder, setting)
        if g:
            lines.append(f"| | Gross, best of all proofs ({g['n']} nets) | | | {fmt(g['Q'])} | {fmt(g['Q_grid'])} | {g['cof']:.0f} |")
        v = convex(folder, setting)
        if v:
            lines.append(f"| | token_convex (peeling, {v['n']} nets) | B/{v['full']:.0f} | {v['acc']:.3f} | {fmt(v['Q'])} | {fmt(v['Q_grid'])} | {v['cof']:.0f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
