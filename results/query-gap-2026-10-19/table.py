"""One table: the agent, Gross et al.'s cubic proof, the best of all their proofs, token_convex with a fixed query
(version 1, results/convex-rule-2026-10-19) and with leaves where the query varies (version 2, here).

    python results/query-gap-2026-10-19/table.py

Medians over each setting's development networks. Every row but the last two comes from
results/convex-rule-2026-10-19/table.py, which this imports. A second table gives the share of the correct inputs
certified within a length budget (B/6,000, B/3,000, B/1,000, B/300): Gross et al.'s frontier of all their proofs against
version 2's partial proofs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v1_table", HERE.parent / "convex-rule-2026-10-19" / "table.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


def version2(folder, setting):
    p = HERE / f"measure_{folder}_{setting}.json"
    if not p.exists():
        return None
    d = list(json.loads(p.read_text()).values())
    return {"Q": v1.med([x["Q"] for x in d]), "Q_grid": v1.med([x["Q_grid"] for x in d]),
            "cof": v1.med([x["cost_of_finishing"] for x in d]), "full": v1.med([x["full_B_over_length"] for x in d]),
            "acc": v1.med([x["full_share_of_correct"] for x in d]), "n": len(d)}


SETTINGS = list(v1.AGENT) + [("attn-max4", "v64")]  # v64: the paper's size (no agent run there yet)


def main():
    lines = ["| Task, size | Proof | Full proof | Accuracy | Q | Q_grid | Cost of finishing |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for folder, setting in SETTINGS:
        task = f"{folder.split('-')[1].replace('max', 'max of ').replace('min', 'min of ')}, {setting}"
        if (folder, setting) in v1.AGENT:
            q, qg, cof, full = v1.AGENT[(folder, setting)]
            lines.append(f"| {task} | agent (token_interval) | {full} | 1 | {q:.3f} | {qg:.3f} | {cof} |")
        else:
            lines.append(f"| {task} | agent | no run yet | | | | |")
        c = v1.cubic(folder, setting)
        if c:
            lines.append(f"| | Gross cubic | B/{c['full']:.0f} | {c['acc']:.3f} | {v1.fmt(c['Q'])} | {v1.fmt(c['Q_grid'])} | {c['cof']:.0f} |")
        g = v1.gross_best(folder, setting)
        if g:
            lines.append(f"| | Gross, best of all proofs ({g['n']} nets) | | | {v1.fmt(g['Q'])} | {v1.fmt(g['Q_grid'])} | {g['cof']:.0f} |")
        a = v1.convex(folder, setting)
        if a:
            lines.append(f"| | token_convex v1, query fixed ({a['n']} nets) | B/{a['full']:.0f} | {a['acc']:.3f} | {v1.fmt(a['Q'])} | {v1.fmt(a['Q_grid'])} | {a['cof']:.0f} |")
        b = version2(folder, setting)
        if b:
            lines.append(f"| | token_convex v2, query varies ({b['n']} nets) | B/{b['full']:.0f} | {b['acc']:.3f} | {v1.fmt(b['Q'])} | {v1.fmt(b['Q_grid'])} | {b['cof']:.0f} |")
    print("\n".join(lines))
    print()
    print(budget_table())


def share_within(points, budget, real_correct):
    """The largest share of the correct inputs certified by a proof no longer than budget (0 if none is)."""
    return max([p["certified"] / real_correct for p in points if p["length"] <= budget] or [0.0])


def budget_table():
    """Median share of the correct inputs certified within B/6,000, B/3,000, B/1,000 and B/300: Gross et al.'s
    frontier of all their proofs against token_convex version 2's partial proofs."""
    budgets = (6000, 3000, 1000, 300)
    lines = ["| Task, size | Proof | " + " | ".join(f"within B/{r:,}" for r in budgets) + " |",
             "| --- | --- | " + " | ".join("---" for _ in budgets) + " |"]
    for folder, setting in SETTINGS:
        task = f"{folder.split('-')[1].replace('max', 'max of ').replace('min', 'min of ')}, {setting}"
        p = v1.GROSS / f"all_proofs_{folder}_{setting}.json"
        if p.exists():
            rows = list(json.loads(p.read_text()).values())
            cells = []
            for r in budgets:
                shares = []
                for x in rows:
                    front = x["scores"]["frontier"]
                    B, correct = front[-1]["length"], front[-1]["certified"]
                    shares.append(share_within(front, B / r, correct))
                cells.append(f"{v1.med(shares):.2f}")
            lines.append(f"| {task} | Gross, best of all proofs | " + " | ".join(cells) + " |")
            task = ""
        q = HERE / f"measure_{folder}_{setting}.json"
        if q.exists():
            rows = list(json.loads(q.read_text()).values())
            cells = [f"{v1.med([share_within(x['points'], x['B'] / r, x['real_correct']) for x in rows]):.2f}" for r in budgets]
            lines.append(f"| {task} | token_convex v2 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
