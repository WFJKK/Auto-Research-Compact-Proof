"""One table: the agent, Gross et al.'s cubic proof, the best of all their proofs, token_convex with a fixed query
(version 1, results/convex-rule-2026-10-19) and with leaves where the query varies (version 2, here).

    python results/query-gap-2026-10-19/table.py

Medians over each setting's development networks. Every row but the last two comes from
results/convex-rule-2026-10-19/table.py, which this imports.
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


if __name__ == "__main__":
    main()
