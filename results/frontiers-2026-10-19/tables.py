"""The Q and cost-of-finishing tables of README section 4, every number read from a file in results/.

    python results/frontiers-2026-10-19/tables.py

Medians over each setting's development networks; for the max2 runs the held-out networks' medians follow in
brackets. Sources: results/max2-2026-10-13/reference_proofs.json (the hand-written max2 proofs),
results/max2-rerun-{q,cof}/final.json (the max2 runs), agent_frontiers.json here (the attention-model runs),
results/gross-cubic-2026-10-16 (Gross et al.'s proofs), results/convex-rule-2026-10-19 and
results/query-gap-2026-10-19 (the hand-built token_convex proofs). Brute force scores Q 0 and cost of finishing
1 by definition. The best value in each column is in bold.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
GROSS = RES / "gross-cubic-2026-10-16"


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


CUBIC: dict = {}
for _p in sorted(GROSS.glob("cubic_results*.json")):
    CUBIC.update(load(_p))
AGENT = load(HERE / "agent_frontiers.json")


def fq(x):
    return "" if x is None else "0" if x == 0 else f"{x:.3f}"


def fc(x):
    if x is None:
        return ""
    return "1" if x == 1 else f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 20 else f"{x:.2f}"


def table(columns: list[str], rows: list[tuple[str, list]], fmt) -> str:
    """rows: (label, cells); a cell is None (no such proof), a number, or (number, held-out number)."""
    best = []
    for j in range(len(columns)):
        vals = [c[j][0] if isinstance(c[j], tuple) else c[j] for _, c in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        best.append(max(vals) if vals else None)
    out = ["| proof | " + " | ".join(columns) + " |", "| --- |" + " --- |" * len(columns)]
    for label, cells in rows:
        txt = []
        for j, c in enumerate(cells):
            if c is None:
                txt.append("not run")
                continue
            v, held = (c if isinstance(c, tuple) else (c, None))
            s = fmt(v)
            if v == best[j] and sum(1 for _, cs in rows if cs[j] is not None) > 1:
                s = f"**{s}**"
            if held is not None:
                s += f" ({fmt(held)})"
            txt.append(s)
        out.append(f"| {label} | " + " | ".join(txt) + " |")
    return "\n".join(out)


# ---- 4.1: max of 2, linear network
def max2() -> dict[str, str]:
    """{"Q": table, "cost_of_finishing": table}"""
    ref = load(RES / "max2-2026-10-13" / "reference_proofs.json")["medians"]
    runs = {k: load(RES / k / "final.json")["settings"] for k in ("max2-rerun-q", "max2-rerun-cof")}
    st = ["v64", "v128"]
    known = [("brute force", "brute_force"), ("symmetry", "symmetry"), ("cubic proof", "cubic"),
             ("cubic proof with a budget knob", "cubic_knob")]
    agents = [("agent, run scored on Q", "max2-rerun-q"), ("agent, run scored on cost of finishing", "max2-rerun-cof")]
    out = {}
    for key, fmt in (("Q", fq), ("cost_of_finishing", fc)):
        rows = [(label, [ref[s]["scores"][k][key] for s in st]) for label, k in known]
        rows += [(label, [(runs[r][s]["development"][f"{key}_median"], runs[r][s]["held_out"][f"{key}_median"]) for s in st])
                 for label, r in agents]
        out[key] = table(st, rows, fmt)
    return out


# ---- 4.2 and 4.3: the attention models
def cubic(folder, setting, key):
    return med([v[key] for k, v in CUBIC.items() if k.startswith(f"{folder}-{setting}-")])


def gross_all(folder, setting, key):
    d = load(GROSS / f"all_proofs_{folder}_{setting}.json")
    return med([v["scores"][key] for v in d.values()]) if d else None


def measured(path, folder, setting, key):
    d = load(RES / path / f"measure_{folder}_{setting}.json")
    return med([v[key] for v in d.values()]) if d else None


def agent(rules, folder, setting, key):
    vals = [v[key] for run in AGENT.values() if run["rules"] == rules
            for n, v in run["networks"].items() if n.startswith(f"{folder}-{setting}-")]
    return med(vals) if vals else None


ATTN2 = [("attn-max2", "v64"), ("attn-max2", "v128"), ("attn-min2", "v64"), ("attn-min2", "v128")]
ATTN4 = [("attn-max4", "v16"), ("attn-max4", "v24"), ("attn-min4", "v16"), ("attn-min4", "v24"), ("attn-max4", "v64")]


def attention(settings: list[tuple[str, str]]) -> dict[str, str]:
    """{"Q": table, "cost_of_finishing": table}"""
    cols = [f"{f.split('-')[1].replace('max', 'max of ').replace('min', 'min of ')}, {s}" for f, s in settings]
    out = {}
    for key, fmt in (("Q", fq), ("cost_of_finishing", fc)):
        brute = 0.0 if key == "Q" else 1.0
        rows = [
            ("brute force", [brute for _ in settings]),
            ("Gross et al.'s cubic proof", [cubic(f, s, key) for f, s in settings]),
            ("Gross et al.'s proofs, best of all", [gross_all(f, s, key) for f, s in settings]),
            ("agent, first run (interval only)", [agent("interval", f, s, key) for f, s in settings]),
            ("agent with token_interval", [agent("token_interval", f, s, key) for f, s in settings]),
            ("token_convex version 1, hand-built", [measured("convex-rule-2026-10-19", f, s, key) for f, s in settings]),
            ("token_convex version 2, hand-built", [measured("query-gap-2026-10-19", f, s, key) for f, s in settings]),
        ]
        rows = [r for r in rows if any(c is not None for c in r[1])]
        out[key] = table(cols, rows, fmt)
    return out


def main() -> None:
    for title, tables in (("4.1, max of 2, linear network", max2()), ("4.2, max and min of 2", attention(ATTN2)),
                          ("4.3, max and min of 4", attention(ATTN4))):
        for key, name in (("Q", "Q"), ("cost_of_finishing", "Cost of finishing")):
            print(f"{name}, section {title}:\n\n{tables[key]}\n")


if __name__ == "__main__":
    main()
