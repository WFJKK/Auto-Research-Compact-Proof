"""Pareto frontiers on the attention models, per task and size: the agent's runs, Gross et al.'s proofs (their
cubic proof alone, and the frontier of all their proofs), the hand-built token_convex proofs, and brute force.

    python results/frontiers-2026-10-19/plot.py

Writes frontiers_attn2.png (max and min of 2, README section 4.2) and frontiers_attn4.png (max and min of 4,
section 4.3). One panel per task and size, one colour per kind of proof, one line per development network.
Sources: agent_frontiers.json here (from the run archives, agent_frontiers.py), cubic_results*.json and
all_proofs_*.json in results/gross-cubic-2026-10-16, measure_*.json in results/convex-rule-2026-10-19
(token_convex version 1) and results/query-gap-2026-10-19 (version 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent
GROSS = RES / "gross-cubic-2026-10-16"
V1 = RES / "convex-rule-2026-10-19"
V2 = RES / "query-gap-2026-10-19"


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


CUBIC: dict = {}
for _p in sorted(GROSS.glob("cubic_results*.json")):
    CUBIC.update(load(_p))
AGENT = load(HERE / "agent_frontiers.json")

# kind of proof: (label, colour, line style)
STYLE = {
    "agent_interval": ("agent, first run (interval only)", "0.55", "-"),
    "agent": ("agent with token_interval", "C3", "-"),
    "gross_all": ("Gross et al., all their proofs", "C0", "-"),
    "v1": ("token_convex v1, hand-built", "C2", "--"),
    "v2": ("token_convex v2, hand-built", "C1", "-"),
}


def staircase(points: list[dict], B: float):
    """The frontier as a step line from its shortest proof to B."""
    xs, ys, best = [], [], -1
    for p in sorted(points, key=lambda p: (p["length"], -p["certified"])):
        if p["certified"] > best:
            xs.append(p["length"])
            ys.append(p["certified"] / p["n_inputs"])
            best = p["certified"]
    if not xs:
        return [], []
    return xs + [B], ys + [ys[-1]]


def agent_runs(prefix: str):
    """(kind, frontier per network) for the agent runs on this task and size."""
    for run_id, run in AGENT.items():
        nets = {n: v for n, v in run["networks"].items() if n.startswith(prefix)}
        if nets:
            yield ("agent_interval" if run.get("rules") == "interval" else "agent"), nets


def panel(ax, folder: str, setting: str) -> None:
    prefix = f"{folder}-{setting}-"
    nets = sorted(k for k in CUBIC if k.startswith(prefix))
    E, B = CUBIC[nets[0]]["E"], CUBIC[nets[0]]["B"]
    seen = set()

    def line(kind, xs, ys):
        label, colour, ls = STYLE[kind]
        ax.step(xs, ys, where="post", color=colour, ls=ls, lw=1.1, alpha=0.75,
                label=None if kind in seen else label)
        seen.add(kind)

    for kind, frontiers in agent_runs(prefix):
        for n, v in frontiers.items():
            line(kind, *staircase(v["frontier"], B))
    for n, v in load(GROSS / f"all_proofs_{folder}_{setting}.json").items():
        line("gross_all", *staircase(v["scores"]["frontier"], B))
    for kind, path in (("v1", V1), ("v2", V2)):
        for n, v in load(path / f"measure_{folder}_{setting}.json").items():
            line(kind, *staircase(v["points"], B))
    xs = [CUBIC[n]["length"] for n in nets]
    ys = [CUBIC[n]["certified"] / CUBIC[n]["n_inputs"] for n in nets]
    ax.scatter(xs, ys, marker="x", s=45, color="C0", zorder=5, label="Gross et al., cubic proof")
    ax.scatter([B] * len(nets), [CUBIC[n]["real_correct"] / CUBIC[n]["n_inputs"] for n in nets], marker="o", s=22,
               color="k", zorder=5, label="brute force")
    ax.axvline(E, ls="--", color="k", lw=0.8)
    ax.axvline(B, ls="--", color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.04)
    task = folder.split("-")[1].replace("max", "max of ").replace("min", "min of ")
    ax.set_title(f"{task}, {setting}")
    ax.set_xlabel("proof length (operations); dashed: E and B")
    ax.set_ylabel("certified accuracy")


def figure(layout: list[list[tuple[str, str] | None]], title: str, out: Path) -> None:
    rows, cols = len(layout), len(layout[0])
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), squeeze=False)
    handles = {}
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            ax = axes[i][j]
            if cell is None:
                ax.axis("off")
                continue
            panel(ax, *cell)
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
    order = [STYLE[k][0] for k in STYLE] + ["Gross et al., cubic proof", "brute force"]
    labs = [lab for lab in order if lab in handles]
    fig.legend([handles[lab] for lab in labs], labs, loc="lower center", ncol=min(4, len(labs)), fontsize=9,
               frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07 if len(labs) > 4 else 0.04, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved", out)


def main() -> None:
    figure([[("attn-max2", "v64"), ("attn-max2", "v128")], [("attn-min2", "v64"), ("attn-min2", "v128")]],
           "Max and min of 2, attention-only transformer: frontier per development network",
           HERE / "frontiers_attn2.png")
    figure([[("attn-max4", "v16"), ("attn-max4", "v24"), ("attn-max4", "v64")],
            [("attn-min4", "v16"), ("attn-min4", "v24"), None]],
           "Max and min of 4, attention-only transformer: frontier per development network",
           HERE / "frontiers_attn4.png")


if __name__ == "__main__":
    main()
