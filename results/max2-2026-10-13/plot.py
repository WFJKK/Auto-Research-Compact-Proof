"""Q and cost of finishing per round for the two runs, with the reference proofs as horizontal lines.

    python -B results/max2-2026-10-13/plot.py

The run numbers are the medians over the six development networks after each round, taken from the
two run archives (the raw runs stay outside the repository). The reference lines come from
reference_proofs.json (reference_proofs.py): checked lines are solid or dashed grey, estimates are
labelled as such.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
runs = {
    "Fable 5.1 high, scored on Q": {
        "v64": [(0, 0.0, 2696387), (1, 0.3241, 312250), (2, 0.3692, 312250), (3, 0.3771, 295768), (4, 0.3789, 295697), (5, 0.387, 295697), (6, 0.3889, 295697), (7, 0.3903, 295697), (8, 0.3921, 295697), (9, 0.3926, 295697), (10, 0.3933, 295697)],
        "v128": [(0, 0.0, 40407555), (1, 0.3287, 2446079), (2, 0.3684, 2446079), (3, 0.3858, 2112262), (4, 0.3881, 2112262), (5, 0.3967, 2112262), (6, 0.3982, 2112262), (7, 0.4022, 2112262), (8, 0.4033, 2112262), (9, 0.4034, 2112262), (10, 0.4036, 2112262)],
    },
    "Fable 5.1 high, scored on cost of finishing": {
        "v64": [(0, 0.0, 2696387), (1, 0.2756, 312260), (2, 0.2756, 312260), (3, 0.2756, 312260), (4, 0.2756, 312260), (5, 0.2756, 312260), (6, 0.2756, 312260), (7, 0.2756, 312260), (8, 0.2866, 312260), (9, 0.324, 297760), (10, 0.325, 294854)],
        "v128": [(0, 0.0, 40407555), (1, 0.307, 2446084), (2, 0.307, 2446084), (3, 0.3095, 2390680), (4, 0.3095, 2390680), (5, 0.3095, 2390680), (6, 0.3095, 2390680), (7, 0.3095, 2390680), (8, 0.3129, 2390680), (9, 0.3465, 2142732), (10, 0.3477, 2117808)],
    },
}
B = {"v64": 2696387, "v128": 40407555}
ref = json.loads((HERE / "reference_proofs.json").read_text())["medians"]

INK, INK2 = "#1f2328", "#57606a"
styles = [dict(color="#eb6834", lw=2.2, marker="o", ms=5), dict(color="#1baf7a", lw=1.6, marker="s", ms=4, ls="--")]
# (score key, label, line style, where the label goes: "left" or "right" end, "above" or "below" the line),
# placed where no run curve passes
q_lines = [
    ("cubic_knob", "cubic proof with a budget knob", dict(color="black", ls=":", lw=1.1), "left", "above"),
    ("cubic", "cubic proof (hand-written), one length", dict(color="gray", ls="--", lw=1.1), "right", "below"),
    ("convexity_accuracy", "convexity for accuracy alone (estimate)", dict(color="#6250d6", ls="-.", lw=1.1), "right", "above"),
    ("symmetry", "symmetry", dict(color="gray", ls="-", lw=1.0), "right", "above"),
    ("convexity_loss", "convexity as in the tutorial (estimate)", dict(color="#6250d6", ls=":", lw=1.1), "right", "above"),
]
c_lines = [
    ("hybrid", "convexity for accuracy + exact checks elsewhere (estimate)", dict(color="#6250d6", ls="--", lw=1.1), "right", "above"),
    ("cubic", "cubic proof (hand-written)", dict(color="gray", ls="--", lw=1.1), "right", "below"),
    ("symmetry", "symmetry", dict(color="gray", ls="-", lw=1.0), "right", "above"),
]


def label(axis, y, text, end, side, pad):
    x, ha = (2.2, "left") if end == "left" else (10.2, "right")
    yy, va = (y + pad, "bottom") if side == "above" else (y - pad, "top")
    axis.text(x, yy, text, ha=ha, va=va, fontsize=7.2, color=INK2)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
for c, st in enumerate(["v64", "v128"]):
    ax, bx = axes[0][c], axes[1][c]
    for (name, d), sty in zip(runs.items(), styles):
        ks = [p[0] for p in d[st]]
        ax.plot(ks, [p[1] for p in d[st]], label=name, zorder=3, **sty)
        bx.plot(ks, [B[st] / p[2] for p in d[st]], label=name, zorder=3, **sty)
    for key, text, sty, end, side in q_lines:
        y = ref[st]["scores"][key]["Q"]
        ax.axhline(y, **sty)
        label(ax, y, f"{text}: {y:.3f}", end, side, 0.005)
    top = max(ref[st]["scores"][k]["cost_of_finishing"] for k, *_ in c_lines)
    for key, text, sty, end, side in c_lines:
        y = ref[st]["scores"][key]["cost_of_finishing"]
        bx.axhline(y, **sty)
        label(bx, y, f"{text}: {y:.2f}", end, side, top * 0.012)
    ax.set_title(f"{st}: median Q over the networks", color=INK)
    ax.set_ylabel("Q")
    ax.set_ylim(0, 0.45)
    bx.set_title(f"{st}: cost of finishing (B / shortest full-accuracy proof)", color=INK)
    bx.set_ylabel("times cheaper than brute force")
    bx.set_ylim(0, top * 1.15)
    for a in (ax, bx):
        a.set_xlabel("round (0 = brute force)")
        a.grid(alpha=0.3)
        a.set_xticks(range(0, 11))
        a.set_xlim(-0.3, 10.3)
h1, l1 = axes[0][0].get_legend_handles_labels()
fig.legend(h1[:2], l1[:2], loc="lower center", ncol=2, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(HERE / "q-and-cost-by-round.png", dpi=170)
print("saved", HERE / "q-and-cost-by-round.png")
