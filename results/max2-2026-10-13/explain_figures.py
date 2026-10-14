"""The explanatory figures of the write-up.

    python -B results/max2-2026-10-13/explain_figures.py

landscape.png        certified accuracy against length for every proof discussed (medians, v64 and v128)
q-explained.png      how Q is computed, on v64 with two proofs
cubic-one-input.png  what the cubic proof (and convexity for accuracy) does with one input, network v64 s0
gaming.png           a toy: the better recipe gets the lower Q when the other one picks its budgets
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

BLUE, ORANGE, GREEN, VIOLET, GRAY, LGRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984", "#c9c8c3"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
ref = json.loads((HERE / "reference_proofs.json").read_text())["medians"]
# shortest full-accuracy proofs of the two runs, medians over the development networks
RUNS = {"v64": {"Q": 295698, "cost": 294854}, "v128": {"Q": 2112262, "cost": 2117809}}


def kfmt(x, _=None):
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)


def landscape():
    fig, axes = plt.subplots(2, 1, figsize=(10, 10.5), facecolor=SURF)
    for ax, st in zip(axes, ("v64", "v128")):
        m = ref[st]
        style(ax)
        B = m["brute_force"]["length"]
        acc = m["cubic"]["accuracy"]
        # the cubic proof with a budget knob, as the staircase it gives
        kn = m["cubic_knob"]
        ax.step([p["length"] for p in kn], [p["accuracy"] for p in kn], where="post", color="black", ls=":", lw=1.2,
                label="cubic proof with a budget knob (5 knob values)")
        ax.plot([p["length"] for p in kn], [p["accuracy"] for p in kn], "o", color="black", ms=4)
        pts = [
            ("brute_force", "brute force", "X", GRAY, True),
            ("symmetry", "symmetry", "s", GRAY, True),
            ("cubic", "cubic proof", "o", "black", True),
            ("convexity_loss", "convexity as in the tutorial (estimate)", "v", VIOLET, False),
            ("convexity_accuracy", "convexity for accuracy (estimate)", "D", VIOLET, False),
            ("hybrid", "convexity for accuracy + exact checks elsewhere (estimate)", "D", VIOLET, True),
        ]
        for key, label, mk, col, filled in pts:
            L, a = m[key]["length"], m[key]["accuracy"]
            ax.plot([L], [a], mk, color=col, ms=9, mfc=col if filled else SURF, mec=col, mew=1.8, zorder=5, label=label)
        ax.plot([RUNS[st]["Q"]], [acc], "o", color=ORANGE, ms=13, mec=SURF, mew=1.2, zorder=6, label="band proof, run scored on Q")
        ax.plot([RUNS[st]["cost"]], [acc], "s", color=GREEN, ms=5.5, mec=SURF, mew=0.8, zorder=7, label="band proof, run scored on cost of finishing")
        # labels, staggered so the full-accuracy cluster stays readable
        notes = [
            (m["brute_force"]["length"], acc, "brute force", (8, -24), "center"),
            (m["symmetry"]["length"], acc, "symmetry", (-8, -24), "center"),
            (m["cubic"]["length"], acc, f"cubic {kfmt(m['cubic']['length'])}", (14, 14), "left"),
            (RUNS[st]["Q"], acc, f"band (both runs) {kfmt(RUNS[st]['Q'])}", (-40, 26), "right"),
            (m["hybrid"]["length"], acc, f"convexity for accuracy\n+ exact elsewhere {kfmt(m['hybrid']['length'])} (est.)", (-30, -46), "right"),
            (m["convexity_accuracy"]["length"], m["convexity_accuracy"]["accuracy"], f"convexity for accuracy\n{kfmt(m['convexity_accuracy']['length'])}, {m['convexity_accuracy']['accuracy']:.0%} (est.)", (-14, 4), "right"),
            (m["convexity_loss"]["length"], m["convexity_loss"]["accuracy"], f"convexity as in the tutorial\n{m['convexity_loss']['accuracy']:.1%} (est.)", (-14, 8), "right"),
        ]
        for x, y, text, off, ha in notes:
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=LGRAY, lw=0.8))
        ax.set_xscale("log")
        ax.set_xlim(1e4 if st == "v64" else 5e4, B * 1.6)
        ax.set_ylim(-0.03, 1.12)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("certified accuracy", color=INK)
        ax.set_title(f"{st}: medians over the six development networks", loc="left", fontsize=11, color=INK)
    axes[1].set_xlabel("proof length (operations the checker performs), log axis", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Every proof discussed: length against certified accuracy", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(HERE / "landscape.png", dpi=160, facecolor=SURF)


def q_explained():
    m = ref["v64"]
    E, B = 1088, m["brute_force"]["length"]
    C, aC = m["convexity_accuracy"]["length"], m["convexity_accuracy"]["accuracy"]
    CUB, aCub = m["cubic"]["length"], m["cubic"]["accuracy"]
    SYM = m["symmetry"]["length"]
    xs, ys = [E, C, C, CUB, CUB, B], [0, 0, aC, aC, aCub, aCub]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.4), facecolor=SURF)
    for ax, log in zip(axes, (False, True)):
        style(ax)
        ax.fill_between(xs, ys, color=BLUE, alpha=0.16, lw=0)
        ax.plot(xs, ys, color=BLUE, lw=2.2)
        ax.plot([C, CUB], [aC, aCub], "o", color=BLUE, ms=9, mec=SURF, mew=2, zorder=5)
        ax.plot([SYM, B], [aCub, aCub], "o", color=SURF, ms=9, mec=GRAY, mew=1.8, zorder=5)
        ax.annotate(f"convexity for accuracy\n{kfmt(C)} operations, {aC:.0%} (estimate)", (C, aC), xytext=(8, -36) if not log else (-10, 6),
                    textcoords="offset points", fontsize=9.5, color=INK, ha="left" if not log else "right")
        ax.annotate(f"cubic proof\n{kfmt(CUB)}, {aCub:.1%}", (CUB, aCub), xytext=(8, -34), textcoords="offset points", fontsize=9.5, color=INK)
        b = 200_000
        ax.axvline(b, color=ORANGE, lw=1.6, ls="--")
        if not log:
            ax.annotate("symmetry, brute force: also full accuracy,\nbut longer, so they never raise the staircase", (SYM, aCub),
                        xytext=(-60, -44), textcoords="offset points", fontsize=8.8, color=INK2)
            ax.annotate(f"budget b = 200k:\nyou can afford the {kfmt(C)} proof,\nnot the {kfmt(CUB)} one, so a(200k) = {aC:.0%}", (b, aC),
                        xytext=(10, 22), textcoords="offset points", fontsize=9, color=ORANGE)
            ax.set_xlim(0, B * 1.02)
            lin = (aC * (CUB - C) + aCub * (B - CUB)) / (B - E)
            ax.set_xlabel("budget b (operations), normal axis", color=INK)
            ax.set_title(f"Normal axis: shaded area / width = {lin:.2f}. Everything below B/10 is squeezed into the first tenth.",
                         fontsize=10.5, color=INK, loc="left")
        else:
            ax.annotate(f"budget b = 200k: you can afford the {kfmt(C)} proof,\nnot the {kfmt(CUB)} one, so a(200k) = {aC:.0%}", (b, aC),
                        xytext=(2.2e3, 0.74), textcoords="data", fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
            ax.set_xscale("log")
            ax.set_xlim(E / 1.15, B * 1.15)
            w1, w2, W = math.log(CUB / C), math.log(B / CUB), math.log(B / E)
            for lo, hi, h, w in ((C, CUB, aC, w1), (CUB, B, aCub, w2)):
                mid = math.sqrt(lo * hi) / (1.2 if lo == C else 1.0)
                ax.text(mid, h / 2, f"height {h:.2f}\nwidth ln({kfmt(hi)}/{kfmt(lo)})\n= {w:.2f}", ha="center", va="center", fontsize=9, color=INK)
            ax.text(math.sqrt(E * C), 0.18, f"nothing certified here\nwidth ln({kfmt(C)}/1k) = {math.log(C / E):.2f}", ha="center", fontsize=9, color=INK2)
            ax.set_xlabel("budget b (operations), log axis: every factor of 10 gets the same width", color=INK)
            q = (aC * w1 + aCub * w2) / W
            ax.set_title(f"Log axis (this is Q): ({aC:.2f} x {w1:.2f} + {aCub:.2f} x {w2:.2f}) / ln(B/E) = {aC * w1 + aCub * w2:.2f} / {W:.2f} = {q:.2f}",
                         fontsize=10.5, color=INK, loc="left")
            ax.text(E, 1.07, "  E = 1k\n  (reading the weights)", fontsize=8, color=INK2, va="center")
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("a(b): best certified accuracy\namong proofs of length at most b", color=INK, fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axvline(E, color=INK2, lw=0.8)
        ax.axvline(B, color=INK2, lw=0.8)
        ax.text(B, 1.07, "B = 2.7M  \n(brute force)  ", fontsize=8, color=INK2, va="center", ha="right")
    fig.suptitle("How Q is computed: v64, with two proofs", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "q-explained.png", dpi=160, facecolor=SURF)


def cubic_one_input():
    from core.model_folder import load_model_folder
    from core.zoo import load_weights

    w = load_weights(load_model_folder(str(ROOT / "models" / "max2")), "max2-v64-s0")
    M = np.asarray(w["unembedding.weight"], np.float64) @ np.asarray(w["linear.weight"], np.float64) @ np.asarray(w["embedding.weight"], np.float64)
    V = M.shape[0]
    a, b = 30, 40
    ca, cb = M[:, a], M[:, b]
    z = ca + cb
    others = [j for j in range(V) if j != b]
    ru = max(others, key=lambda j: z[j])
    fav_a, fav_b = max(others, key=lambda j: ca[j]), max(others, key=lambda j: cb[j])
    lead_a, lead_b = ca[b] - ca[fav_a], cb[b] - cb[fav_b]
    j = np.arange(V)
    fig, (p1, p2) = plt.subplots(2, 1, figsize=(9.5, 8.6), facecolor=SURF)
    for ax in (p1, p2):
        style(ax)
        ax.set_xlim(-1, V)
    p1.plot(j, ca, color=ORANGE, lw=2, label=f"column {a}: what token {a} adds to each output's score")
    p1.plot(j, cb, color=BLUE, lw=2, label=f"column {b}: what token {b} adds to each output's score")
    p1.plot([fav_a, b], [ca[fav_a], ca[b]], "o", color=ORANGE, ms=8, mec=SURF, mew=1.5)
    p1.plot([fav_b, b], [cb[fav_b], cb[b]], "o", color=BLUE, ms=8, mec=SURF, mew=1.5)
    p1.annotate(f"column {a} gives output {b}\n{-lead_a:.2f} less than its favourite, {fav_a}", (b, ca[b]), xytext=(6, 38), textcoords="data",
                fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    p1.annotate(f"column {b} gives output {b}\n{lead_b:.2f} more than its runner-up, {fav_b}", (b, cb[b]), xytext=(46, 44), textcoords="data",
                fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    p1.set_title("Step 1, done once per token: each token's column of M (64 numbers)", loc="left", fontsize=11, color=INK)
    p1.set_xlabel("output j", color=INK)
    p1.set_ylabel("contribution to the score of output j", color=INK)
    p1.legend(loc="upper left", fontsize=9, frameon=False)
    p1.set_ylim(-30, 66)
    p1.text(0.0, -0.30, f"Convexity for accuracy uses only the two marked gaps: {lead_a:.2f} + {lead_b:.2f} = {lead_a + lead_b:.2f} < 0, so it cannot prove ({a}, {b}).",
            transform=p1.transAxes, fontsize=10, color=INK)
    cols = [LGRAY] * V
    cols[b], cols[ru], cols[a] = BLUE, "#52514e", ORANGE
    p2.bar(j, z, color=cols, width=0.8)
    p2.annotate(f"output {b} (the right answer): {z[b]:.1f}", (b, z[b]), xytext=(44, z[b] + 8), fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    p2.annotate(f"runner-up, output {ru}: {z[ru]:.1f}\n{b} wins by {z[b] - z[ru]:.2f}", (ru, z[ru]), xytext=(21, z[ru] + 14), fontsize=9, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#52514e"))
    p2.annotate(f"output {a}: {z[a]:.1f}\n(column {b} adds only {cb[a]:.1f} to it)", (a, z[a]), xytext=(0, z[a] + 18), fontsize=9, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    p2.set_ylim(min(z) - 5, z[b] + 22)
    p2.set_title("Step 2, done for each of the 2,080 sorted inputs: add the two columns, check the right answer is highest", loc="left", fontsize=11, color=INK)
    p2.set_xlabel("output j", color=INK)
    p2.set_ylabel(f"the network's score for output j\non input ({a}, {b})", color=INK)
    p2.text(0.0, -0.27, f"The cubic proof does exactly this: 64 additions, 63 comparisons. {b} is highest, so ({a}, {b}) is proved.",
            transform=p2.transAxes, fontsize=10, color=INK)
    fig.suptitle(f"The cubic proof on one input, ({a}, {b}), network v64 s0", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(h_pad=3.2)
    fig.savefig(HERE / "cubic-one-input.png", dpi=160, facecolor=SURF)


def gaming():
    E, B = 1_000, 2_700_000
    W = math.log(B / E)
    T = math.log(10)
    good = lambda L: 0 if L <= 25e3 else 1 if L >= 250e3 else math.log(L / 25e3) / T  # noqa: E731
    bad = lambda L: 0 if L <= 30e3 else 1 if L >= 300e3 else math.log(L / 30e3) / T  # noqa: E731
    good_b = [250e3 * f for f in (0.01, 0.0316, 0.1, 0.316, 1.0)]
    bad_b = [30e3 * math.exp(T * k / 5) for k in range(1, 6)]

    def stair(curve, budgets):
        pts = [p for p in sorted((L, curve(L)) for L in budgets) if p[1] > 0]
        area = sum(a * math.log((pts[i + 1][0] if i + 1 < len(pts) else B) / L) for i, (L, a) in enumerate(pts))
        xs, ys = [E], [0]
        for L, a in pts:
            xs += [L, L]
            ys += [ys[-1], a]
        return xs + [B], ys + [ys[-1]], area / W

    xs = [E * (B / E) ** (i / 600) for i in range(601)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURF)
    style(ax)
    ax.plot(xs, [good(x) for x in xs], color=BLUE, ls="--", lw=1.4)
    ax.plot(xs, [bad(x) for x in xs], color=ORANGE, ls="--", lw=1.4)
    gx, gy, gq = stair(good, good_b)
    bx, by, bq = stair(bad, bad_b)
    ax.plot(gx, gy, color=BLUE, lw=2.4, label=f"recipe A, better at every budget, hands in proofs at fixed budgets: Q = {gq:.3f}")
    ax.plot(bx, by, color=ORANGE, lw=2.4, label=f"recipe B, worse at every budget, picks its budgets to maximise Q: Q = {bq:.3f}")
    ax.plot(good_b, [good(L) for L in good_b], "o", color=BLUE, ms=8, mec=SURF, mew=1.5, zorder=5)
    ax.plot(bad_b, [bad(L) for L in bad_b], "o", color=ORANGE, ms=8, mec=SURF, mew=1.5, zorder=5)
    ax.text(1.3e3, 0.62, "dashed: what each recipe could certify at each budget\n(A's dashed line is always to the left of B's: A is better)\n"
            "solid: the staircase Q actually measures, from the 5 proofs handed in", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(E, B * 1.1)
    ax.set_ylim(0, 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("proof length (operations), log axis", color=INK)
    ax.set_ylabel("certified accuracy", color=INK)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title("A toy example: the better recipe gets the lower Q", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gaming.png", dpi=160, facecolor=SURF)


if __name__ == "__main__":
    landscape()
    q_explained()
    cubic_one_input()
    gaming()
    print("saved")
