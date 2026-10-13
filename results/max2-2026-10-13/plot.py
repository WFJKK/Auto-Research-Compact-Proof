import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
 "Fable 5.1 high, scored on Q": {
  "v64": [(0,0.0,2696387),(1,0.3241,312250),(2,0.3692,312250),(3,0.3771,295768),(4,0.3789,295697),(5,0.387,295697),(6,0.3889,295697),(7,0.3903,295697),(8,0.3921,295697),(9,0.3926,295697),(10,0.3933,295697)],
  "v128": [(0,0.0,40407555),(1,0.3287,2446079),(2,0.3684,2446079),(3,0.3858,2112262),(4,0.3881,2112262),(5,0.3967,2112262),(6,0.3982,2112262),(7,0.4022,2112262),(8,0.4033,2112262),(9,0.4034,2112262),(10,0.4036,2112262)]},
 "Fable 5.1 high, scored on cost of finishing": {
  "v64": [(0,0.0,2696387),(1,0.2756,312260),(2,0.2756,312260),(3,0.2756,312260),(4,0.2756,312260),(5,0.2756,312260),(6,0.2756,312260),(7,0.2756,312260),(8,0.2866,312260),(9,0.324,297760),(10,0.325,294854)],
  "v128": [(0,0.0,40407555),(1,0.307,2446084),(2,0.307,2446084),(3,0.3095,2390680),(4,0.3095,2390680),(5,0.3095,2390680),(6,0.3095,2390680),(7,0.3095,2390680),(8,0.3129,2390680),(9,0.3465,2142732),(10,0.3477,2117808)]},
}
louis = {"v64": (0.276, 0.323, 312264), "v128": (0.307, 0.347, 2446083)}
B = {"v64": 2696387, "v128": 40407555}
colors = ["#ff7f0e", "#2ca02c"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
styles = [dict(color="#ff7f0e", lw=2.2, marker="o", ms=5), dict(color="#2ca02c", lw=1.4, marker="s", ms=4, ls="--")]
for c, st in enumerate(["v64", "v128"]):
    ax, bx = axes[0][c], axes[1][c]
    q1, qk, L = louis[st]
    for (name, d), sty in zip(runs.items(), styles):
        ks = [p[0] for p in d[st]]
        ax.plot(ks, [p[1] for p in d[st]], label=name, **sty)
        bx.plot(ks, [B[st] / p[2] for p in d[st]], label=name, **sty)
    ax.axhline(q1, color="gray", ls="--", lw=1, label="hand-written proof (Louis), one length")
    ax.axhline(qk, color="black", ls=":", lw=1, label="hand-written proof with a budget knob")
    bx.axhline(B[st] / L, color="gray", ls="--", lw=1, label="hand-written proof")
    ax.set_title(f"{st}: median Q over the networks"); ax.set_ylabel("Q"); ax.set_ylim(0, 0.43)
    bx.set_title(f"{st}: cost of finishing (B / shortest full-accuracy proof)"); bx.set_ylabel("times cheaper than brute force")
    bx.set_ylim(0, B[st] / L * 1.25)
    for a in (ax, bx):
        a.set_xlabel("round (0 = brute force)"); a.grid(alpha=0.3); a.set_xticks(range(0, 11))
h1, l1 = axes[0][0].get_legend_handles_labels()
fig.legend(h1, l1, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("plot.png", dpi=170)
