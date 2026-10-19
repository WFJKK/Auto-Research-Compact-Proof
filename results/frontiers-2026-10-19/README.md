# Q, cost of finishing and Pareto frontiers against the known proofs (19 October 2026)

The tables and frontier figures of the main README's section 4, one set per network: Q, cost of finishing and the Pareto frontier of the agent's proofs, next to the known proofs on the same networks.

- `tables.py` prints the Q and cost-of-finishing tables of sections 4.1 to 4.3. Every number is read from a file in `results/`: the hand-written max2 proofs (`max2-2026-10-13/reference_proofs.json`), the max2 runs (`max2-rerun-q`, `max2-rerun-cof`), Gross et al.'s proofs (`gross-cubic-2026-10-16`), the hand-built `token_convex` proofs (`convex-rule-2026-10-19`, `query-gap-2026-10-19`) and `agent_frontiers.json` here.
- `plot.py` draws `frontiers_attn2.png` (section 4.2) and `frontiers_attn4.png` (section 4.3): one panel per task and size, one line per development network. Section 4.1's figure is `max2-2026-10-13/frontiers.png`.
- `agent_frontiers.py <runs dir>` reads the archives of the five attention-model runs and writes `agent_frontiers.json`: per run and network, the frontier and its Q, Q_grid and cost of finishing, computed by `core.scoring` as in the run reports. Raw run data stays outside the repository.

The runs: `attn-max2-q-api-3` (the first run, with `interval` only) and `attn-{max2,min2,max4,min4}-q-api-tokens` (with `token_interval`). All used Fable 5.1 at effort high, 4 rounds, scored on Q. Their medians recomputed here agree with those in `results/gross-cubic-2026-10-16/README.md` to within 0.001. The small differences (0.284 here against 0.285 there, for instance) come from rounding the reports' four-decimal values a second time.

Not here: Gross et al.'s sub-cubic proofs at two tokens. Tested on `attn-max2-v64-s0`, about 81 of their 100 variants had started after 15 minutes. So the 24 two-token networks would take many hours on this machine, with v128 slower than v64.
