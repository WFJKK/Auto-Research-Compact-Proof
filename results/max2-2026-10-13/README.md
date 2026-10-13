# max2, 13 October 2026: two runs of Claude Fable 5.1

Two runs, identical except for the metric the agent is told to maximise: `config/max2-rerun-q.yaml` (Q) and `config/max2-rerun-cof.yaml` (cost of finishing). Effort high, 10 rounds each, container sandbox, cost model cost-v2. The per-run reports and held-out evaluations are in `../max2-rerun-q/` and `../max2-rerun-cof/`.

Shortest full-accuracy proof, median over six development networks (four held-out networks in brackets):

| | v64 | v128 |
| --- | --- | --- |
| Hand-written proof (Louis) | 312,264 | 2,446,083 |
| Scored on Q | 295,698, -5.3% (292,558, -6.3%) | 2,112,262, -13.6% (2,093,111, -14.4%) |
| Scored on cost of finishing | 294,854, -5.6% (292,558, -6.3%) | 2,117,809, -13.4% (2,117,356, -13.4%) |

Both runs found the same idea: the margin is small only near the diagonal a close to b, so the far region is certified through coarse row-segment approximations of the contracted matrix at several error levels, and only a band near the diagonal is checked exactly. The Q run found it in round 3, the cost run in rounds 9 and 10. Median Q: Louis 0.276 and 0.307 (0.323 and 0.347 with the budget knob of baseline 4), Q run 0.393 and 0.404, cost run 0.325 and 0.348.

`compact-proofs-max2-results.pdf` is the write-up; `q-and-cost-by-round.png` the figure (`plot.py` draws it from the numbers in the two archives).
