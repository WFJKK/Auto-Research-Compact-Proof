# The hand reference proofs on the tiny attention networks (23 October 2026)

The question: what does an open-lane pilot on a tiny attention model have to beat? The tiny models (`tests/attn_{max2,min2,max4,min4}_tiny`) have only a brute-force baseline, and the strong hand proofs were measured on the zoo's larger networks only (`results/gross-cubic-2026-10-16`, `results/convex-rule-2026-10-19`). `run.py` measures them on each tiny model's development network, scored exactly as a run scores the agent's proofs: Q, Q_exact, Q_grid and cost of finishing from `core.scoring`, with brute force at B as the last point and the single rule's exact charge per input as the fallback cost.

- Brute force: the legacy lane's proof file (B) and the open lane's (single charge times inputs, plus E; no tree or read charge).
- The token_convex peeling proof (`results/convex-rule-2026-10-19/peel.py`): the legacy lane's best attention proof, checked exactly by the Python checker; full, and partial versions that keep the boxes with the most inputs per leaf.
- Gross et al.'s cubic proof (`results/gross-cubic-2026-10-16/run_cubic.py`, their code unchanged): float64, not checked by our checker, its length counted in our units by walking their loops.

## Results

| Network | B | open-lane brute force | proof | full proof | B / full | Q_exact | Q_grid | cost of finishing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_max2_tiny-v8d4-s1 (64 inputs, E 136) | 12,571 | 12,360 | peeling | 7,384 | 1.7 | 0.250 | 0.193 | 1.70 |
| | | | cubic | 12,436 | 1.0 | 0.002 | 0.000 | 1.01 |
| attn_min2_tiny-v8d4-s1 (64 inputs, E 136) | 12,571 | 12,360 | peeling | 7,156 | 1.8 | 0.256 | 0.193 | 1.76 |
| | | | cubic | 12,436 | 1.0 | 0.002 | 0.000 | 1.01 |
| attn_max4_tiny-v4d4-s0 (256 inputs, E 112) | 74,523 | 73,584 | peeling | 7,288 | 10.2 | 0.457 | 0.421 | 10.23 |
| | | | cubic | 5,486 | 13.6 | 0.401 | 0.333 | 13.58 |
| attn_min4_tiny-v4d4-s0 (256 inputs, E 112) | 74,523 | 73,584 | peeling | 7,596 | 9.8 | 0.458 | 0.421 | 9.81 |
| | | | cubic | 5,486 | 13.6 | 0.401 | 0.333 | 13.58 |

Every proof certifies every input (64 of 64, 256 of 256). The single rule charges 191 per input on the two-position networks and 287 on the four-position ones.

## Reading

- On the two-position networks the cubic proof saves almost nothing (12,436 against B 12,571): with 8 tokens its work (the path matrices, the attention and logit loops) costs about as much as checking all 64 inputs. The peeling proof is the bar there: 7,384 and 7,156, Q_exact about 0.25.
- On the four-position networks both compress well. The cubic proof has the shortest full proof (5,486, B/13.6) but is a single point, so its Q_exact is lower; the peeling proof traces a curve (64 inputs certified for 1,824) and has the higher Q_exact (0.457). The cubic proof's count is computed in float64 and not checked by our checker; the peeling proof's is exact.
- So an open-lane proof beats the hand proofs on these networks if its full proof is under about 7,200 (two positions) or 5,500 (four positions), and its Q_exact is above about 0.26 or 0.46. The open lane's own brute force is 12,360 and 73,584; it has no tree or read charge, so the lanes' lengths are comparable but not identical for the same argument.
