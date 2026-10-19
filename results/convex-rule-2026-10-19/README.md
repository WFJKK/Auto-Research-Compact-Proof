# token_convex: the cubic proof's idea as a checker rule (19 October 2026)

The question: the agent's four-token proofs stayed at B/23 to B/124 while Gross et al.'s cubic proof reaches B/223 and B/705. Is the gap the agent's, or the checker's?

It was the checker's. With token_interval, a leaf where several positions vary is bounded weight by weight, and the softmax weights of the non-maximal positions can add up to far more than one; so a proof needs about one leaf per input of all but one position. `rules/token_convex` (the rule route's first rule, docs/RULE_ROUTE.md) bounds the attention-weighted sum by its exact minimum: over the box of scores (a ratio of linear functions of the exponentials, minimised at the ends of the score ranges), or over every choice of one token per position with its own score (Dinkelbach's method), pessimising over the wrong outputs first as the cubic proof does. `rules/token_convex/SOUNDNESS.md` has the argument.

## The proof

`peel.py` builds a proof that uses only token_convex. It fixes the query (the last position), then peels: with the other positions allowed 0..M, either position 0 holds M (one box: position 0 = M, the others 0..M), or position 0 is below M and position 1 holds M, and so on; when all are below M, the same with M - 1. Every box has the constant label M, so a proof has about (n - 1) V boxes per query, 376 at v16 and 852 at v24, instead of V^(n-1) inputs per query. A box the rule cannot certify is split by peeling the top token off its widest varying position. The checker decides the splits here (a recipe would compute the same bound in floats), and only the final tree is counted.

`measure.py` checks the full proof and seven partial ones (the boxes with the most inputs per leaf first, the rest skipped) on every development network, and scores them with `core.scoring`, brute force at B as the last point, exactly as the agent's proofs and Gross et al.'s are scored. `table.py` makes the table below.

## Results (median over the six development networks)

| Task, size | Proof | Full proof | Accuracy | Q | Q_grid | Cost of finishing |
| --- | --- | --- | --- | --- | --- | --- |
| max of 4, v16 | agent (token_interval) | B/23 to B/30 | 1 | 0.481 | 0.481 | 27 |
| | Gross cubic | B/223 | 0.995 | 0.460 | 0.435 | 102 |
| | Gross, best of all proofs (6 nets) | | | 0.572 | 0.563 | 102 |
| | token_convex (peeling, 6 nets) | B/217 | 1.000 | 0.538 | 0.536 | 217 |
| max of 4, v24 | agent (token_interval) | B/53 to B/76 | 1 | 0.481 | 0.452 | 56 |
| | Gross cubic | B/705 | 0.992 | 0.491 | 0.470 | 109 |
| | Gross, best of all proofs (6 nets) | | | 0.615 | 0.588 | 109 |
| | token_convex (peeling, 6 nets) | B/630 | 1.000 | 0.561 | 0.546 | 630 |
| min of 4, v16 | agent (token_interval) | B/30 to B/47 | 1 | 0.476 | 0.471 | 39 |
| | Gross cubic | B/223 | 0.994 | 0.459 | 0.434 | 93 |
| | Gross, best of all proofs (6 nets) | | | 0.570 | 0.562 | 93 |
| | token_convex (peeling, 6 nets) | B/219 | 1.000 | 0.536 | 0.536 | 219 |
| min of 4, v24 | agent (token_interval) | B/86 to B/124 | 1 | 0.495 | 0.470 | 99 |
| | Gross cubic | B/705 | 0.995 | 0.492 | 0.471 | 165 |
| | Gross, best of all proofs (6 nets) | | | 0.617 | 0.590 | 165 |
| | token_convex (peeling, 6 nets) | B/669 | 1.000 | 0.562 | 0.546 | 669 |

Accuracy is the share of the correct inputs the full proof certifies. Gross et al.'s rows: their cubic proof alone (`results/gross-cubic-2026-10-16/run_cubic.py`), and the frontier of all their proofs (brute force, cubic and every sub-cubic variant, `run_all_proofs.py`).

## Reading

- **Full certification.** With token_convex, a full proof costs about what the cubic proof costs (B/180 to B/232 at v16, B/530 to B/769 at v24, over the 24 networks) and certifies every input the network gets right, where the cubic proof certifies 99.2 to 99.5%. So cost of finishing, which rewards the best single trade-off, goes from the agent's 27 to 99 and Gross et al.'s 93 to 165 to about 220 at v16 and 630 to 670 at v24.
- **Q.** The curve of partial proofs lifts Q from the agent's 0.48 to 0.50 to 0.54 and 0.56, but Gross et al.'s best proofs still score higher (0.57 to 0.62): their sub-cubic proofs certify 54 to 71% of the inputs at about B/1,500, where these partial proofs certify about 11%. A proof here pays a fixed price per box (the query is fixed in every box) and for the tables; theirs share the attention analysis across queries.
- **The paper's size.** At v64 (spot checks on attn-max4-v64-s0, no full proof yet) the boxes with the largest token 63 are settled by level 0 at 800 to 3,500 per leaf, but boxes with a mid-range largest token (40, say) need the per-output search for many outputs and cost about 70,000 per leaf, so a full v64 proof would be much longer than the cubic proof's B/12,283. At v24, splitting the boxes where the rule is expensive saved almost nothing (`peel.py --cost-aware`). A cheaper level for all outputs at once is the next thing to try.
- **What the agent would need.** The proof is generic in structure (fix the query, peel by the largest token) and is the kind of tree the agent writes; with token_convex in its rule set and the rule's description, the next runs test whether it finds it.

## Caveats

- The proofs are oracle-guided (the checker decides where to split). The bound is computable in floats by a recipe, so this is what the rule allows, not yet what the agent achieves.
- Python-checked, like every number here; token_convex has no Lean proof yet.
- Gross et al.'s lengths are our count of their cubic proof scaled by their own instruction counter (see `results/gross-cubic-2026-10-16/README.md`); token_convex's lengths are measured by the checker.
