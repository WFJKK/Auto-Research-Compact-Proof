# Leaves where the query varies: token_convex version 2 (19 October 2026)

The question: which known idea gives the highest Q, and can the checker express it? Of the proofs measured on our networks, Gross et al.'s sub-cubic proofs score highest (Q 0.57 at v16 and 0.62 at v24, `results/gross-cubic-2026-10-16/README.md`). They bound the attention for every query at once, through a low-rank approximation of the query-key product and the gaps between the largest token's score and the others'. token_convex version 1 could not express that: it needed the query fixed in every leaf, so a proof paid a softmax bound per query.

## The rule

Version 2 of `rules/token_convex` adds leaves where the query position varies (the rule route's second rule; three passes through it, `docs/RULE_ROUTE.md`). Three things are kept exactly that were boxed before:

- the query vector, per token of the query position;
- the attention scores, per pair of tokens: position p's score for each token at p and each query token, dot products of exact rows kept for the whole proof;
- the softmax of such scores.

A softmax depends only on score differences. So against a reference position r (one with the fewest tokens, typically the one holding the largest token), each position's weight relative to r's lies in a box over the whole leaf: for each query token, the lowest and highest entry of its column of the score table over the position's tokens, against r's. The decision gets two levels before version 1's:

- **Level A** bounds the attention-weighted average over that box for every output at once, with version 1's exact box minimum;
- **level B** does the same for each output;
- outputs still unsettled then go **query token by query token**, where the scores are exact per token and version 1's four levels apply.

`rules/token_convex/SOUNDNESS.md` (sections 6 to 9) has the argument, and its Lemma C is the relative box.

This is Gross et al.'s sub-cubic argument with the exact score table in place of the low-rank approximation. At these sizes the table is cheap: (n - 1) V^2 dot products, about 11,000 operations at v16 and 39,000 at v24, against B = 65 million and 684 million. It is never looser than an approximation.

Three more changes:

- A fixed query's scores are read from the same table.
- The smallest difference over the other outputs is read from each row's two largest entries.
- With at most one varying position, the softmax structure is tried first, and the leaf is evaluated again as token_interval does when that does not certify it (version 2.1).

## The proof

`sdecomp.py` builds a proof from leaves "the positions in S hold M, every other position (the query included) is below M". There is one box per largest token M and nonempty set S, and these boxes partition the inputs whose largest token is M. The tree reaches them by splitting each position in turn at M, giving 2^n - 1 boxes per M: about 15 V at n = 4, against about 3 V^2 / 2 for version 1's proof, which fixed the query.

A box the rule does not certify is refined by peeling the top token off its widest varying position. With `--depth 2` (`--depth 1` at v64), a box whose leaf costs more than 50 V is also split, where its two halves certify for less.

`measure.py` checks the full proof and partial ones on every development network: the boxes with the most inputs per unit of their own charges first, the rest skipped. It scores them with `core.scoring`, as every other proof here. `table.py` makes the tables below.

## Results (median over the six development networks)

| Task, size | Proof | Full proof | Accuracy | Q | Q_grid | Cost of finishing |
| --- | --- | --- | --- | --- | --- | --- |
| max of 4, v16 | agent (token_interval) | B/23 to B/30 | 1 | 0.481 | 0.481 | 27 |
| | Gross cubic | B/223 | 0.995 | 0.460 | 0.435 | 102 |
| | Gross, best of all proofs (6 nets) | | | 0.572 | 0.563 | 102 |
| | token_convex v1, query fixed (6 nets) | B/217 | 1.000 | 0.538 | 0.536 | 217 |
| | token_convex v2, query varies (6 nets) | B/460 | 1.000 | 0.610 | 0.615 | 460 |
| max of 4, v24 | agent (token_interval) | B/53 to B/76 | 1 | 0.481 | 0.452 | 56 |
| | Gross cubic | B/705 | 0.992 | 0.491 | 0.470 | 109 |
| | Gross, best of all proofs (6 nets) | | | 0.615 | 0.588 | 109 |
| | token_convex v1, query fixed (6 nets) | B/630 | 1.000 | 0.561 | 0.546 | 630 |
| | token_convex v2, query varies (6 nets) | B/1,766 | 1.000 | 0.627 | 0.607 | 1,766 |
| min of 4, v16 | agent (token_interval) | B/30 to B/47 | 1 | 0.476 | 0.471 | 39 |
| | Gross cubic | B/223 | 0.994 | 0.459 | 0.434 | 93 |
| | Gross, best of all proofs (6 nets) | | | 0.570 | 0.562 | 93 |
| | token_convex v1, query fixed (6 nets) | B/219 | 1.000 | 0.536 | 0.536 | 219 |
| | token_convex v2, query varies (6 nets) | B/497 | 1.000 | 0.610 | 0.610 | 497 |
| min of 4, v24 | agent (token_interval) | B/86 to B/124 | 1 | 0.495 | 0.470 | 99 |
| | Gross cubic | B/705 | 0.995 | 0.492 | 0.471 | 165 |
| | Gross, best of all proofs (6 nets) | | | 0.617 | 0.590 | 165 |
| | token_convex v1, query fixed (6 nets) | B/669 | 1.000 | 0.562 | 0.546 | 669 |
| | token_convex v2, query varies (6 nets) | B/1,818 | 1.000 | 0.630 | 0.609 | 1,818 |
| max of 4, v64 | agent | no run yet | | | | |
| | Gross cubic | B/12,283 | 0.975 | 0.537 | 0.528 | 40 |
| | token_convex v2, query varies (6 nets) | B/14,836 | 1.000 | 0.636 | 0.635 | 14,836 |

Accuracy is the share of the correct inputs the full proof certifies. Gross et al.'s rows: their cubic proof alone, and the frontier of all their proofs (brute force, cubic and every sub-cubic variant; not run at v64 here, `results/gross-cubic-2026-10-16/README.md`).

The share of the correct inputs certified within a length budget, the frontier of all of Gross et al.'s proofs against version 2's partial proofs:

| Task, size | Proof | within B/6,000 | within B/3,000 | within B/1,000 | within B/300 |
| --- | --- | --- | --- | --- | --- |
| max of 4, v16 | Gross, best of all proofs | 0.00 | 0.00 | 0.68 | 0.68 |
|  | token_convex v2 | 0.00 | 0.05 | 0.80 | 1.00 |
| max of 4, v24 | Gross, best of all proofs | 0.74 | 0.74 | 0.74 | 0.99 |
|  | token_convex v2 | 0.12 | 0.81 | 1.00 | 1.00 |
| min of 4, v16 | Gross, best of all proofs | 0.00 | 0.00 | 0.68 | 0.68 |
|  | token_convex v2 | 0.00 | 0.05 | 0.80 | 1.00 |
| min of 4, v24 | Gross, best of all proofs | 0.75 | 0.75 | 0.75 | 1.00 |
|  | token_convex v2 | 0.12 | 0.81 | 1.00 | 1.00 |

## Reading

- **Q.** Version 2 lifts Q from version 1's 0.54 (v16) and 0.56 (v24) to 0.61 and 0.63, above the best of all of Gross et al.'s proofs (0.57 and 0.62). At the paper's size, v64, it reaches 0.64; their cubic proof scores 0.54 there (their sub-cubic proofs were not run at v64 here).
- **Full proofs.** Every input the network gets right is certified at B/460 to B/497 (v16), B/1,766 to B/1,818 (v24) and B/14,836 (v64). That is 2.1 to 2.8 times shorter than version 1's proofs, and 2.1 to 2.6 times shorter than Gross et al.'s cubic proof at v16 and v24 (which certifies 99.2 to 99.5%). At v64 it is shorter than their cubic proof (B/12,283, certifying 97.5%). Cost of finishing goes from their 93 to 165 to 460 to 1,818 at v16 and v24, and from their 40 to 14,836 at v64.
- **Where their proofs are still ahead: the shortest proofs at v24.** Within B/6,000 their sub-cubic proofs certify 74% of the correct inputs and version 2 certifies 12%. Version 2's floor is the shared tables. Every row is computed per position: the value path, the keys and the score table. These tables make up 82 to 86% of the length of the first partial proof that certifies half the inputs (v16 and v24). Their proofs compute the token part and the position part of each path separately (EVOU and PVOU).
- **The next idea.** Keep each row's token part and position part apart through linear maps, since (E[t] + P[p]) W = E[t] W + P[p] W. The tables would then shrink by about 2.4 times at v24. Rescoring the measured partial proofs with their tables cut that much (leaf charges unchanged) gives Q of about 0.66 at v16 and 0.68 at v24. That is an estimate, not a measurement.
- **At v64.** Half of the correct inputs are certified at B/72,000 and 90% at B/24,000. Building and checking a proof, with the checker as the oracle, takes about two and a half minutes per network on one core.

## Caveats

- **Oracle-guided.** The checker decides where to split. The bounds are computable in floats by a recipe, so this shows what the rule allows, not yet what the agent achieves.
- **Python-checked,** like every number here; token_convex has no Lean proof yet.
- **Different length counts.** Gross et al.'s lengths are our count of their cubic proof, scaled by their own instruction counter (`results/gross-cubic-2026-10-16/README.md`). token_convex's lengths are measured by the checker.
- **No sub-cubic comparison at v64.** Their sub-cubic proofs were not run at v64 here, so at the paper's size the comparison is with their cubic proof only.
