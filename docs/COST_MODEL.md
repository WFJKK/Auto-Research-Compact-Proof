# Cost model `cost-v2`

A proof's length is the work the checker does to check it, counted in exact operations. The definition lives in `core/check/cost.py`, whose docstring is shown to the agent; this page adds the reasoning and the numbers for max2.

## What counts

Each operation on exact numbers counts 1: a multiply-add, an addition or subtraction, or a comparison.

| Category | Charged for | Cost |
| --- | --- | --- |
| `extract` | converting the weights exactly | 1 per parameter (this is E) |
| `read` | reading the proof file | 1 per number, string or boolean |
| `tree` | walking the split tree | 1 per node |
| `contract` | multiplying a×b by b×c matrices | a·b·c, in the cheapest order; biases cost one matrix-vector product each |
| `approximate` | checking an approximation | 2 per entry, plus a·r·c to multiply out low-rank factors; `row_segments` also 1 per entry and 2 per segment for the column index that locates a range's segments. A range bounded through an approximation is charged to `interval`: `dense` 2 per token per output; `low_rank` 2r per token plus 3r per output; `row_segments` 2 per merge step of the two rows' segments plus 2 per output to locate them; then 1 per output to subtract the error |
| `single` | one exact forward pass | linear on a one-hot input: output size per position; linear on a vector: in·out; bias, sum or add: its size; final comparison: outputs − 1 |
| `interval` | bounds over a set of inputs | linear on a token set: 2·out per token; linear on a box: 2·in·out; output linear on token sets: 2·outputs per token (or what the approximation's form needs); output linear on a box: 3·in·outputs |
| `symmetry` | counting the inputs a leaf certifies | 1 per accepted leaf |

## Why these choices

- **Everything the checker does is charged.** That includes reading the proof file and converting weights, so no work can hide in a large proof file or in a hint the checker trusts. The checker trusts nothing a proof claims. For this charge to be fair, every operation must be on a small number: proof numbers are required to be dyadic and bounded (see `docs/RULES.md`), so a proof cannot make one operation secretly expensive. A proof that is still slow to check, for any reason, is cut off by a per-proof time limit (`check_timeout_s`) and rejected.
- **Searching is free.** A recipe may spend any amount of floating-point work finding a proof. Only checking it counts, as in Gross et al.
- **One-hot inputs are lookups.** A linear map applied to a one-hot token costs its output size, the cost of reading one column, not a full matrix-vector product. Brute force is charged the same way, so this choice does not favour any proof.
- **Bounds cost double.** An interval operation works on a centre and a radius.

## Change from `cost-v1`

`cost-v2` changes the approximate rule's charges and nothing else. Bounding a token range through a `row_segments` approximation walks the segments of two rows together, and each step takes a difference and a comparison, exactly what a dense range does per token. `cost-v1` charged such a step 1 instead of 2. In the first max2 runs the agent noticed (its round-5 claim: "one step per segment touched instead of the dense two steps per token") and wrote the contracted matrix as an "approximation" with one segment per entry and error 0: the matrix itself, checked by the same work at half the charge. That made full proofs look 1.5 times shorter than the hand-written one without any new idea. Under `cost-v2` the same proof is longer than the hand-written one, and segments pay only where they are long, which is the actual saving the rule was meant to reward.

The audit that followed found three smaller places where the checker did work the approximate rule did not charge, all of them favouring approximations over the exact check: locating a row's segments in a range (`cost-v1` used a binary search it did not count; now a column index, built when the approximation is read and charged, makes it two lookups, charged), subtracting the error from every output, and the differences of the low-rank factors. None of them could be exploited for a gain, but they are charged now so that the rule costs exactly what it does. E and B do not change, since brute force uses no approximation; results measured under `cost-v1` are re-measured from their saved proofs with `python -m core.remeasure`.

## E and B for max2

E is the extract cost, the least any proof pays. B is the length of the brute-force proof, which checks every input with `single` and nothing else. `core/check/costs.py` computes both without running brute force, and a test checks that B equals the length the checker reports for the brute-force proof. That holds on all 20 max2 networks.

| Size setting | Inputs | E | B | B/E |
| --- | --- | --- | --- | --- |
| v64 (d_vocab 64, d_model 8) | 4,096 | 1,088 | 2,696,387 | 2,478 |
| v128 (d_vocab 128, d_model 16) | 16,384 | 4,352 | 40,407,555 | 9,285 |

Baselines on `max2-v128-s0` (certified accuracy 16383/16384 for each):

| Proof | Length | B / length |
| --- | --- | --- |
| Brute force | 40,407,555 | 1.0 |
| Symmetry | 20,372,796 | 2.0 |
| Contraction with interval leaves by larger token | 2,446,083 | 16.5 |
