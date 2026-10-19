# Gross et al.'s cubic proof on our attention zoos (16 October 2026)

The question: how does the agent's best work on the one-head transformer compare with the hand-written cubic proof of Gross et al., measured the same way?

`run_cubic.py` runs their cubic proof for Max-of-K, unchanged (`compute_extreme_softmaxed_right_attention_cubic_simple`, `compute_largest_wrong_logit_cubic`, `count_correct_sequences_cubic` from `gbmi/exp_max_of_n/verification/cubic.py`, commit `fb5b181`, MIT licence), on the 24 development networks of `attn-max2`, `attn-min2`, `attn-max4` and `attn-min4`. It feeds their code the path matrices their `verify_proof` computes from a TransformerLens model (EUPU, EVOU, PVOU, EQKE, EQKP), built from our weights. For the min tasks the vocabulary is reversed, which makes the network compute a max, so their proof applies as is. Their proof runs in float64; its certified count never exceeded a network's true count of correct inputs (the script stops if it would). Its length is counted in our cost model's units (`core/check/cost.py`) by walking the loops their code runs, and Q, Q_grid and cost of finishing come from `core.scoring`, with brute force at B as the second point, exactly as for the agent's proofs. `cubic_results.json` has every network.

## Results (median over the six development networks)

Agent: Fable 5.1, effort high, 4 rounds, scored on Q, with `token_interval` (`config/attn-<task>-q-api-tokens.yaml`). Accuracy is the share of the inputs the network gets right that the proof certifies.

| Task, size | Proof | Length | Accuracy | Q | Q_grid | Cost of finishing |
| --- | --- | --- | --- | --- | --- | --- |
| max of 2, v64 | cubic | B/6 | 0.991 | 0.219 | 0.165 | 6.0 |
| | agent | B/7.5 | 1 | **0.329** | **0.285** | **7.5** |
| max of 2, v128 | cubic | B/4 | 0.983 | 0.149 | 0.151 | 4.0 |
| | agent | B/5.8 | 1 | **0.285** | **0.263** | **5.8** |
| min of 2, v64 | cubic | B/6 | 0.992 | 0.219 | 0.165 | 6.1 |
| | agent | B/7.5 | 1 | **0.332** | **0.295** | **7.5** |
| min of 2, v128 | cubic | B/4 | 0.983 | 0.149 | 0.151 | 4.0 |
| | agent | B/5.8 | 1 | **0.269** | **0.243** | **5.8** |
| max of 4, v16 | cubic | B/223 | 0.995 | 0.460 | 0.435 | **102** |
| | agent | B/23 to B/30 | 1 | **0.481** | **0.481** | 27 |
| max of 4, v24 | cubic | B/705 | 0.992 | **0.491** | **0.470** | **109** |
| | agent | B/53 to B/76 | 1 | 0.481 | 0.452 | 56 |
| min of 4, v16 | cubic | B/223 | 0.994 | 0.459 | 0.434 | **93** |
| | agent | B/30 to B/47 | 1 | **0.476** | **0.471** | 39 |
| min of 4, v24 | cubic | B/705 | 0.995 | 0.492 | **0.471** | **165** |
| | agent | B/86 to B/124 | 1 | **0.495** | 0.470 | 99 |

### At the paper's own size (added 19 October 2026)

`attn-max4` now has the paper's setting v64 (d_vocab 64, n_ctx 4, d_model = d_head 32; ten networks, 99.97 to 100% accurate). `run_cubic.py <checkout> attn-max4:v64` writes `cubic_results_attn-max4_v64.json`. There B = 220,822,257,923 (about 2^37.7) and E = 8,320.

| Task, size | Proof | Length | Accuracy | Q | Q_grid | Cost of finishing |
| --- | --- | --- | --- | --- | --- | --- |
| max of 4, v64 | cubic | B/12,283 | 0.975 | 0.537 | 0.528 | 40 |

The paper reports a 0.953 bound on accuracy itself (ours is the share of correct inputs) and 2^47 against 2^25 FLOPs for brute force and the cubic proof; our count gives a ratio of about 2^13.6, because their FLOP estimate of brute force carries a much larger constant than our count of the checker's operations. No agent run at v64 yet: `config/attn-max4-v64-q-api-tokens.yaml`.

## All of their proofs: their best case (added 19 October 2026)

`run_all_proofs.py` runs every Max-of-K proof in their code on our development networks: brute force, the cubic proof, and all 100 sub-cubic configurations (`LargestWrongLogitQuadraticConfig`: 5 x 2 x 10 choices), through a TransformerLens model built from our weights (its logits agree with our module's within 1e-6 before anything runs). The cubic proof's length is our count (`run_cubic.py`); every other proof's length is that count times the ratio of its instruction count to the cubic proof's, both from their own counter (`gbmi.utils.instructions`). Per network, the frontier over all their proofs plus brute force at B is scored with `core.scoring`, as the agent's proofs are. `all_proofs_<folder>_<setting>.json` has every proof of every network. Medians over the six development networks:

| Task, size | Networks | Q | Q_grid | Cost of finishing | Cheapest proof certifying half or more |
| --- | --- | --- | --- | --- | --- |
| max of 4, v16 | 6 | 0.572 | 0.563 | 102 | B/1,515 (64%) |
| max of 4, v24 | 6 | 0.615 | 0.588 | 109 | B/6,380 (73%) |
| min of 4, v16 | 6 | 0.570 | 0.562 | 93 | B/1,515 (61%) |
| min of 4, v24 | 6 | 0.617 | 0.590 | 165 | B/6,380 (74%) |

Their Q is higher than the cubic proof's alone (0.46 to 0.49) because the sub-cubic proofs certify most inputs at a small fraction of the cubic proof's length; their cost of finishing is the cubic proof's, which stays their best single trade-off. The v64 setting and the two-token sizes did not finish here (background jobs stop after two hours; a v64 network takes longer than the v24 ones by far); the script resumes per network:

```text
<venv with transformer_lens<3>/bin/python results/gross-cubic-2026-10-16/run_all_proofs.py <gbmi checkout> attn-max4:v64
```

The comparison with the agent and with `token_convex` proofs is in `results/convex-rule-2026-10-19/README.md`.

## Reading

- **Two tokens:** the agent's proofs are better on every measure. With two positions the cubic proof's pessimisation costs accuracy and saves little length.
- **Four tokens:** the cubic proof is a much shorter single proof, B/223 and B/705 against the agent's full proofs at B/23 to B/124, but it certifies 99.2 to 99.5% of the correct inputs, not all of them. On Q and Q_grid the two are level (within 0.02 either way): the agent's curve of partial proofs makes up for its longer full proof. On cost of finishing, which rewards the single best trade-off, the cubic proof wins by about 2 to 4 times.
- So the agent, given `token_interval`, matches the paper's hand-written proof on the area measures but has not found its central idea: pessimising once per (query token, max token, non-max token, copies) and a convexity argument over the other tokens, instead of checking groups of inputs.

## Caveats

- The cubic proof's length is our count of its operations under our conventions (1 per multiply-add, addition or comparison, 3 per softmax entry; a fixed small constant per inner-loop iteration), not a measurement by our checker. The constants are within a factor of about 2 of any reasonable count; Q moves by about 0.03 per factor of 2 at these sizes.
- Their proof is float arithmetic and not checked by our exact checker; the agent's proofs are. Their counts were only checked against the true number of correct inputs.
- The cubic proof is written for this model and task by its authors; the agent works through a generic checker and was told nothing about the model beyond its source.
