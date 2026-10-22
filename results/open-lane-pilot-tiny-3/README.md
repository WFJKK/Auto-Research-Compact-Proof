# open-lane-pilot-tiny-3 (22 October 2026): the open lane's second pilot

The first run in which the agent produced scored Lean proofs by itself, under version 4 of the counted
language (docs/OPEN_LANE.md; the Lean project at c11b8e4). Tiny max2 network: 8 tokens, 64 inputs, 80
weights, brute force 4,323. Four agent sessions of up to 90 minutes, three submissions each, on the Mac
through the Claude Code CLI; stopped by hand during round 5 after round 4 had improved the full proof. A
first start of the same config (`open-lane-pilot-tiny-2`) was stopped after twenty minutes when the agent
searched for its pilot-1 proof and tried to read the hand port under the Lean project's `examples/`;
`read_file` was limited to the two examples of the prompt (f929cfa) and the run restarted from scratch.

What is here: `report.md` and the plots (`python -m core.report`), `config.yaml`, `versions.json` (with
the Lean service's identity the run recorded), `archive.jsonl` (every record, including round 0's
baseline replay and the seven open-lane receipts as records), and per round the agent's notes, the
response metadata, every tool call with its outcome (`tools.jsonl`, with the source of every `lean_check`
attempt) and each submission's manifest and receipt. The accepted Lean modules themselves are in the Lean
repository as `examples/open_lane/pilot2_max2_tiny`, with a test that resubmits them.

| after round | 135 | 270 | 540 | 1080 | 2161 | Q_grid | full proof |
|---|---|---|---|---|---|---|---|
| 0 (brute force) | 0 | 0 | 0 | 0 | 0 | 0.000 | 4,323 |
| 1 | 0 | 0 | 21/64 | 64/64 | 64/64 | 0.466 | 1,031 |
| 2 | 0 | 6/64 | 32/64 | 64/64 | 64/64 | 0.519 | 1,031 |
| 3 | 0 | 6/64 | 38/64 | 64/64 | 64/64 | 0.538 | 1,031 |
| 4 | 0 | 6/64 | 38/64 | 64/64 | 64/64 | 0.538 | 719 |

For comparison, the four hand-written max2 baselines at the same budgets on this network (run in the
container from `tests/max2_tiny` with the baselines of `models/max2`): brute force and symmetry fit no
budget (4,323 and 2,545); the cubic proof (contraction and interval leaves) with its budget knob gives
0, 0, 0, 51/64, 64/64, Q_grid 0.359, full proof 1,227.

How to read the numbers: the mathematics is the same throughout (the logits of `[a, b]` are two columns
of `M = U L E`, which the hand proofs also use); the Q_grid gain is mostly the budget grid rewarding a
low fixed cost and the choice of cheap inputs, and the full-proof length is the number that cannot be
helped by either: 719 against 1,227, a compression ratio of 6.0 against 3.5. The agent's notes in
`rounds/round_00*/notes.md` give its own account, including the ceilings it computed under the cost
model and what the language refused it. The discussion of what this does and does not show is in the
project's notes (open-lane-pilot-2-gaming-note).
