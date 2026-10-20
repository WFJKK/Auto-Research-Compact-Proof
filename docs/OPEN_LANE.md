# The open lane

The open lane lets the agent prove things about a network in Lean instead of writing a split-tree proof
file: any argument the kernel accepts counts, and a proof written in the counted language gets a length in
the same cost-v3 units as a proof file. The Lean side (the acceptance service, the counted language, the
scoring policy) lives in the Lean project CompactCheck
(https://github.com/WFJKK/Auto-Research-Compact-Proof-in-Lean, `docs/OPEN_LANE.md` and `docs/COUNTED.md`
there). This document is the pipeline side: how a run uses it.

## What changes in a run

With `open_lane.enabled: true` in the run config, an agent round is a session with tools instead of one
response:

- The prompt (`core/agent/prompts/open_lane_v1.md`, built by `core/open_lane/prompt.py`) gives the
  specification (Spec.lean), the protected module's interface, the counted language with its prices
  (Counted.lean, CountedOps.lean), a scored brute-force proof as the worked example, the tools and their
  budgets, the scoreboard, the receipts so far, the sources of the agent's accepted submissions from earlier
  rounds, and its latest notes. No rule descriptions.
- The tools (`core/open_lane/session.py`): `lean_check` saves a module in the session's workspace and
  compiles it against the protected module of a network and claim (which the service builds and caches);
  `run_python` runs code in the recipe sandbox with the network's weights; `search_library` searches the
  Lean project's sources and docs; `submit_candidate` hands modules to the acceptance service and returns
  the receipt. Each call is logged append-only (`tools.jsonl`); the budgets (`compile_calls`,
  `python_calls`, `submissions`) persist in `session.json`, so a session that is started again continues
  them.
- The records (`core/open_lane/records.py`) come from the receipts: `ok` when verified and scored (count
  and length; it enters the frontier like a proof file's record), `verified` when accepted without a
  length, else the receipt's status with its reason. An attempt without a submission gets `no_submission`.
  A receipt certifying more inputs than the network gets right halts the run, as the Python checker's
  invariant does.
- Round 0 still replays the baselines named in the config as proof files (`rule_set` must be written in
  the config: the pilot uses `[single, skip]`); the legacy one-response loop is the baseline condition,
  run separately (`config/open-lane-baseline-tiny.yaml`).
- The stall trigger: after `stall_rounds` agent rounds without an accepted submission, the prompt asks
  for a change of approach rather than a refinement.

Layout of an open-lane attempt:

    round_003/attempt_0/
        open_lane/session.json, tools.jsonl, workspace/, submissions/000/{manifest.json, Candidate/, receipt.json}
        mcp.json                 the CLI's MCP configuration (claude_code backend)
        response.md              the session's final text (CLAIM, WHY IT HELPS, NOTES)
        response_meta.json       tokens, turns, tools used, submissions
        results.json             the records

## Backends

- `fake`: `open_lane.fake_scripts` lists JSON scripts of tool calls (`core/open_lane/backends.py`),
  replayed in order against the real tools and service; everything else is as in a real run. The tests
  (`tests/test_open_lane.py`) run the scored brute-force example through it.
- `claude_code`: the Claude Code CLI headless, as the legacy backend, with the tools attached as an MCP
  server over stdio (`core/open_lane/mcp_server.py`, started by the CLI with the command in `mcp.json`) and
  nothing else: no built-in tools, no files, no project settings, `--max-turns open_lane.max_turns`, the
  session's wall clock `open_lane.worker_seconds`. The CLI's environment has the API keys removed. The
  session's final text is the response; what it submitted stands even if the session runs out of time.
  Not exercised with a live CLI yet: the pilot is the first run (see below).

## Running the pilot

On the machine with the Claude Code login and the Lean project:

1. Build the Lean project's targets: `lake build CompactCheck.Soundness CompactCheck.CountedOps compactcheck
   acceptcheck` (a plain `lake build` also builds the tutorial module, which imports all of Mathlib).
2. Point `open_lane.lean_project` in `config/open-lane-pilot-tiny.yaml` at it; `preflight` checks for the
   built `acceptcheck`.
3. `python -m core.loop run --config config/open-lane-pilot-tiny.yaml`, in tmux. The run resumes with
   `python -m core.loop resume --run <runs_dir>/open-lane-pilot-tiny`.
4. Then `config/open-lane-pilot-max2-v64.yaml` (one network), and the baseline condition.

The receipts record the isolation the service had: Landlock on Linux, `process (weak)` on a Mac, where
Docker is not used by the service. The pilot's gate (the ticket): one agent-produced open-lane candidate
accepted, or the archive recording transparently why none was.

## What a scored proof costs

Length = the bill of the checker on the protected weights + E (one per weight entry) + one per certificate
number. The counted operations charge what cost-v3 charges the corresponding leaf (`single`: one exact
forward pass; `interval`: the interval rule on a box). There is no `tree` charge: a scored proof's structure
is its program, bounded by the service's literal budget instead. So the open lane's brute force on the tiny
network is 4,112 against B = 4,323 for the proof-file brute force (the difference is the tree and read
charges), and the two lanes' lengths are comparable but not identical on the same argument.
