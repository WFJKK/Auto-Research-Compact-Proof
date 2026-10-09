# Auto-Research-Compact-Proof

An agentic loop that hill-climbs towards better compact proofs of model performance, in the sense of [Gross et al., "Compact Proofs of Model Performance via Mechanistic Interpretability"](https://arxiv.org/abs/2406.11779).

Each round, an agent proposes a circuit claim and a recipe: a program that turns a trained network's weights into a proof of a lower bound on its accuracy. A sandbox runs the recipe on each network, a trusted checker verifies every proof with exact arithmetic and measures its length, and the scores feed the next round. A better explanation of the network shows up as a shorter proof of the same bound.

## Status

Steps 0 to 8 of the spec are built and tested: the max2 model folder with its trained zoo, the exact checker, the sandbox with its hardened modes, diagnostics, the prompt builder, and the loop with fake, manual and API backends (the API one tested against a mock only). Next come reports and the final held-out evaluation (Step 9), then the first real runs with an API key (Step 10).

The first model is the max-of-2 network from the [proof-based approach tutorial](https://github.com/LouisYRYJ/Proof_based_approach_tutorial/blob/master/proof_public.ipynb).

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m core.loop run --config config/max2-round0.yaml
python -m core.loop status --run ~/auto-research-compact-proof-runs/max2-round0-fake
```

The zoo's weights are in the repository, so nothing needs training. The run replays the three max2 baselines as round 0, then five deliberately broken responses, and prints the frontier. It takes a few minutes, mostly waiting for the response that never ends to time out. Add `--diagnostics` to `status` to see where each attempt lost accuracy.

Recipes run in a sandbox. On Linux, Landlock keeps them in with no setup. On a Mac, install Docker Desktop and run `python -m core.sandbox build` once; without it, runs fall back to a weak mode and say so. See [docs/SANDBOX.md](docs/SANDBOX.md).

## Layout

The core in `core/` is model-agnostic. Each model lives in its own folder, `models/<name>/`, with its architecture in `model.py` and every number in `config.yaml`; switching models means pointing a run config at a different folder. Section 4 of the spec has the details.

## Building

Open a Claude Code session in this repository and ask it to follow `docs/SPEC.md`, starting at Step 0. `CLAUDE.md` holds the working rules.

## Running with your own API key

1. Put the key in a file outside the repository, readable only by you:

   ```bash
   mkdir -p ~/.config/auto-research-compact-proof
   printf 'CPL_ANTHROPIC_KEY=%s\n' 'sk-ant-...' > ~/.config/auto-research-compact-proof/env
   chmod 600 ~/.config/auto-research-compact-proof/env
   ```

2. Load it into the shell that starts the run. Never use `ANTHROPIC_API_KEY`: Claude Code would bill its own work to that key.

   ```bash
   set -a; . ~/.config/auto-research-compact-proof/env; set +a
   ```

3. Commit your changes (an API run starts only from a clean commit), then start the run and check on it:

   ```bash
   python -m core.loop run --config config/<run>.yaml
   python -m core.loop status --run ~/auto-research-compact-proof-runs/<run_id> --diagnostics
   ```

A run stops at `rounds_max`, after `patience` rounds without a gain, or when the API reports something waiting can't fix (a spend limit, say). Ctrl-C is safe at any point; `python -m core.loop resume --run <runs_dir>/<run_id>` continues without paying for any saved response again. Each round's `meta.json` records the tokens used and the estimated cost, priced from `config/prices.yaml`.

To try the loop without a key, `config/max2-manual.yaml` lets you play the agent: each round writes a prompt for you to paste into Claude, and waits for you to save the answer as `response.md`.
