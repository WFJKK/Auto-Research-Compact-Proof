# Auto-Research-Compact-Proof

An agentic loop that hill-climbs towards better compact proofs of model performance, in the sense of [Gross et al., "Compact Proofs of Model Performance via Mechanistic Interpretability"](https://arxiv.org/abs/2406.11779).

Each round, an agent proposes a circuit claim and a recipe: a program that turns a trained network's weights into a proof of a lower bound on its accuracy. A sandbox runs the recipe on each network, a trusted checker verifies every proof with exact arithmetic and measures its length, and the scores feed the next round. A better explanation of the network shows up as a shorter proof of the same bound.

## Status

Specification only; no code yet. The build plan is [docs/SPEC.md](docs/SPEC.md).

The first model is the max-of-2 network from the [proof-based approach tutorial](https://github.com/LouisYRYJ/Proof_based_approach_tutorial/blob/master/proof_public.ipynb).

## Layout

The core in `core/` is model-agnostic. Each model lives in its own folder, `models/<name>/`, with its architecture in `model.py` and every number in `config.yaml`; switching models means pointing a run config at a different folder. Section 4 of the spec has the details.

## Building

Open a Claude Code session in this repository and ask it to follow `docs/SPEC.md`, starting at Step 0. `CLAUDE.md` holds the working rules.

## Running with your own API key

Once the loop exists (Step 8 of the spec), keep your Anthropic API key outside the repository and export it as `CPL_ANTHROPIC_KEY`, never as `ANTHROPIC_API_KEY`. Run data goes to a folder outside the repository, set by `runs_dir` in the run config.
