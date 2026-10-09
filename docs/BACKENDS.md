# Agent backends

The run config's `backend` picks where each round's response comes from. All four produce the same files in the round folder, so everything downstream (parsing, sandbox, checker, reports) is identical.

| Backend | Who answers | What it costs | Use it for |
| --- | --- | --- | --- |
| `fake` | files replayed in order | nothing | tests, baselines, dry runs |
| `manual` | you, pasting each prompt into claude.ai and saving the answer | nothing beyond your subscription | trying the loop by hand, a few rounds |
| `api` | the Messages API with an API key | per token (see `config/prices.yaml`) | unattended runs with full control of thinking and effort |
| `claude_code` | the official Claude Code CLI, headless, under your own Claude login | nothing per call; counts against the subscription's usage limits | unattended runs on a Pro or Max subscription |

## The `claude_code` backend

Each attempt runs the CLI as a plain model call: `claude -p --bare --tools "" --output-format json --no-session-persistence --model ... --effort ...`, with the prompt on stdin and an empty temporary folder as its working directory. With no tools, no hooks and no project settings it can only answer; it cannot read or write anything of the pipeline's. The CLI's JSON result supplies the text and the token counts. `ANTHROPIC_API_KEY` (and the pipeline's own key variable) are removed from the CLI's environment, so it can never fall back to billing an API key.

Before a run starts, the loop checks that the CLI is installed and logged in (`claude auth status`). A usage or rate limit makes the loop wait and retry with growing waits (`api_retries` in the run config; `config/max2-pilot-subscription.yaml` waits up to 30 minutes between tries, since a subscription window can take hours to reset). If it still fails, the run stops cleanly and `resume` continues later without losing a round. A login problem stops the run at once.

Each round's `meta.json` records `cost_usd: 0` (nothing was billed) and `api_equivalent_usd`, the CLI's own estimate of what the same tokens would have cost on the API, so you can still see how much of your allowance a run consumed.

**Terms.** Whether a subscription may be used for scripted, headless Claude Code runs is set by Anthropic's terms for Claude Code, which have changed during 2026. At the time of writing, signing in to the unmodified Claude Code binary with your own subscription counts as ordinary use, headless runs draw on the subscription's limits, and products built on the Agent SDK are expected to use an API key. Read the current policy for your own account before relying on this backend.

**Model names.** `agent_model` is passed to `--model`, which accepts either a full model name or an alias the CLI knows (such as `opus`). `effort` goes to `--effort` (`low`, `medium`, `high`, `xhigh`, `max`). `thinking` and `max_output_tokens` are API-only and ignored here; the CLI manages its own thinking.
