"""The agent's frontier per development network in the attention-model runs, from the runs' archives.

    python results/frontiers-2026-10-19/agent_frontiers.py <runs dir>

Raw run data stays outside the repository. This keeps what the figures and tables need: per run and network,
the frontier (the proofs no other proof of the run beats on both length and certified inputs, round 0's brute
force included) and its Q, Q_grid and cost of finishing, all from core.scoring, as in the run reports.
Writes agent_frontiers.json here.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from core.scoring import frontier, network_metrics  # noqa: E402

# run folder: the rules the agent was offered besides single and skip
RUNS = {
    "attn-max2-q-api-3": "interval",
    "attn-max2-q-api-tokens": "token_interval",
    "attn-min2-q-api-tokens": "token_interval",
    "attn-max4-q-api-tokens": "token_interval",
    "attn-min4-q-api-tokens": "token_interval",
}
KEYS = ("setting", "E", "B", "real_correct", "real_accuracy", "n_inputs")


def one_run(folder: Path, rules: str) -> dict:
    records = [json.loads(line) for line in (folder / "archive.jsonl").read_text().splitlines() if line.strip()]
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    entries: dict[str, dict] = {}
    for r in records:
        if r.get("network"):
            entries.setdefault(r["network"], {k: r.get(k) for k in KEYS})
    fr = frontier(records)
    nets = {}
    for net, e in sorted(entries.items()):
        pts = fr.get(net, [])
        m = network_metrics(pts, e, cfg.get("budget_ratio"))
        nets[net] = {
            **e,
            "frontier": [{k: p[k] for k in ("length", "certified", "n_inputs", "round", "knob")} for p in pts],
            "Q": m["Q"], "Q_grid": m["Q_grid"], "cost_of_finishing": m["cost_of_finishing"],
            "shortest_full": m["shortest_full"],
        }
    rounds = sorted({r["round"] for r in records if r.get("source") == "agent"})
    return {"rules": rules, "model": cfg.get("model"), "agent_model": cfg.get("agent_model"),
            "effort": cfg.get("effort"), "metric": cfg.get("metric"), "agent_rounds": rounds, "networks": nets}


def main(runs_dir: str) -> None:
    out = {run_id: one_run(Path(runs_dir) / run_id, rules) for run_id, rules in RUNS.items()}
    (HERE / "agent_frontiers.json").write_text(json.dumps(out, indent=1))
    for run_id, run in out.items():
        for setting in sorted({v["setting"] for v in run["networks"].values()}):
            vs = [v for v in run["networks"].values() if v["setting"] == setting]
            full = [v["B"] / v["shortest_full"] for v in vs if v["shortest_full"]]
            print(f"{run_id} {setting}: {len(vs)} networks, rounds {run['agent_rounds']}, "
                  f"Q {statistics.median(v['Q'] for v in vs):.3f}, "
                  f"Q_grid {statistics.median(v['Q_grid'] for v in vs):.3f}, "
                  f"cost of finishing {statistics.median(v['cost_of_finishing'] for v in vs):.2f}, "
                  f"full proofs B/{min(full, default=0):.1f} to B/{max(full, default=0):.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
