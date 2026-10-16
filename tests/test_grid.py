"""Scoring at fixed budgets (metric Q_grid): the grid, the score, the recipe's budget, the prompt and the baseline."""

import json
import re
from pathlib import Path

import pytest
import yaml

from core import loop
from core.agent.build_prompt import build_prompt, load_template, placeholders
from core.check.checker import check_proof
from core.model_folder import load_model_folder
from core.run_config import RunConfigError, merge
from core.scoring import budget_grid, budget_position, q_grid, summarize
from core.zoo import load_weights

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
BUDGET_RECIPE = '''```python
import helpers


def make_proof(weights, info, knob):
    # Brute force within the budget: one single leaf per input while the budget lasts, skip the rest.
    space = info["input_space"]
    per_input = info["B"] / (space[0] * space[1])
    n_ok = int(max(0, info["budget"] - info["E"]) // per_input)
    tree = helpers.full_tree(space)
    count = [0]

    def trim(node):
        if "leaf" in node:
            count[0] += 1
            return node if count[0] <= n_ok else {"leaf": "skip"}
        return {**node, "children": [trim(c) for c in node["children"]]}

    return helpers.proof(trim(tree), notes="budget %d knob %.3f" % (info["budget"], knob))
```'''


def _quiet(*a, **k):
    pass


def response(recipe: str) -> str:
    return (
        "CLAIM: It spends its budget on brute force.\n\nWHY IT HELPS: no.\n\nPREDICTION: accuracy grows with the budget.\n\n"
        f"KNOB: the budget.\n\nRECIPE:\n{recipe}\n\nNOTES: none.\n"
    )


def grid_config(tmp_path, model, responses, **over) -> Path:
    cfg = {
        "run_id": "grid-run",
        "model": str(model),
        "runs_dir": str(tmp_path / "runs"),
        "metric": "Q_grid",
        "budget_ratio": 2,
        "sandbox": "process",
        "fake_responses": [str(p) for p in responses],
        "rounds_max": 2,
        **over,
    }
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_the_grid_halves_from_b_down_to_e():
    grid = budget_grid(1_088, 2_696_387, 2)
    assert len(grid) == 11 and grid[0] == 2_696_387 // 2 and all(b >= 1_088 for b in grid)
    assert all(grid[i] // 2 == grid[i + 1] or grid[i] // 2 == grid[i + 1] + 1 for i in range(10))
    assert len(budget_grid(4_352, 40_407_555, 2)) == 13
    assert budget_grid(100, 150, 2) == [] and budget_grid(None, 1000, 2) == [] and budget_grid(10, 1000, 1) == []
    assert budget_position(1_088, 1_088, 2_696_387) == 0.0 and budget_position(2_696_387, 1_088, 2_696_387) == 1.0
    assert 0.5 < budget_position(2_696_387 // 16, 1_088, 2_696_387) < 0.7


def test_q_grid_reads_accuracy_only_at_the_grid():
    E, B = 1_000, 1_024_000  # 10 budgets: 512k, 256k, ..., 1k
    full = {"length": 300_000, "certified": 10, "n_inputs": 10}
    assert q_grid([full], E, B, 2) == pytest.approx(1 / 10)  # fits under 512k only
    assert q_grid([{**full, "length": 256_000}], E, B, 2) == pytest.approx(2 / 10)  # a budget itself counts
    assert q_grid([{**full, "length": 260_000}], E, B, 2) == pytest.approx(1 / 10)  # 2% shorter earns nothing yet
    half = {"length": 3_000, "certified": 5, "n_inputs": 10}
    assert q_grid([half, full], E, B, 2) == pytest.approx((1.0 * 1 + 0.5 * 7) / 10)  # 512k: the full proof; 256k to 4k: half
    assert q_grid([], E, B, 2) == 0.0


def test_config_validates_the_metric_the_ratio_and_the_template(trained_stub):
    base = {"run_id": "x", "model": str(trained_stub)}
    assert merge({**base, "metric": "Q_grid"})["prompt_template"] == "v3"
    with pytest.raises(RunConfigError, match="budget_ratio"):
        merge({**base, "budget_ratio": 1})
    with pytest.raises(RunConfigError, match="KNOB"):
        merge({**base, "metric": "Q_grid", "prompt_template": "v1"})
    merge({**base, "metric": "Q", "prompt_template": "v1"})  # the old template still serves the old metrics


def test_template_v2_is_v1_with_the_knob_paragraph_as_a_placeholder():
    p1, q1 = load_template("v1")
    p2, q2 = load_template("v2")
    assert placeholders(p2) == placeholders(p1) | {"KNOB"} and placeholders(q1) == placeholders(q2)
    assert "knob is a number between 0 and 1 that we vary" in p1 and "{KNOB}" in p2


def test_a_grid_run_gives_the_recipe_its_budget_and_scores_at_the_grid(trained_stub, tmp_path):
    folder = tmp_path / "responses"
    folder.mkdir()
    (folder / "00.md").write_text(response(BUDGET_RECIPE))
    ctx = loop.start(grid_config(tmp_path, trained_stub, [folder]), log=_quiet)
    loop.drive(ctx)
    records = [r for r in ctx.run.read_archive() if r["round"] == 1 and r["status"] == "ok"]
    assert records, "the budget recipe ran"
    for e in ctx.entries:
        mine = sorted((r for r in records if r["network"] == e["id"]), key=lambda r: r["budget"])
        grid = budget_grid(e["E"], e["B"], 2)
        assert [r["budget"] for r in mine] == sorted(grid)  # one execution per fixed budget
        for r in mine:
            assert r["knob"] == pytest.approx(budget_position(r["budget"], e["E"], e["B"]))
            assert r["record_id"].endswith(f"/b{r['budget']}")
            assert (ctx.run.round_path(1) / "attempt_0" / "proofs" / f"{e['id']}__b{r['budget']}.json.gz").exists()
        # the recipe saw the budget: brute force within it certifies more at larger budgets, never beyond the budget's share
        certified = [r["certified"] for r in mine]
        assert certified == sorted(certified) and certified[-1] > certified[0]
    s = summarize(ctx.run.read_archive(), ctx.entries, 2)
    for v in s.values():
        assert 0 < v["Q_grid_median"] <= 1 and v["Q_grid_worst"] <= v["Q_grid_median"]
        for m in v["networks"].values():
            assert set(m["grid"]) == {str(b) for b in budget_grid(m["E"], m["B"], 2)}
    # the next prompt scores and ranks by Q_grid and shows the grid scoreboard with budgets, not knob values
    p = build_prompt(ctx.cfg, ctx.folder, ctx.entries, ctx.run, 2)
    assert "Your score is Q_grid" in p.part1 and 'info["budget"]' in p.part1 and "knob is a number between 0 and 1 that we vary" not in p.part1
    assert "budget B/2 (" in p.part2 and "budget B/4 (" in p.part2 and "its own Q_grid" in p.part2
    assert re.search(r"budget B/\d+ \([\d,]+\): certified [\d.]+% \(median\), round 1 on \d of 2 networks", p.part2)
    assert "length by rule at budget B/" in p.part2
    assert "knob 0:" not in p.part2


def test_the_knob_baseline_stays_within_a_fixed_budget():
    folder = load_model_folder(str(REPO / "models" / "max2"))
    src = (REPO / "models" / "max2" / "baselines" / "04_contract_interval_budget.md").read_text()
    ns = {}
    import sys

    sys.path.insert(0, str(REPO / "core"))
    exec(re.search(r"```python\n(.*?)```", src, re.S).group(1), ns)
    manifest = json.loads((REPO / "models" / "max2" / "zoo" / "manifest.json").read_text())
    e = next(x for x in manifest["networks"] if x["id"] == "max2-v64-s0")
    w = load_weights(folder, e["id"])
    V = w["embedding.weight"].shape[1]
    certified = []
    for b in budget_grid(e["E"], e["B"], 2):  # B/2 down to B/2048
        proof = ns["make_proof"](w, {"input_space": [V, V], "budget": b, "E": e["E"], "B": e["B"]}, budget_position(b, e["E"], e["B"]))
        proof["network"] = e["id"]
        r = check_proof(proof, folder, e["id"])
        assert r["status"] == "ok" and r["length"] <= b, (b, r["length"])
        certified.append(r["certified"])
    assert certified[0] == e["real_correct"] and certified == sorted(certified, reverse=True)
    assert certified[5] > 0 and certified[-1] == 0  # B/64 still buys part of the largest group; near E nothing fits
