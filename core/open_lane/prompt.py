"""The open-lane prompt: the specification, the cost language with its prices, the tools, the scoreboard,
the receipts so far, the agent's accepted lemmas from earlier rounds and its latest notes. No rule
descriptions: the agent is told what a proof is and what it costs, not which argument shapes to use.

Template: core/agent/prompts/open_lane_v1.md. Part 1 is the same every round; part 2 is rebuilt from the
archive. The Lean sources shown (Spec.lean, Counted.lean, CountedOps.lean, the scored example) are read
from the Lean project the run config names, so the prompt always matches the service that will check
the proof.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..agent.build_prompt import CHARS_PER_TOKEN, Prompt, PromptError, fill, forbidden, grid_text, load_template, metric_text
from ..archive import RunDir
from ..model_folder import ModelFolder
from ..scoring import summarize
from ..util import REPO_ROOT
from .session import Session

MAX_LEMMA_CHARS = 12000
MAX_NOTES_CHARS = 6000
MAX_REASON = 300
STALL_TEXT = (
    "No submission of the last {n} rounds was accepted. Change the approach rather than refining the last one: "
    "a different account of the computation, a smaller claim that you can prove completely, or a different "
    "shape of checker. Say in WHY IT HELPS what you are changing and why."
)
AIM_TEXT = (
    "Raise the score. It stands at: {scores}. Before writing, weigh several approaches (a better account of "
    "how the network computes, a cheaper checker for the same inputs, covering inputs still uncertified) and "
    "pursue the one you expect to raise it most."
)


def lean_project(cfg: dict) -> Path:
    return Path(cfg["open_lane"]["lean_project"]).expanduser().resolve()


def _read(project: Path, rel: str) -> str:
    p = project / rel
    if not p.is_file():
        raise PromptError(f"the Lean project has no {rel}")
    return p.read_text().rstrip()


def lean_block(text: str) -> str:
    return "```lean\n" + text.rstrip() + "\n```"


def protected_interface() -> str:
    return lean_block(
        "namespace Protected.N_<network>_c<claimed>   -- for example Protected.N_max2_v64_s0_c4095\n"
        "def graph : Graph := ...            -- the network's graph, operation by operation\n"
        "def weights : Weights ℚ := ...      -- every weight, an exact dyadic rational\n"
        "def task : Task := ...              -- the task: input space and label function\n"
        "def cert : Cert := #[...]           -- the certificate of a scored submission (empty otherwise)\n"
        "def claimedCount : Nat := <claimed>\n"
        "def Target : Prop := claimedCount ≤ numCorrect graph weights task\n"
        "end Protected.N_<network>_c<claimed>"
    )


def tools_text() -> str:
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in Session.TOOLS.items())


def networks_text(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        real = e.get("real_correct")
        acc = f"{real:,} of {e['n_inputs']:,} ({100 * real / e['n_inputs']:.3f}%)" if real is not None else "unknown"
        lines.append(f"- {e['id']} ({e['setting']}): correct {acc}; E {e['E']:,}; B {e['B']:,}")
    return "\n".join(lines)


def static_values(cfg: dict, folder: ModelFolder, entries: list[dict]) -> dict:
    project = lean_project(cfg)
    settings = list(dict.fromkeys(e["setting"] for e in entries))
    size_settings = "; ".join(
        f"{s}: " + ", ".join(f"{k}={v}" for k, v in folder.sizes(s).items()) + f" ({folder.n_inputs(s):,} inputs)" for s in settings
    )
    ol = cfg["open_lane"]
    budgets = (f"{ol['compile_calls']} lean_check calls, {ol['python_calls']} run_python calls, "
               f"{ol['submissions']} submissions, {ol['worker_seconds']:,} s of wall clock")
    return {
        "MODEL_SOURCE": "\n```python\n" + folder.source().rstrip() + "\n```",
        "TASK_DESCRIPTION": folder.task.DESCRIPTION,
        "SIZE_SETTINGS": size_settings,
        "NETWORKS": networks_text(entries),
        "PROTECTED_INTERFACE": protected_interface(),
        "SPEC": lean_block(_read(project, "CompactCheck/Spec.lean")),
        "COUNTED": lean_block(_read(project, "CompactCheck/Counted.lean")),
        "COUNTED_OPS": lean_block(_read(project, "CompactCheck/CountedOps.lean")),
        "EXAMPLE": lean_block(_read(project, "examples/open_lane/max2_tiny_scored/Candidate/Max2TinyScored.lean")),
        "TOOLS": tools_text(),
        "BUDGETS": budgets,
        "METRIC": metric_text(cfg, folder, entries) + (
            f"\nThe budget grid, per size setting: {grid_text(entries, cfg['budget_ratio'])}." if cfg["metric"] == "Q_grid" else ""),
    }


# part 2 ------------------------------------------------------------------------------------
def scoreboard(records: list[dict], entries: list[dict], budget_ratio) -> str:
    lines = []
    for setting, s in summarize(records, entries, budget_ratio).items():
        lines.append(f"{setting}: Q median {s['Q_median']:.4f}, worst {s['Q_worst']:.4f}; cost of finishing median {s['cost_of_finishing_median']:.2f}")
        for nid, m in s["networks"].items():
            full = f"{m['shortest_full']:,} (B/{m['B'] / m['shortest_full']:.1f})" if m["shortest_full"] else "none"
            lines.append(f"  {nid}: Q {m['Q']:.4f}; best certified accuracy {100 * m['best_accuracy']:.3f}% of "
                         f"{100 * (m['real_accuracy'] or 0):.3f}%; shortest full-accuracy proof {full}; {m['points']} frontier proofs")
    return "\n".join(lines) if lines else "nothing checked yet"


def receipts_text(records: list[dict]) -> str:
    lines = []
    for r in records:
        if r.get("lane") != "open" or r.get("status") == "no_submission":
            continue
        head = f"- round {r['round']} submission {r.get('submission')}: {r.get('network')}, claimed {r.get('claimed_count')}: {r['status']}"
        if r["status"] == "ok":
            lines.append(f"{head}, certified {r['certified']}, length {r['length']:,} (bill {r['length_by_rule']['bill']:,})")
        elif r["status"] == "lemma":
            lines.append(f"{head} (auxiliary lemma accepted): {(r.get('statement') or '')[:MAX_REASON]}")
        elif r["status"] == "verified":
            lines.append(f"{head}, certified {r['certified']}; {(r.get('error') or '')[:MAX_REASON]}")
        else:
            lines.append(f"{head}: {(r.get('error') or '')[:MAX_REASON]}")
    return "\n".join(lines) if lines else "none yet"


def accepted_sources(run: RunDir, records: list[dict], n: int, max_chars: int) -> str:
    accepted = [r for r in records if r.get("lane") == "open" and r["status"] in ("ok", "verified", "lemma")]
    accepted.sort(key=lambda r: (r["round"], r["attempt"], r.get("submission") or 0), reverse=True)
    parts, used = [], 0
    for r in accepted[:n]:
        d = run.round_path(r["round"]) / f"attempt_{r['attempt']}" / "open_lane" / "submissions" / f"{r.get('submission', 0):03d}"
        for p in sorted(d.glob("Candidate/**/*.lean")):
            text = p.read_text()
            if used + len(text) > max_chars:
                text = text[: max(0, max_chars - used)] + "\n-- (cut)"
            used += len(text)
            parts.append(f"round {r['round']}, {r.get('network')}, {r['status']}, {p.relative_to(d)}:\n{lean_block(text)}")
            if used >= max_chars:
                break
        if used >= max_chars:
            break
    return "\n\n".join(parts) if parts else "none yet"


def latest_notes(run: RunDir, k: int, max_chars: int) -> str:
    for kk in range(k - 1, 0, -1):
        j = 0
        texts = []
        while (run.round_path(kk) / f"attempt_{j}").is_dir():
            t = run.read(kk, f"attempt_{j}/response.md")
            if t:
                m = re.search(r"NOTES:(.*)\Z", t, re.S)
                texts.append((m.group(1) if m else t).strip())
            j += 1
        if texts:
            out = "\n".join(texts)
            return out[:max_chars] + ("\n(cut)" if len(out) > max_chars else "")
    return "none yet"


def stalled(records: list[dict], k: int, n: int) -> bool:
    """No accepted open-lane submission in agent rounds k-n to k-1 (and there were that many rounds)."""
    if k - 1 < n:
        return False
    recent = [r for r in records if r.get("lane") == "open" and k - n <= r["round"] <= k - 1]
    return not any(r["status"] in ("ok", "verified", "lemma") for r in recent)


def aim(records: list[dict], entries: list[dict], cfg: dict, k: int) -> str:
    summary = summarize(records, entries, cfg.get("budget_ratio"))
    key = {"Q": "Q", "Q_grid": "Q_grid", "cost_of_finishing": "cost_of_finishing"}[cfg["metric"]]
    fmt = "{:.2f}" if cfg["metric"] == "cost_of_finishing" else "{:.4f}"
    scores = "; ".join(f"{s} median {fmt.format(v[key + '_median'])}, worst {fmt.format(v[key + '_worst'])}" for s, v in summary.items())
    text = AIM_TEXT.format(scores=scores)
    if stalled(records, k, cfg["open_lane"]["stall_rounds"]):
        text += "\n" + STALL_TEXT.format(n=cfg["open_lane"]["stall_rounds"])
    return text


def build_open_lane_prompt(cfg: dict, folder: ModelFolder, entries: list[dict], run: RunDir, k: int) -> Prompt:
    part1_t, part2_t = load_template(cfg["prompt_template"])
    terms = forbidden(folder)
    part1 = fill(part1_t, static_values(cfg, folder, entries))
    leaked = [t for t in terms if t in part1]
    if leaked:
        raise PromptError(f"the model folder puts forbidden strings into the prompt: {leaked}")
    records = [r for r in run.read_archive() if r["round"] < k]
    budget = cfg["prompt_budget_tokens"] * CHARS_PER_TOKEN
    lemma_chars, notes_chars = MAX_LEMMA_CHARS, MAX_NOTES_CHARS
    trimmed = []
    values = {
        "SCOREBOARD": scoreboard(records, entries, cfg.get("budget_ratio")),
        "RECEIPTS": receipts_text(records),
        "AIM": aim(records, entries, cfg, k),
    }
    while True:
        values["LEMMAS"] = accepted_sources(run, records, cfg["shown_top_recipes"], lemma_chars)
        values["NOTES"] = latest_notes(run, k, notes_chars)
        part2 = fill(part2_t, values)
        if len(part1) + len(part2) <= budget or (lemma_chars == 0 and notes_chars == 0):
            break
        if lemma_chars > 0:
            lemma_chars = 0
            trimmed.append("lemmas")
        else:
            notes_chars = 0
            trimmed.append("notes")
    stats = {"chars": len(part1) + len(part2), "tokens_estimate": (len(part1) + len(part2)) // CHARS_PER_TOKEN,
             "trimmed": trimmed, "template": cfg["prompt_template"], "lane": "open", "stalled": stalled(records, k, cfg["open_lane"]["stall_rounds"])}
    return Prompt(part1, part2, stats)
