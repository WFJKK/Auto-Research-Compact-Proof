"""The loop driver.

    python -m core.loop run --config config/<run>.yaml
    python -m core.loop resume --run <runs_dir>/<run_id>
    python -m core.loop status --run <runs_dir>/<run_id>

Round 0 replays the model folder's baselines (the baseline frontier); rounds
1 to rounds_max ask the run's backend for recipes_per_round responses each. A
run can stop at any point and resume: completed rounds are kept, and an
incomplete round reuses any response it already saved. A run refuses to go on
if the checker, the sandbox code or the model folder changed since it started.

Recipes run in the sandbox the run config asks for (core.sandbox). Each start
and resume first runs the sandbox probe; a hardened mode that lets anything
critical out is refused, and the weak process mode is reported.

Prompt building (Step 7), patience and the API backend (Step 8) come later;
until then the fake backend stands in for the agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.backends import BackendError
from .agent.backends.fake import FakeBackend
from .agent.backends.manual import ManualBackend
from .agent.build_prompt import Prompt, PromptError, build_prompt
from .diagnostics import render_attempt
from .archive import RunDir
from .model_folder import load_model_folder
from .rounds import Context, RoundError, now, run_round
from .run_config import load_run_config, resolve
from .runner import Limits
from .sandbox import SandboxError, check_probe, choose, probe, protected_paths
from .scoring import summarize
from .util import git_state, read_json, write_json
from .versions import changed, collect
from .zoo import networks


class LoopError(RuntimeError):
    pass


class _NoResponses:
    name = "none"

    def available(self, index):
        return False


def select_networks(folder, cfg) -> list[dict]:
    dev = networks(folder, "dev")
    order = {s: i for i, s in enumerate(folder.setting_names())}
    dev.sort(key=lambda e: (order.get(e["setting"], len(order)), e["seed"]))
    if cfg["networks"] == "all":
        chosen = dev
    else:
        by_id = {e["id"]: e for e in dev}
        unknown = [n for n in cfg["networks"] if n not in by_id]
        if unknown:
            raise LoopError(f"not development networks of {folder.name}: {unknown}")
        chosen = [by_id[n] for n in cfg["networks"]]
    if not chosen:
        raise LoopError(f"{folder.name} has no development networks; train its zoo first")
    missing = [e["id"] for e in chosen if e.get("E") is None or e.get("B") is None]
    if missing:
        raise LoopError(f"E and B are missing for {missing}; run python -m core.check.costs on the model folder")
    return chosen


def make_backend(cfg, log=print):
    if cfg["backend"] == "fake":
        return FakeBackend([resolve(p) for p in cfg["fake_responses"]])
    if cfg["backend"] == "manual":
        return ManualBackend(log=log)
    raise LoopError(f"the {cfg['backend']} backend is not built yet")


def round_prompt(ctx: Context, k: int) -> Prompt:
    """The prompt round k sent, if it has one already; otherwise a new one from the archive."""
    saved = ctx.run.read(k, "prompt.json")
    if saved is not None:
        return Prompt.from_json(saved)
    p = build_prompt(ctx.cfg, ctx.folder, ctx.entries, ctx.run, k)
    ctx.log(f"  prompt: about {p.stats['tokens_estimate']:,} tokens" + (f"; trimmed {', '.join(p.stats['trimmed'])}" if p.stats["trimmed"] else ""))
    return p


def prepare_sandbox(cfg: dict, folder, run: RunDir, entries: list[dict], log=print):
    """Choose the run's sandbox and probe it; returns the sandbox and the probe's report."""
    sb = choose(cfg, protected_paths(cfg, folder, run.path))
    report = probe(sb, folder, entries[0]["id"], run.path / "probe", Limits.from_config(cfg))
    check_probe(sb, report)
    if sb.hardened:
        log(f"sandbox: {sb.mode}; the probe's {len(report['attempts'])} escape attempts were all blocked")
    else:
        log(
            f"warning: sandbox mode process is weak; recipes can read and write what you can "
            f"(the probe got out with {', '.join(report['critical_escapes'])}). Use container or landlock mode "
            "for recipes you do not trust."
        )
    return sb, report


def open_context(run: RunDir, cfg: dict, log=print) -> Context:
    sys.dont_write_bytecode = True  # so compiled files that appear in trusted folders can only come from elsewhere
    folder = load_model_folder(resolve(cfg["model"]))
    started = read_json(run.path / "versions.json")
    bad = changed(started, folder)
    if bad:
        raise LoopError(
            f"{', '.join(bad)} changed since run {run.run_id} started; results would not be comparable. "
            "Start a new run instead."
        )
    entries = select_networks(folder, cfg)
    sb, report = prepare_sandbox(cfg, folder, run, entries, log)
    with open(run.path / "sandbox.jsonl", "a") as f:
        f.write(json.dumps({"at": now(), **sb.describe(), "probe": report}) + "\n")
    return Context(
        cfg=cfg,
        folder=folder,
        run=run,
        entries=entries,
        versions=started,
        limits=Limits.from_config(cfg),
        sandbox=sb,
        log=log,
    )


def start(config_path: str | Path, log=print, run_id: str | None = None) -> Context:
    cfg = load_run_config(config_path, run_id=run_id)
    folder = load_model_folder(resolve(cfg["model"]))
    select_networks(folder, cfg)  # fail early
    run = RunDir(resolve(cfg["runs_dir"]), cfg["run_id"])
    run.create(cfg)
    config_sha = (run.path / "config.sha256").read_text().strip()
    versions = collect(folder, cfg, config_sha)
    write_json(run.path / "versions.json", {**versions, "git": git_state(), "created": now()})
    log(f"run {cfg['run_id']} at {run.path}")
    return open_context(run, run.config(), log)


def resume(run_path: str | Path, log=print) -> Context:
    run_path = Path(run_path).expanduser().resolve()
    run = RunDir(run_path.parent, run_path.name)
    if not run.exists():
        raise LoopError(f"no run at {run_path}")
    return open_context(run, run.config(), log)


def drive(ctx: Context) -> None:
    cfg, run, folder = ctx.cfg, ctx.run, ctx.folder
    if not run.round_done(0):
        baselines = folder.path / "baselines"
        if cfg["baselines"] and baselines.is_dir():
            fake = FakeBackend([baselines])
            run_round(ctx, 0, fake, 0, len(fake), "baseline")
        else:
            run_round(ctx, 0, _NoResponses(), 0, 0, "baseline")
    backend = make_backend(cfg, ctx.log)
    per_round = cfg["recipes_per_round"]
    k = max(run.next_round(), 1)
    while k <= cfg["rounds_max"]:
        first = (k - 1) * per_round
        fresh = not run.round_done(k) and run.read(k, "attempt_0/response.md") is None
        if fresh and not backend.available(first):
            ctx.log(f"the {backend.name} backend has no more responses; stopping after round {k - 1}")
            break
        run_round(ctx, k, backend, first, per_round, "agent", prompt=round_prompt(ctx, k))
        k += 1


# status ------------------------------------------------------------------------
def status_text(run: RunDir, diagnostics: bool = False) -> str:
    cfg = run.config()
    lines = [f"run {run.run_id}: model {cfg['model']}, backend {cfg['backend']}, rounds done {run.done_rounds()}"]
    records = run.read_archive()
    try:
        grouping = load_model_folder(resolve(cfg["model"])).group_label()
    except Exception:
        grouping = "group"
    for k in sorted({r["round"] for r in records}):
        for j in sorted({r["attempt"] for r in records if r["round"] == k}):
            rs = [r for r in records if r["round"] == k and r["attempt"] == j]
            meta = run.read(k, f"attempt_{j}/response_meta.json") or {}
            counts: dict[str, int] = {}
            for r in rs:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            text = ", ".join(f"{s} {c}" for s, c in sorted(counts.items()))
            errs = sorted({(r["error"] or "").splitlines()[-1][:100] for r in rs if r["error"]})
            lines.append(f"  round {k} attempt {j} ({rs[0]['source']}, {meta.get('source', '')}): {text}")
            if diagnostics:
                lines += ["      " + ln for ln in render_attempt(rs, grouping).splitlines()]
            else:
                lines += [f"      {e}" for e in errs[:2]]
    try:
        folder = load_model_folder(resolve(cfg["model"]))
        entries = select_networks(folder, cfg)
    except Exception as exc:  # status should still print what it can
        lines.append(f"(no metrics: {exc})")
        return "\n".join(lines)
    lines.append("frontier metrics (Python-checked):")
    for setting, s in summarize(records, entries).items():
        lines.append(
            f"  {setting}: Q median {s['Q_median']:.4f}, worst {s['Q_worst']:.4f}; "
            f"cost of finishing median {s['cost_of_finishing_median']:.2f}"
        )
        for nid, m in s["networks"].items():
            full = f"{m['shortest_full']:,} (B/{m['B'] / m['shortest_full']:.1f})" if m["shortest_full"] else "none"
            real = f"{m['real_accuracy']:.5f}" if m["real_accuracy"] is not None else "?"
            lines.append(
                f"    {nid}: Q {m['Q']:.4f}, best certified accuracy {m['best_accuracy']:.5f} of {real}, "
                f"shortest full-accuracy proof {full}"
            )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="start a run from a run config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--run-id", help="use this run id instead of the config's, e.g. to repeat a run")
    for name in ("resume", "status"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="the run's folder, <runs_dir>/<run_id>")
    sub.choices["status"].add_argument("--diagnostics", action="store_true", help="show each attempt's diagnostics")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "status":
            path = Path(args.run).expanduser().resolve()
            print(status_text(RunDir(path.parent, path.name), diagnostics=args.diagnostics))
            return 0
        ctx = start(args.config, run_id=args.run_id) if args.cmd == "run" else resume(args.run)
        drive(ctx)
        print(status_text(ctx.run))
        return 0
    except (LoopError, RoundError, BackendError, SandboxError, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
