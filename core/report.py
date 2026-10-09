"""Reports from a run's archive: tables and plots.

    python -m core.report --run <runs_dir>/<run_id> [--results]

Writes <run>/report/report.md and its images: per size setting the metrics
(Q, cost of finishing and the directional targets: the shortest proof at full
accuracy and the best accuracy within B/10, B/100 and B/1000), the median and
the worst network; a frontier plot per size setting, with the round-0
baselines; the metric after each round; the claims of the recipes that hold
the frontier; the held-out numbers if core.final has run; and the API tokens
and cost. Every number is labelled Python-checked unless Lean checked it.

With --results, the report (and the held-out summary) is copied into
results/<run_id>/ and a line is added to results/RESULTS.md, ready to commit.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

from .agent.build_prompt import label
from .archive import RunDir
from .lean import lean_command
from .loop import LoopError, select_networks
from .model_folder import load_model_folder
from .run_config import resolve
from .scoring import BUDGET_DIVISORS, frontier, summarize
from .util import REPO_ROOT, read_json

RESULTS = REPO_ROOT / "results"


def lean_lines(run: RunDir) -> int:
    path = run.path / "lean.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _metric_rows(s: dict) -> list[tuple[str, str, str]]:
    nets = list(s["networks"].values())
    rows = [
        ("Q", f"{s['Q_median']:.4f}", f"{s['Q_worst']:.4f}"),
        ("cost of finishing (B / total)", f"{s['cost_of_finishing_median']:.2f}", f"{s['cost_of_finishing_worst']:.2f}"),
    ]
    full = [m["B"] / m["shortest_full"] if m["shortest_full"] else None for m in nets]
    if all(f is not None for f in full):
        rows.append(("shortest proof at full accuracy, as B/x", f"B/{statistics.median(full):.1f}", f"B/{min(full):.1f}"))
    else:
        rows.append(("shortest proof at full accuracy, as B/x", "not reached on every network", ""))
    for d in BUDGET_DIVISORS:
        acc = [m["within"][f"B/{d}"] for m in nets]
        rows.append((f"best certified accuracy within B/{d}", f"{100 * statistics.median(acc):.3f}%", f"{100 * min(acc):.3f}%"))
    return rows


def plot_frontiers(records, entries, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = frontier(records)
    paths = []
    for setting in dict.fromkeys(e["setting"] for e in entries):
        nets = [e for e in entries if e["setting"] == setting]
        E, B = nets[0]["E"], nets[0]["B"]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for e in nets:
            pts = fr.get(e["id"], [])
            xs = [p["length"] for p in pts] + [B]
            ys = [p["certified"] / p["n_inputs"] for p in pts]
            if pts:
                ax.step(xs, ys + [ys[-1]], where="post", lw=1.2, label=e["id"])
                ax2.step(xs, [max(1 - y, 1e-6) for y in ys + [ys[-1]]], where="post", lw=1.2)
        ok = [r for r in records if r["status"] == "ok" and r["setting"] == setting]
        for source, marker, color in (("baseline", "s", "0.5"), ("agent", "o", "C3")):
            rs = [r for r in ok if r["source"] == source]
            if rs:
                xs = [r["length"] for r in rs]
                ys = [r["certified"] / r["n_inputs"] for r in rs]
                ax.scatter(xs, ys, s=12, marker=marker, color=color, alpha=0.5, label=f"{source} proofs")
                ax2.scatter(xs, [max(1 - y, 1e-6) for y in ys], s=12, marker=marker, color=color, alpha=0.5)
        for a in (ax, ax2):
            a.set_xscale("log")
            a.axvline(E, ls="--", color="k", lw=0.8)
            a.axvline(B, ls="--", color="k", lw=0.8)
            a.set_xlabel("proof length (operations); dashed: E and B")
        ax2.set_yscale("log")
        ax.set_ylabel("certified accuracy")
        ax2.set_ylabel("uncertified share of inputs")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{setting}: frontier per network")
        ax2.set_title(f"{setting}: what is left uncertified")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        path = out / f"frontier_{setting}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def progress(run: RunDir, records, entries, metric_key: str) -> list[tuple[int, dict]]:
    out = []
    for k in run.done_rounds():
        s = summarize([r for r in records if r["round"] <= k], entries)
        out.append((k, {setting: v[metric_key] for setting, v in s.items()}))
    return out


def plot_progress(points, out: Path, metric_name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    settings = list(points[0][1]) if points else []
    for s in settings:
        ax.step([k for k, _ in points], [v[s] for _, v in points], where="post", marker="o", ms=3, label=s)
    ax.set_xlabel("round (0 = baselines)")
    ax.set_ylabel(f"{metric_name}, median over networks")
    ax.legend()
    fig.tight_layout()
    path = out / "progress.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def build_report(run_path: str | Path) -> Path:
    run_path = Path(run_path).expanduser().resolve()
    run = RunDir(run_path.parent, run_path.name)
    if not run.exists():
        raise LoopError(f"no run at {run_path}")
    cfg = run.config()
    folder = load_model_folder(resolve(cfg["model"]))
    entries = select_networks(folder, cfg)
    records = run.read_archive()
    versions = read_json(run.path / "versions.json")
    out = run.path / "report"
    out.mkdir(exist_ok=True)
    summary = summarize(records, entries)
    metric_key = "cost_of_finishing_median" if cfg["metric"] == "cost_of_finishing" else "Q_median"

    metas = [run.read(k, "meta.json") or {} for k in run.done_rounds()]
    tokens_in = sum((m.get("tokens") or {}).get("input", 0) for m in metas)
    tokens_out = sum((m.get("tokens") or {}).get("output", 0) for m in metas)
    cost = sum(m.get("cost_usd") or 0 for m in metas)
    modes = sorted({(m.get("sandbox") or {}).get("mode", "?") for m in metas if m.get("sandbox")})
    statuses: dict[str, int] = {}
    for r in records:
        if r["round"] > 0:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    lines = [
        f"# Report: {run.run_id}",
        "",
        f"- Model: `{cfg['model']}`; backend {cfg['backend']}"
        + (f", agent model {cfg['agent_model']} (thinking {cfg['thinking']}, effort {cfg['effort']})" if cfg["backend"] == "api" else ""),
        f"- Rounds done: {len([k for k in run.done_rounds() if k > 0])} agent rounds after round 0; metric {cfg['metric']}",
        f"- Code: commit {(versions.get('git') or {}).get('commit')}" + (" (with uncommitted changes)" if (versions.get("git") or {}).get("dirty") else ""),
        f"- Cost model {versions.get('cost_model')}, rule set {versions.get('rule_set')}, prompt template {cfg['prompt_template']}, sandbox {', '.join(modes) or 'unknown'}",
        f"- API use: {tokens_in:,} input and {tokens_out:,} output tokens, about ${cost:.2f}",
        "- Every number is Python-checked: this model has no Lean checker yet."
        if lean_command(folder) is None
        else f"- Numbers are Python-checked; Lean spot checks on frontier entries are in lean.jsonl ({lean_lines(run)} so far).",
        "",
        "## Results per size setting",
        "",
    ]
    for setting, s in summary.items():
        lines += [f"### {setting} ({len(s['networks'])} development networks)", "", "| Metric | Median network | Worst network |", "| --- | --- | --- |"]
        lines += [f"| {a} | {b} | {c} |" for a, b, c in _metric_rows(s)]
        lines += ["", f"![frontier {setting}](frontier_{setting}.png)", ""]

    points = progress(run, records, entries, metric_key)
    plot_frontiers(records, entries, out)
    if points:
        plot_progress(points, out, cfg["metric"])
        lines += ["## Progress", "", "![progress](progress.png)", ""]

    keys = sorted({(p["round"], p["attempt"]) for pts in frontier(records).values() for p in pts})
    lines += ["## Recipes that hold the frontier", ""]
    fr = frontier(records)
    for key in keys:
        held = sorted({nid for nid, pts in fr.items() for p in pts if (p["round"], p["attempt"]) == key})
        parsed = run.read(key[0], f"attempt_{key[1]}/parsed.json") or {}
        lines += [
            f"**{label(run, key, cfg['recipes_per_round'])}** (on {len(held)} of {len(entries)} networks)",
            "",
            f"> {parsed.get('claim', '').strip()}",
            "",
        ]
    if statuses:
        lines += ["## Agent attempts by status", "", ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())), ""]

    final = run.path / "final" / "final.json"
    if final.exists():
        f = read_json(final)
        lines += [f"## Held-out networks (checked by {f['checked_by']})", ""]
        order = [st for st in dict.fromkeys(e["setting"] for e in entries) if st in f["settings"]]
        for setting in order:
            s = f["settings"][setting]
            lines += [f"### {setting}", "", "| Metric | Development | Held out |", "| --- | --- | --- |"]
            for m in s["development"]:
                d, h = s["development"][m], s["held_out"][m]
                lines.append(f"| {m} | {d if d is None else round(d, 4)} | {h if h is None else round(h, 4)} |")
            lines.append("")
    else:
        lines += ["## Held-out networks", "", "Not evaluated yet: run `python -m core.final --run <this run>`.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return out / "report.md"


def copy_to_results(run_path: str | Path) -> Path:
    run_path = Path(run_path).expanduser().resolve()
    run = RunDir(run_path.parent, run_path.name)
    cfg = run.config()
    dest = RESULTS / run.run_id
    if dest.exists():
        raise LoopError(f"{dest} already exists; results are never overwritten")
    shutil.copytree(run.path / "report", dest)
    for name in ("final.json", "final.md"):
        src = run.path / "final" / name
        if src.exists():
            shutil.copy2(src, dest / name)
    shutil.copy2(run.path / "config.yaml", dest / "config.yaml")
    versions = read_json(run.path / "versions.json")
    folder = load_model_folder(resolve(cfg["model"]))
    s = summarize(run.read_archive(), select_networks(folder, cfg))
    headline = "; ".join(f"{k} Q {v['Q_median']:.4f} (worst {v['Q_worst']:.4f})" for k, v in s.items())
    index = RESULTS / "RESULTS.md"
    if not index.exists():
        index.write_text(
            "# Results\n\nOne line per run. Raw run data stays outside the repository.\n\n"
            "| Run | Model | Commit | Backend | Development (median Q) |\n| --- | --- | --- | --- | --- |\n"
        )
    with open(index, "a") as f:
        f.write(
            f"| [{run.run_id}]({run.run_id}/report.md) | {cfg['model']} | {(versions.get('git') or {}).get('commit', '')[:10]} "
            f"| {cfg['backend']} | {headline} |\n"
        )
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run's folder, <runs_dir>/<run_id>")
    ap.add_argument("--results", action="store_true", help="also copy the report into results/<run_id>/")
    args = ap.parse_args(argv)
    try:
        path = build_report(args.run)
        print(f"report: {path}")
        if args.results:
            print(f"copied to {copy_to_results(args.run)}")
        return 0
    except LoopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
