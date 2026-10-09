"""Run configs: one YAML file per run, merged over config/defaults.yaml.

A run config must name `run_id` and `model` (the model folder) and may override
any default. Unknown keys are refused, so a typo never silently falls back to
a default. Relative paths are relative to the repository root.
"""

from __future__ import annotations

import re
from pathlib import Path

from .check.rules import GENERIC_RULES
from .util import REPO_ROOT, read_yaml

DEFAULTS_PATH = REPO_ROOT / "config" / "defaults.yaml"
REQUIRED = ("run_id", "model")
BACKENDS = ("fake", "manual", "api")
METRICS = ("Q", "cost_of_finishing")
SANDBOX_MODES = ("auto", "container", "landlock", "process")
FORBIDDEN_KEY_VARIABLES = ("ANTHROPIC_API_KEY",)


class RunConfigError(ValueError):
    pass


def resolve(path: str | Path) -> Path:
    """A path from a run config: absolute, home-relative, or relative to the repository."""
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def defaults() -> dict:
    return dict(read_yaml(DEFAULTS_PATH))


def load_run_config(path: str | Path, run_id: str | None = None) -> dict:
    raw = read_yaml(Path(path))
    if not isinstance(raw, dict):
        raise RunConfigError(f"{path}: a run config is a mapping")
    if run_id is not None:
        raw = {**raw, "run_id": run_id}
    return merge(raw, source=str(path))


def merge(raw: dict, source: str = "run config") -> dict:
    base = defaults()
    unknown = sorted(set(raw) - set(base) - set(REQUIRED))
    if unknown:
        raise RunConfigError(f"{source}: unknown keys {unknown}")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise RunConfigError(f"{source}: missing {missing}")
    cfg = {**base, **raw}
    validate(cfg, source)
    return cfg


def _number(cfg, key, errors, integer=False, minimum=None, positive=False):
    v = cfg.get(key)
    if isinstance(v, bool) or not isinstance(v, int if integer else (int, float)):
        errors.append(f"{key} must be {'an integer' if integer else 'a number'}")
    elif positive and v <= 0:
        errors.append(f"{key} must be positive")
    elif minimum is not None and v < minimum:
        errors.append(f"{key} must be at least {minimum}")


def validate(cfg: dict, source: str = "run config") -> None:
    errors = []
    if not isinstance(cfg["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cfg["run_id"]):
        errors.append("run_id must be letters, digits, '.', '_' or '-'")
    if not isinstance(cfg["model"], str) or not (resolve(cfg["model"]) / "model.py").is_file():
        errors.append(f"model {cfg['model']!r} is not a model folder")
    knobs = cfg["knob_values"]
    if (
        not isinstance(knobs, list)
        or not knobs
        or not all(isinstance(k, (int, float)) and not isinstance(k, bool) and 0 <= k <= 1 for k in knobs)
        or len(set(float(k) for k in knobs)) != len(knobs)
    ):
        errors.append("knob_values must be a non-empty list of distinct numbers in [0, 1]")
    for key in ("time_limit_s", "memory_limit_mb", "max_proof_mb"):
        _number(cfg, key, errors, positive=True)
    for key in ("threads_per_recipe", "workers", "recipes_per_round", "rounds_max", "patience", "prompt_budget_tokens"):
        _number(cfg, key, errors, integer=True, minimum=1)
    for key in ("shown_top_recipes", "shown_last_attempts", "seed"):
        _number(cfg, key, errors, integer=True, minimum=0)
    for key in ("screen", "baselines", "lean_spot_check"):
        if not isinstance(cfg[key], bool):
            errors.append(f"{key} must be true or false")
    if cfg["backend"] not in BACKENDS:
        errors.append(f"backend must be one of {BACKENDS}")
    if not isinstance(cfg["fake_responses"], list) or not all(isinstance(p, str) for p in cfg["fake_responses"]):
        errors.append("fake_responses must be a list of paths")
    if not isinstance(cfg["api_key_env"], str) or not cfg["api_key_env"]:
        errors.append("api_key_env must name an environment variable")
    elif cfg["api_key_env"] in FORBIDDEN_KEY_VARIABLES:
        errors.append(f"api_key_env must not be {cfg['api_key_env']}: Claude Code would bill its own work to that key")
    rules = cfg["rule_set"]
    if not isinstance(rules, list) or not rules or not set(rules) <= set(GENERIC_RULES):
        errors.append(f"rule_set must be a non-empty list drawn from {list(GENERIC_RULES)}")
    if not isinstance(cfg["prompt_template"], str) or not (REPO_ROOT / "core" / "agent" / "prompts" / f"{cfg['prompt_template']}.md").is_file():
        errors.append(f"prompt_template {cfg['prompt_template']!r} is not in core/agent/prompts")
    if cfg["metric"] not in METRICS:
        errors.append(f"metric must be one of {METRICS}")
    nets = cfg["networks"]
    if nets != "all" and (not isinstance(nets, list) or not nets or not all(isinstance(n, str) for n in nets)):
        errors.append('networks must be "all" or a list of network ids')
    if cfg["sandbox_dir"] is not None and not isinstance(cfg["sandbox_dir"], str):
        errors.append("sandbox_dir must be a path or null")
    if cfg["sandbox"] not in SANDBOX_MODES:
        errors.append(f"sandbox must be one of {SANDBOX_MODES}")
    if not isinstance(cfg["sandbox_image"], str) or not cfg["sandbox_image"]:
        errors.append("sandbox_image must name a container image")
    if errors:
        raise RunConfigError(f"{source}: " + "; ".join(errors))
