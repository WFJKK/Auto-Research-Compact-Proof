"""Train a model's zoo from its folder alone.

    python -m core.train models/<name> [--settings S ...] [--seeds N ...] [--force]

Every network is trained from its own seed (parameter initialisation and data),
on fresh uniformly random inputs each step, with the settings in the model's
config.yaml. Training is resumable: finished networks are skipped unless their
training settings changed or --force is given. The manifest is rewritten
atomically after each network.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import inputs
from .model_folder import ModelFolder, load_model_folder
from .util import sha256_file, sha256_json
from .zoo import read_manifest, split_assignment, write_manifest

TRAINER_VERSION = "train-v1"


def training_key(folder: ModelFolder, setting: str, seed: int) -> str:
    """What a network's training depends on; a change means it must be retrained."""
    return sha256_json(
        {
            "trainer": TRAINER_VERSION,
            "model_source": folder.source(),
            "task_source": (folder.path / "task.py").read_text(),
            "sizes": folder.sizes(setting),
            "training": folder.config["training"],
            "seed": seed,
        }
    )


def train_network(folder: ModelFolder, setting: str, seed: int, log=print):
    import torch

    cfg = folder.config["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    module = folder.build(setting)
    opt_cls = getattr(torch.optim, cfg["optimizer"])
    opt_kwargs = {"lr": cfg["lr"]}
    if "weight_decay" in cfg:
        opt_kwargs["weight_decay"] = cfg["weight_decay"]
    opt = opt_cls(module.parameters(), **opt_kwargs)
    space = folder.input_space(setting)
    sizes = folder.sizes(setting)
    loss_fn = torch.nn.CrossEntropyLoss()
    module.train()
    steps = int(cfg["steps"])
    report_every = max(1, steps // 5)
    loss = None
    for step in range(steps):
        tokens = inputs.random_inputs(space, int(cfg["batch_size"]), rng)
        x = folder.encode(tokens, setting)
        y = torch.as_tensor(np.asarray(folder.task.label(tokens, sizes), dtype=np.int64))
        loss = loss_fn(module(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % report_every == 0:
            log(f"    step {step + 1}/{steps} loss {loss.item():.4f}")
    module.eval()
    return module, float(loss.item()) if loss is not None else float("nan")


def float_correct(folder: ModelFolder, setting: str, module, batch: int = 8192) -> np.ndarray:
    """Whether each input's label logit is strictly larger than every other logit (float32)."""
    import torch

    space = folder.input_space(setting)
    sizes = folder.sizes(setting)
    toks = inputs.all_inputs(space)
    out = np.zeros(len(toks), dtype=bool)
    with torch.no_grad():
        for i in range(0, len(toks), batch):
            t = toks[i : i + batch]
            logits = module(folder.encode(t, setting)).numpy()
            lab = np.asarray(folder.task.label(t, sizes))
            own = logits[np.arange(len(t)), lab]
            others = logits.copy()
            others[np.arange(len(t)), lab] = -np.inf
            out[i : i + batch] = own > others.max(axis=1)
    return out


def train_zoo(folder: ModelFolder, settings=None, seeds=None, force=False, log=print) -> dict:
    split = split_assignment(folder)
    manifest = read_manifest(folder)
    manifest["model"] = folder.name
    by_id = {n["id"]: n for n in manifest.get("networks", [])}
    folder.zoo_dir().mkdir(parents=True, exist_ok=True)
    for setting in settings or folder.setting_names():
        for seed in seeds if seeds is not None else folder.config["seeds"]:
            nid = folder.network_id(setting, seed)
            key = training_key(folder, setting, seed)
            old = by_id.get(nid)
            wpath = folder.zoo_dir() / f"{nid}.npz"
            if (
                not force
                and old is not None
                and old.get("training_key") == key
                and wpath.exists()
                and sha256_file(wpath) == old.get("sha256")
            ):
                log(f"{nid}: up to date")
                continue
            log(f"{nid}: training")
            t0 = time.time()
            module, final_loss = train_network(folder, setting, seed, log=log)
            weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in module.state_dict().items()}
            np.savez(wpath, **weights)
            correct = float_correct(folder, setting, module)
            cpath = folder.zoo_dir() / f"{nid}.correct.npy"
            np.save(cpath, correct)
            entry = {
                "id": nid,
                "setting": setting,
                "seed": seed,
                "split": split[(setting, seed)],
                "weights": wpath.name,
                "sha256": sha256_file(wpath),
                "correct": cpath.name,
                "n_inputs": int(len(correct)),
                "real_correct": int(correct.sum()),
                "real_accuracy": float(correct.mean()),
                "final_loss": final_loss,
                "training_key": key,
                "train_seconds": round(time.time() - t0, 2),
                "E": old.get("E") if old else None,
                "B": old.get("B") if old else None,
            }
            # A retrained network invalidates its costs.
            if old is not None and old.get("sha256") != entry["sha256"]:
                entry["E"] = entry["B"] = None
            by_id[nid] = entry
            manifest["networks"] = sorted(by_id.values(), key=lambda n: (n["setting"], n["seed"]))
            write_manifest(folder, manifest)
            log(f"{nid}: real accuracy {entry['real_accuracy']:.4f} ({entry['split']})")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_folder")
    ap.add_argument("--settings", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    folder = load_model_folder(args.model_folder)
    train_zoo(folder, settings=args.settings, seeds=args.seeds, force=args.force)


if __name__ == "__main__":
    main()
