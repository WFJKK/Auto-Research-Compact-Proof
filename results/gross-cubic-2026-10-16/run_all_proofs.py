"""All of Gross et al.'s proofs for Max-of-K on our attention zoos: brute force, cubic and every sub-cubic
variant, so that their Q is computed from their whole curve, as the agent's is from its.

    <venv with transformer_lens<3>/bin/python results/gross-cubic-2026-10-16/run_all_proofs.py \
        <checkout of JasonGross/guarantees-based-mechanistic-interpretability> <folder:setting> [network ...]

Their code runs unchanged on a TransformerLens model built from our weights (attention only, one layer, one
head, no biases, no layer norm, output read at the last position: our model). For the min tasks the
vocabulary is reversed so the network computes a max. A check compares the TransformerLens logits with our
module's before anything else runs.

Lengths. The cubic proof's length in our units is counted by walking its loops (run_cubic.py,
length_in_our_units). Every other proof's length is that number times the ratio of its instruction count to
the cubic proof's, both counted by their own symbolic counter (gbmi.utils.instructions.CountTensorOperations:
floating-point operations, integer operations and branches). So all their proofs sit on one scale, anchored
to our count of the cubic proof; the ratios carry their counter's view of relative cost.

Scores. Their frontier per network is the best certified count at each length over all their proofs plus
brute force at B with the network's true count; Q, Q_grid and cost of finishing come from core.scoring,
exactly as for the agent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def build_tl_model(w: dict, sizes: dict, reverse: bool):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    V, n, d, dh = sizes["d_vocab"], sizes["n_ctx"], sizes["d_model"], sizes["d_head"]
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=1, d_model=d, d_head=dh, n_ctx=n, d_vocab=V, d_vocab_out=V, attn_only=True,
        normalization_type=None, positional_embedding_type="standard", default_prepend_bos=False,
        device="cpu", dtype=torch.float64, seed=0,
    )
    model = HookedTransformer(cfg)
    t = {k: torch.tensor(np.asarray(v), dtype=torch.float64) for k, v in w.items()}
    E, U = t["E"], t["U"]
    if reverse:
        E, U = E.flip(0), U.flip(1)
    state = model.state_dict()
    state["embed.W_E"] = E
    state["pos_embed.W_pos"] = t["P"]
    state["blocks.0.attn.W_Q"] = t["WQ"][None]
    state["blocks.0.attn.W_K"] = t["WK"][None]
    state["blocks.0.attn.W_V"] = t["WV"][None]
    state["blocks.0.attn.W_O"] = t["WO"][None]
    for b in ("b_Q", "b_K", "b_V"):
        state[f"blocks.0.attn.{b}"] = torch.zeros(1, dh, dtype=torch.float64)
    state["blocks.0.attn.b_O"] = torch.zeros(d, dtype=torch.float64)
    state["unembed.W_U"] = U
    state["unembed.b_U"] = torch.zeros(V, dtype=torch.float64)
    model.load_state_dict(state)
    model.eval()
    return model


def check_same_function(model, folder, setting, nid, reverse, n=256):
    from core.zoo import load_module

    sizes = folder.sizes(setting)
    V, P = sizes["d_vocab"], sizes["n_ctx"]
    ours = load_module(folder, nid).double()
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=(n, P))
    with torch.no_grad():
        z_ours = ours(folder.encode(toks, setting).double()).numpy()
        tl_toks = (V - 1 - toks) if reverse else toks
        z_tl = model(torch.tensor(tl_toks))[:, -1, :].numpy()
    if reverse:
        z_tl = z_tl[:, ::-1]
    err = float(np.abs(z_ours - z_tl).max())
    if err > 1e-6:
        raise SystemExit(f"{nid}: TransformerLens model differs from ours by {err}")
    return err


def counted(fn):
    """Run fn under their symbolic instruction counter; return (result, total instructions, flops)."""
    from gbmi.utils import instructions
    from gbmi.utils.instructions import CountTensorOperations, PatchTorch

    with PatchTorch():
        with instructions.set_sanity_check(False):
            with CountTensorOperations() as count:
                result = fn()
    return result, int(count.flop + count.int_op + count.branch), int(count.flop)


def run_network(folder, setting, entry, reverse, length_cubic_ours):
    import gbmi.exp_max_of_n.analysis.quadratic as analysis_quadratic
    import gbmi.exp_max_of_n.analysis.subcubic as analysis_subcubic
    import gbmi.exp_max_of_n.verification.cubic as cubic
    import gbmi.exp_max_of_n.verification.subcubic as subcubic
    from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
    from gbmi.utils.dataclass import enumerate_dataclass_values
    from gbmi.utils.instructions import CountHookedTransformer

    from core.zoo import load_weights

    sizes = folder.sizes(setting)
    model = build_tl_model(load_weights(folder, entry["id"]), sizes, reverse)
    err = check_same_function(model, folder, setting, entry["id"], reverse)
    cmodel = CountHookedTransformer(model)
    out = {"id": entry["id"], "check_max_abs_diff": err, "proofs": []}

    t0 = time.time()
    res = cubic.verify_proof(model, {}, print_complexity=False, print_results=False, sanity_check=False)
    _, cubic_count, cubic_flop = counted(
        lambda: cubic.verify_proof(cmodel, {}, print_complexity=False, print_results=False, sanity_check=False)
    )
    out["proofs"].append({
        "proof": "cubic", "certified": int(res["correct_count_lower_bound"]), "their_count": cubic_count,
        "their_flop": cubic_flop, "length": int(length_cubic_ours), "seconds": round(time.time() - t0, 1),
    })
    scale = length_cubic_ours / cubic_count

    W_EP_direction_kwargs, find_min_gaps_kwargs, size_and_query_directions_kwargs, _ = (
        analysis_subcubic.find_proof_shared(model)
    )
    for tricks in enumerate_dataclass_values(LargestWrongLogitQuadraticConfig):
        t0 = time.time()
        try:
            min_gaps, _ = analysis_subcubic.find_min_gaps_with_EQKE(
                model=model, **find_min_gaps_kwargs, **size_and_query_directions_kwargs, tricks=tricks,
                record_time=True,
            )
            W_EP_direction = analysis_quadratic.W_EP_direction_for_tricks(**W_EP_direction_kwargs, tricks=tricks)
            kwargs = dict(W_EP_direction=W_EP_direction, **size_and_query_directions_kwargs, min_gaps=min_gaps,
                          tricks=tricks, sanity_check=False, print_complexity=False, print_results=False)
            res = subcubic.verify_proof(model, **kwargs)
            _, count, flop = counted(lambda: subcubic.verify_proof(cmodel, **kwargs))
        except Exception as exc:  # a variant their code cannot run here is reported, not hidden
            out["proofs"].append({"proof": tricks.short_description(), "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        out["proofs"].append({
            "proof": tricks.short_description(), "certified": int(res["correct_count_lower_bound"]),
            "their_count": count, "their_flop": flop, "length": int(round(count * scale)),
            "seconds": round(time.time() - t0, 1),
        })
    return out


def scores(net: dict, entry: dict) -> dict:
    from core.scoring import network_metrics

    n_inputs = entry["n_inputs"]
    points = [p for p in net["proofs"] if "certified" in p and p["certified"] > 0]
    points.append({"length": entry["B"], "certified": entry["real_correct"]})
    frontier, best = [], -1
    for p in sorted(points, key=lambda p: (p["length"], -p["certified"])):
        if p["certified"] > best:
            frontier.append({"length": p["length"], "certified": p["certified"], "n_inputs": n_inputs})
            best = p["certified"]
    m = network_metrics(frontier, entry)
    return {"Q": m["Q"], "Q_grid": m["Q_grid"], "cost_of_finishing": m["cost_of_finishing"], "frontier": frontier}


def main(checkout: str, which: str, ids: list[str]) -> None:
    sys.path.insert(0, checkout)
    from core.model_folder import load_model_folder
    from core.zoo import networks

    sys.path.insert(0, str(HERE))
    from run_cubic import length_in_our_units

    name, setting = which.split(":")
    folder = load_model_folder(REPO / "models" / name)
    sizes = folder.sizes(setting)
    entries = [e for e in networks(folder) if e["setting"] == setting and (not ids or e["id"] in ids)]
    path = HERE / f"all_proofs_{name}_{setting}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for e in entries:
        if e["id"] in done:
            print(f"{e['id']}: done before", flush=True)
            continue
        L = length_in_our_units(sizes["d_vocab"], sizes["n_ctx"], sizes["d_model"], sizes["d_head"], e["E"])
        net = run_network(folder, setting, e, "min" in name, L)
        net["scores"] = scores(net, e)
        done[e["id"]] = net
        path.write_text(json.dumps(done, indent=1))  # after every network, so an interrupted run resumes
        s = net["scores"]
        ok = [p for p in net["proofs"] if "certified" in p]
        print(f"{e['id']}: {len(ok)} proofs ({len(net['proofs']) - len(ok)} failed), frontier "
              f"{len(s['frontier'])} points, Q {s['Q']:.4f}, Q_grid {s['Q_grid']:.4f}, "
              f"cost of finishing {s['cost_of_finishing']:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
