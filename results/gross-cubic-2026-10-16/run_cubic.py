"""Gross et al.'s cubic proof for Max-of-K, run on our attention zoos and measured in our units.

    python results/gross-cubic-2026-10-16/run_cubic.py <checkout of JasonGross/guarantees-based-mechanistic-interpretability>

The proof itself is their code, unchanged: compute_extreme_softmaxed_right_attention_cubic_simple,
compute_largest_wrong_logit_cubic and count_correct_sequences_cubic from
gbmi/exp_max_of_n/verification/cubic.py (MIT licence), loaded from their checkout with its package
imports removed (they pull in TransformerLens, which the three functions do not need). We supply what
their verify_proof would compute from a TransformerLens model, from our weights:

    EUPU = (E + P[-1]) U          the direct path of the query token
    EVOU = E WV WO U              the value-output path of each key token
    PVOU = P WV WO U              the same for each key position
    EQKE = (E + P[-1]) WQ WK^T E^T    query token to key token attention scores
    EQKP = (E + P[-1]) WQ WK^T P^T    query token to key position
    attn_scale = sqrt(d_head)

which is our model's computation (no biases, no layer norm, output read at the last position). For the
min tasks the vocabulary is reversed (t -> V - 1 - t, rows of E and columns of U), which turns the
network into one computing the max of the reversed tokens, so their max proof applies unchanged.

Their proof runs in float64 and is not checked by our exact checker, so its count is compared with the
network's true number of correct inputs as a sanity check (it must never exceed it).

Its length is counted here in our cost model's units (core/check/cost.py: 1 per multiply-add, addition,
subtraction or comparison, 3 per softmax entry, matrix products in the cheapest order), by walking the
same loops their code runs: E (reading the weights) + the five path matrices + the attention loop +
the logit loop + the counting loop. Q, Q_grid and cost of finishing then come from core.scoring with the
proof's point plus brute force at B, exactly as for the agent's proofs.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.model_folder import load_model_folder  # noqa: E402
from core.scoring import network_metrics  # noqa: E402
from core.zoo import load_weights, networks  # noqa: E402

FOLDERS = ["attn-max2", "attn-min2", "attn-max4", "attn-min4"]


def load_their_cubic(checkout: Path) -> dict:
    """Their cubic.py, minus imports from gbmi and transformer_lens, with stand-ins for the names only
    their instruction-counting mode uses."""
    path = checkout / "gbmi" / "exp_max_of_n" / "verification" / "cubic.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("gbmi", "transformer_lens"))
    ]

    class _Unused:  # only used in isinstance checks for their FLOP-counting mode
        pass

    def count_sequences(sequence_length, nonmax_count, num_nonmax_tok_choices, nonmax_strict=False):
        # verbatim from gbmi/utils/sequences.py
        combinations = math.comb(sequence_length, nonmax_count)
        token_variations = num_nonmax_tok_choices**nonmax_count if num_nonmax_tok_choices > 0 else 0
        if nonmax_strict and num_nonmax_tok_choices > 1:
            token_variations -= (num_nonmax_tok_choices - 1) ** nonmax_count
        return combinations * token_variations

    ns = {
        "CountTensor": _Unused, "InstructionCount": _Unused, "PerfCollector": _Unused, "PerfCounter": _Unused,
        "count_sequences": count_sequences, "count_sequences_instructions": None, "EU_PU": None,
        "all_EQKE": None, "all_EQKP": None, "all_EVOU": None, "all_PVOU": None, "complexity_of": None,
        "HookedTransformer": object, "__name__": "gross_cubic",
    }
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def paths(w: dict, reverse: bool) -> dict:
    t = {k: torch.tensor(np.asarray(v), dtype=torch.float64) for k, v in w.items()}
    E, P, WQ, WK, WV, WO, U = (t[k] for k in ("E", "P", "WQ", "WK", "WV", "WO", "U"))
    if reverse:
        E, U = E.flip(0), U.flip(1)
    EP = E + P[-1]
    QK = WQ @ WK.T
    VOU = WV @ WO @ U
    return {
        "EUPU": EP @ U,
        "EVOU": E @ VOU,
        "PVOU": P @ VOU,
        "EQKE": EP @ QK @ E.T,
        "EQKP": EP @ QK @ P.T,
        "attn_scale": math.sqrt(WQ.shape[1]),
    }


def length_in_our_units(V: int, n: int, d: int, dh: int, E_cost: int) -> int:
    """The work of their cubic proof under core/check/cost.py's conventions, loop by loop."""
    ops = E_cost
    # the five path matrices, cheapest order (as the contract rule would charge)
    ops += V * d  # E + P[-1]
    ops += d * dh * d  # WQ WK^T
    ops += V * d * d + V * d * V + V * d * n  # EQKE and EQKP from (E+P)(WQ WK^T)
    ops += d * dh * d + d * d * V  # WV WO U
    ops += V * d * V + n * d * V  # EVOU, PVOU
    ops += V * d * V  # EUPU
    # compute_extreme_softmaxed_right_attention_cubic_simple
    ops += V * (n - 1) * max(1, math.ceil(math.log2(max(2, n - 1))))  # sorting EQKP per query
    per_softmax = 2 * n + 6 * n + 2 * (n - 1)  # scale both rows, softmax (3 per entry), sums
    for m in range(V):
        for q in range(m + 1):
            ops += 1
            for k in range(m + 1):
                if k == m:
                    if q != m:
                        ops += (n - 1) + per_softmax
                    continue
                for c in range(n):
                    if q != m and c >= n - 1:
                        continue
                    ops += 2 * (n - 1) + per_softmax
    # compute_largest_wrong_logit_cubic
    for m in range(V):
        ops += n * V + 2 * V * V  # centre PVOU, EUPU, EVOU
        ops += (n - 1) * V + 4 * V  # pessimise PVOU; logits with only the max token
        ops += m * 14 * V  # query token plus copies of the max
        ops += V * (V - 1) + V * V + V * (V - 1)  # pessimise EUPU per query, EVOU + PVOU per key
        for k in range(m):
            for c in range(1, n):
                for q in range(m + 1):
                    if q != m and c >= n - 1:
                        continue
                    ops += 12  # three products per bound end, sums, maximum, add
    # count_correct_sequences_cubic
    for q in range(V):
        for m in range(V):
            for c in range(n):
                if q > m or (q != m and c >= n - 1):
                    continue
                ops += 1 if c == 0 else 2 * m + 6
    return ops


def main(checkout: str) -> None:
    cubic = load_their_cubic(Path(checkout))
    out = {}
    for name in FOLDERS:
        folder = load_model_folder(REPO / "models" / name)
        reverse = "min" in name
        for e in networks(folder):
            if e["split"] != "dev":
                continue
            s = folder.sizes(e["setting"])
            V, n, d, dh = s["d_vocab"], s["n_ctx"], s["d_model"], s["d_head"]
            m = paths(load_weights(folder, e["id"]), reverse)
            extreme = cubic["compute_extreme_softmaxed_right_attention_cubic_simple"](
                EQKE=m["EQKE"], EQKP=m["EQKP"], attn_scale=m["attn_scale"], pbar=_NoBar()
            )
            wrong = cubic["compute_largest_wrong_logit_cubic"](extreme, EUPU=m["EUPU"], EVOU=m["EVOU"], PVOU=m["PVOU"])
            certified = int(cubic["count_correct_sequences_cubic"](wrong))
            n_inputs = V**n
            if certified > e["real_correct"]:
                raise SystemExit(f"{e['id']}: their bound {certified} exceeds the true count {e['real_correct']}")
            L = length_in_our_units(V, n, d, dh, e["E"])
            points = [{"length": L, "certified": certified, "n_inputs": n_inputs}]
            if e["real_correct"] > certified:
                points.append({"length": e["B"], "certified": e["real_correct"], "n_inputs": n_inputs})
            met = network_metrics(points, e)
            out[e["id"]] = {
                "certified": certified, "n_inputs": n_inputs, "real_correct": e["real_correct"],
                "accuracy_bound": certified / n_inputs, "normalised": certified / e["real_correct"],
                "length": L, "B": e["B"], "E": e["E"], "B_over_length": e["B"] / L,
                "Q": met["Q"], "Q_grid": met["Q_grid"], "cost_of_finishing": met["cost_of_finishing"],
            }
            r = out[e["id"]]
            print(f"{e['id']}: certified {certified}/{n_inputs} ({r['normalised']:.4f} of correct), "
                  f"length {L:,} (B/{r['B_over_length']:.0f}), Q {r['Q']:.4f}, Q_grid {r['Q_grid']:.4f}, "
                  f"cost of finishing {r['cost_of_finishing']:.2f}", flush=True)
    (Path(__file__).parent / "cubic_results.json").write_text(json.dumps(out, indent=1))


class _NoBar:
    def update(self, n):
        pass


if __name__ == "__main__":
    main(sys.argv[1])
