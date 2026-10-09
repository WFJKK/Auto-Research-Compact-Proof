"""Frontiers and metrics, computed from archive records alone (section 5 of the spec).

Per network, a(b) is the best certified accuracy among checked proofs of
length at most b. The metrics:

- Q: the area under a(b) on a log length scale from E to B, divided by ln(B/E).
- shortest_full: the shortest proof that certifies the network's real accuracy.
- within: the best certified accuracy within B/10, B/100 and B/1000.
- cost_of_finishing: B divided by (length + uncertified correct inputs x B/n),
  for the best proof.

Per size setting, the median and the worst network. All numbers are
Python-checked until rules have Lean proofs.
"""

from __future__ import annotations

import math
import statistics

BUDGET_DIVISORS = (10, 100, 1000)


def scored(records: list[dict]) -> list[dict]:
    """Records that count: checked proofs the checker accepted."""
    return [
        r
        for r in records
        if r.get("status") == "ok" and r.get("network") and r.get("length") is not None and r.get("certified") is not None
    ]


def frontier(records: list[dict]) -> dict[str, list[dict]]:
    """Per network, the proofs no other proof beats on both length and certified inputs, by increasing length.

    A proof that certifies nothing is never on it.
    """
    by_net: dict[str, list[dict]] = {}
    for r in scored(records):
        if r["certified"] > 0:
            by_net.setdefault(r["network"], []).append(r)
    out = {}
    for net, rs in by_net.items():
        rs.sort(key=lambda r: (r["length"], -r["certified"], r["record_id"]))
        points, best = [], -1
        for r in rs:
            if r["certified"] > best:
                points.append(
                    {
                        "length": r["length"],
                        "certified": r["certified"],
                        "n_inputs": r["n_inputs"],
                        "record_id": r["record_id"],
                        "round": r["round"],
                        "attempt": r["attempt"],
                        "knob": r["knob"],
                    }
                )
                best = r["certified"]
        out[net] = points
    return out


def accuracy_within(points: list[dict], budget: float) -> float:
    best = 0.0
    for p in points:
        if p["length"] <= budget:
            best = max(best, p["certified"] / p["n_inputs"])
    return best


def q_score(points: list[dict], E: float, B: float) -> float:
    if not E or not B or B <= E:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        lo = max(p["length"], E)
        hi = min(points[i + 1]["length"], B) if i + 1 < len(points) else B
        if hi > lo:
            total += p["certified"] / p["n_inputs"] * (math.log(hi) - math.log(lo))
    return total / math.log(B / E)


def cost_of_finishing(points: list[dict], real_correct: int, B: float) -> float:
    """B over the cheapest total of proof length plus brute force on the correct inputs it leaves uncertified."""
    if not points or not real_correct:
        return 0.0
    n = points[0]["n_inputs"]
    best = None
    for p in points:
        total = p["length"] + max(0, real_correct - p["certified"]) * B / n
        best = total if best is None else min(best, total)
    return B / best


def network_metrics(points: list[dict], entry: dict) -> dict:
    E, B, real = entry.get("E"), entry.get("B"), entry.get("real_correct")
    full = [p["length"] for p in points if real is not None and p["certified"] >= real]
    return {
        "Q": q_score(points, E, B),
        "best_accuracy": max((p["certified"] / p["n_inputs"] for p in points), default=0.0),
        "real_accuracy": entry.get("real_accuracy"),
        "shortest_full": min(full) if full else None,
        "within": {f"B/{d}": accuracy_within(points, B / d) for d in BUDGET_DIVISORS} if B else {},
        "cost_of_finishing": cost_of_finishing(points, real, B) if B else 0.0,
        "E": E,
        "B": B,
        "points": len(points),
    }


def summarize(records: list[dict], entries: list[dict]) -> dict:
    """Per size setting: metrics per network, then the median and the worst network."""
    fr = frontier(records)
    by_setting: dict[str, dict] = {}
    for e in entries:
        by_setting.setdefault(e["setting"], {})[e["id"]] = network_metrics(fr.get(e["id"], []), e)
    out = {}
    for setting, nets in by_setting.items():
        q = [m["Q"] for m in nets.values()]
        cof = [m["cost_of_finishing"] for m in nets.values()]
        full = [m["shortest_full"] / m["B"] if m["shortest_full"] and m["B"] else None for m in nets.values()]
        out[setting] = {
            "networks": nets,
            "Q_median": statistics.median(q),
            "Q_worst": min(q),
            "cost_of_finishing_median": statistics.median(cof),
            "cost_of_finishing_worst": min(cof),
            "shortest_full_over_B_median": statistics.median(full) if all(f is not None for f in full) else None,
        }
    return out
