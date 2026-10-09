"""Cost model cost-v1: a proof's length is the work the checker does to check it.

Each operation on exact numbers (one multiply-add, addition, subtraction or
comparison) counts 1. Specifically:

- extract: converting the weights exactly, 1 per parameter (this is E, the
  minimum any proof pays).
- read: reading the proof file, 1 per number, string or boolean in it.
- tree: 1 per node of the split tree.
- contract: multiplying an a x b by a b x c matrix costs a*b*c; pushing a
  bias through costs one matrix-vector product.
- approximate: 2 per entry of the approximated matrix (difference and
  comparison), plus a*r*c to multiply out low-rank factors.
- single: running the network on one input. A linear map applied to a one-hot
  input costs its output size per position (a column lookup); applied to a
  vector, in*out; a bias, a sum or an addition costs its output size; the
  final comparison of the label's output with every other output costs
  (outputs - 1).
- interval: like single, but on bounds over a set of inputs. A linear map
  applied to a set of one-hot tokens costs 2*out per token in the set (minimum
  and maximum); applied to a box it costs 2*in*out (centre and radius). For
  the output, a linear map applied directly to one-hot sets costs
  2*outputs per token in each position's set; with an approximation it costs
  what the approximation's form needs (see the approximate rule); applied to a
  box it costs 3*in*outputs.
- symmetry: 1 per leaf, for counting the inputs it certifies.

E is the extract cost. B is the length of the brute-force proof, which checks
every input with the single rule and nothing else.
"""

from __future__ import annotations

from collections import defaultdict

VERSION = "cost-v1"

CATEGORIES = ("extract", "read", "tree", "contract", "approximate", "single", "interval", "symmetry")


class Counter:
    def __init__(self):
        self.by_rule: dict[str, int] = defaultdict(int)

    def add(self, category: str, n) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown cost category {category!r}")
        n = int(n)
        if n < 0:
            raise ValueError("costs are never negative")
        self.by_rule[category] += n

    @property
    def total(self) -> int:
        return int(sum(self.by_rule.values()))

    def as_dict(self) -> dict:
        return {k: int(v) for k, v in sorted(self.by_rule.items()) if v}


def count_scalars(obj, limit: int | None = None) -> int:
    """Numbers, strings and booleans in a JSON value (iterative; stops past limit)."""
    n = 0
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif x is not None:
            n += 1
            if limit is not None and n > limit:
                return n
    return n
