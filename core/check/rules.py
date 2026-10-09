"""The generic rules. These descriptions are shown to the agent verbatim.

Leaf rules apply to one piece of the split tree; global rules are top-level
keys of the proof file. A model folder may add rules of its own in rules.py.
"""

RULES = {
    "split": (
        'Tree node. {"split": {"position": p, "cuts": [c1, ..., ck]}, "children": [...]} divides the '
        "current token range [lo, hi] of position p into [lo, c1-1], [c1, c2-1], ..., [ck, hi]. Cuts "
        "must increase strictly with lo < c1 and ck <= hi; there is one child per piece, in order. The "
        "root covers every input, so the leaves always partition the inputs."
    ),
    "single": (
        'Leaf rule {"leaf": "single"}. The leaf must hold exactly one input. The checker runs the '
        "network on it exactly; it is certified if the label's output is strictly larger than every "
        "other output."
    ),
    "interval": (
        'Leaf rule {"leaf": "interval"}. The correct output must be the same for every input in the '
        "leaf (the task checks this). The checker bounds the network over the whole leaf exactly: "
        "one-hot inputs stay symbolic until the first linear map, which sees the exact columns of the "
        "allowed tokens (per coordinate, the minimum and maximum over them); later linear maps act on "
        "boxes (centre and radius). If the last linear map is applied directly to the one-hot inputs "
        "(for example after a contraction), the bound is exact per position: the minimum over that "
        "position's tokens of the label's row minus every other row. The leaf is certified if every "
        "margin's lower bound is positive. With \"approx\": \"<name>\", ranges of more than one token use "
        "that approximation instead of the dense minimum."
    ),
    "skip": 'Leaf rule {"leaf": "skip"}. Certifies nothing and costs almost nothing.',
    "contract": (
        'Global rule. "contract": [{"name": N, "chain": [a, b, ...]}] replaces consecutive linear maps '
        "a, then b, ... (each feeding only the next) by their exact product, computed once in the "
        "cheapest order and charged once. Later leaves use the product. Names are the network graph's "
        "node names (for an nn.Linear submodule, its attribute name)."
    ),
    "symmetry": (
        'Global rule. "symmetry": true is accepted only if the network uses its input only through a '
        "sum over positions, so its output is the same for every reordering of the positions, and the "
        "task declares that reordering never changes the correct output. Then "
        "each accepted leaf certifies the inputs in it whose tokens are in non-decreasing order, "
        "together with all their reorderings. Inputs in a leaf that are not in order count nothing "
        "there; they are certified through their sorted version wherever that lies."
    ),
    "approximate": (
        'Global rule. "approximate": [{"name": P, "of": M, "form": ...}] gives a simpler matrix P for '
        "linear map M (usually a contraction), in one of three forms: dense (\"values\": the full "
        'matrix), low_rank ("left": outputs x r, "right": r x inputs) or row_segments ("rows": for '
        "each output row, [start, end, value] triples covering every column in order). The checker "
        "computes eps = max |M - P| exactly. Interval leaves with \"approx\": P then bound the minimum "
        "over a position's tokens t of M[l,t] - M[j,t] by the same minimum for P, minus 2 eps, "
        "computed from P's form: row_segments costs about one step per segment touched, low_rank about "
        "r per output."
    ),
}

GENERIC_RULES = ("single", "interval", "skip", "contract", "symmetry", "approximate")
LEAF_RULES = ("single", "interval", "skip")
NUMBERS = (
    "Numbers in proof files are integers, strings such as \"-3/4\" or \"1.25\", or finite floats; all are "
    "read exactly."
)


def describe(rule_names=GENERIC_RULES) -> str:
    lines = [f"- split: {RULES['split']}"]
    lines += [f"- {name}: {RULES[name]}" for name in rule_names if name in RULES]
    lines.append(NUMBERS)
    return "\n".join(lines)
