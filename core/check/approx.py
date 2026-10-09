"""Approximations of a contracted matrix, for the approximate rule.

A proof file may give a simpler matrix P for a contracted product W. The
checker computes eps = max |W - P| over all entries exactly; interval leaves
may then bound min over tokens t of (W[l,t] - W[j,t]) from below by
min over t of (P[l,t] - P[j,t]) - 2*eps, computed from P's form:

    dense         P written out in full
    low_rank      P = left @ right, left (outputs x r), right (r x inputs)
    row_segments  each row of P is constant on consecutive column ranges,
                  given as [start, end, value] triples covering every column
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .arith import Exact, ProofNumberError, parse_number
from .cost import Counter


class ApproximationError(ValueError):
    pass


def _matrix(rows, n_rows: int | None, n_cols: int | None, what: str) -> np.ndarray:
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        raise ApproximationError(f"{what} must be a list of rows")
    if n_rows is not None and len(rows) != n_rows:
        raise ApproximationError(f"{what} needs {n_rows} rows")
    width = len(rows[0])
    if n_cols is not None and width != n_cols:
        raise ApproximationError(f"{what} needs {n_cols} columns")
    if any(len(r) != width for r in rows):
        raise ApproximationError(f"{what} rows differ in length")
    try:
        return np.array([[parse_number(v) for v in r] for r in rows], dtype=object)
    except ProofNumberError as exc:
        raise ApproximationError(f"{what}: {exc}") from exc


class Approximation:
    def __init__(self, name: str, of: str, spec: dict, W: Exact, counter: Counter):
        self.name, self.of = name, of
        self.form = spec.get("form")
        n_out, n_in = W.shape
        self.n_out, self.n_in = n_out, n_in
        Wf = W.to_fractions()
        if self.form == "dense":
            self.P = _matrix(spec.get("values"), n_out, n_in, "values")
            self.eps = max(abs(x) for x in (Wf - self.P).reshape(-1))
            counter.add("approximate", 2 * n_out * n_in)
        elif self.form == "low_rank":
            self.F = _matrix(spec.get("left"), n_out, None, "left")
            r = self.F.shape[1]
            self.G = _matrix(spec.get("right"), r, n_in, "right")
            P = np.dot(self.F, self.G)
            counter.add("approximate", n_out * r * n_in)
            self.eps = max(abs(x) for x in (Wf - P).reshape(-1))
            counter.add("approximate", 2 * n_out * n_in)
        elif self.form == "row_segments":
            rows = spec.get("rows")
            if not isinstance(rows, list) or len(rows) != n_out:
                raise ApproximationError(f"rows needs one list of segments per output ({n_out})")
            self.seg = []
            eps = Fraction(0)
            for i, segs in enumerate(rows):
                if not isinstance(segs, list) or not segs:
                    raise ApproximationError(f"row {i} has no segments")
                starts, ends, vals = [], [], []
                expect = 0
                for s in segs:
                    if not (isinstance(s, list) and len(s) == 3 and all(isinstance(v, int) and not isinstance(v, bool) for v in s[:2])):
                        raise ApproximationError(f"row {i}: segments are [start, end, value]")
                    a, b = s[0], s[1]
                    if a != expect or b < a or b >= n_in:
                        raise ApproximationError(f"row {i}: segments must cover the columns in order")
                    try:
                        v = parse_number(s[2])
                    except ProofNumberError as exc:
                        raise ApproximationError(f"row {i}: {exc}") from exc
                    seg_w = W.ints[i, a : b + 1]
                    hi = Fraction(int(seg_w.max())) * Fraction(2) ** W.exp
                    lo = Fraction(int(seg_w.min())) * Fraction(2) ** W.exp
                    eps = max(eps, abs(hi - v), abs(lo - v))
                    starts.append(a)
                    ends.append(b)
                    vals.append(v)
                    expect = b + 1
                if expect != n_in:
                    raise ApproximationError(f"row {i}: segments must reach the last column")
                self.seg.append((np.array(starts), np.array(ends), vals))
            self.eps = eps
            counter.add("approximate", 2 * n_out * n_in)
        else:
            raise ApproximationError(f"unknown form {self.form!r}; use dense, low_rank or row_segments")

    def _clip(self, i: int, lo: int, hi: int):
        starts, ends, vals = self.seg[i]
        k0 = int(np.searchsorted(ends, lo, side="left"))
        k1 = int(np.searchsorted(starts, hi, side="right"))
        return [(max(int(starts[k]), lo), min(int(ends[k]), hi), vals[k]) for k in range(k0, k1)]

    def row_min_diffs(self, l: int, lo: int, hi: int, counter: Counter) -> np.ndarray:
        """For every output j: a lower bound on min over t in [lo, hi] of P[l, t] - P[j, t]."""
        s = hi - lo + 1
        if self.form == "dense":
            d = self.P[l, lo : hi + 1] - self.P[:, lo : hi + 1]
            counter.add("interval", 2 * self.n_out * s)
            return d.min(axis=1)
        if self.form == "low_rank":
            r = self.G.shape[0]
            gmin = self.G[:, lo : hi + 1].min(axis=1)
            gmax = self.G[:, lo : hi + 1].max(axis=1)
            c = self.F[l] - self.F
            out = np.empty(self.n_out, dtype=object)
            for j in range(self.n_out):
                out[j] = sum((ck * (gmin[k] if ck >= 0 else gmax[k]) for k, ck in enumerate(c[j])), Fraction(0))
            counter.add("interval", 2 * r * s + 2 * self.n_out * r)
            return out
        # row_segments: merge the two rows' segments over [lo, hi]
        mine = self._clip(l, lo, hi)
        out = np.empty(self.n_out, dtype=object)
        ops = 0
        for j in range(self.n_out):
            theirs = self._clip(j, lo, hi)
            a = b = 0
            best = None
            while a < len(mine) and b < len(theirs):
                d = mine[a][2] - theirs[b][2]
                best = d if best is None or d < best else best
                ops += 1
                if mine[a][1] < theirs[b][1]:
                    a += 1
                elif mine[a][1] > theirs[b][1]:
                    b += 1
                else:
                    a += 1
                    b += 1
            out[j] = best
        counter.add("interval", ops + self.n_out)
        return out
