# Soundness of the generic rules

Every rule here is generic: it works for any model folder whose `model.py` uses the supported operations (linear maps, sums over one dimension, additions) and whose inputs are one-hot tokens. The rules' short descriptions, shown to the agent, are in `core/check/rules.py`. None of these arguments has a Lean proof yet, so every result is labelled Python-checked.

**What is certified.** The checker certifies the network as a function over the rationals, defined by its float32 weights converted exactly. A float32 forward pass can disagree only on inputs whose margin is within rounding error. On all 20 max2 networks, exact brute force and float32 agree on every input.

**Arithmetic.** Float32 weights are dyadic rationals and convert to integers times a power of two with no rounding (`Exact.from_float`). Every later operation is an integer operation on those, and numbers in proof files are parsed as fractions. Nothing is ever rounded.

## Split tree

A split divides the current range [lo, hi] of one position at strictly increasing cuts inside (lo, hi]. The pieces [lo, c1−1], ..., [ck, hi] are disjoint and cover [lo, hi], and the number of children must equal the number of pieces. By induction from the root, which covers every input, the leaves partition the input space. So summing leaf sizes never counts an input twice. The walk is iterative, with node and size limits, so a hostile tree cannot exhaust the stack.

## single

The leaf must hold exactly one input. The checker evaluates the network exactly and certifies the input if the label's logit is strictly larger than every other logit. Ties are never certified.

## interval

The task's `constant_label` must confirm that every input in the leaf has the same label l.

- **One-hot sets.** For position p with tokens in [lo, hi], a linear map W applied to the one-hot vector gives a column W[:, t] for some t in the range. For each coordinate, min over t and max over t bound that column. Summed over positions, the centre and radius of those bounds give a box containing the summed vector for every input in the leaf.
- **Boxes through linear maps.** If x lies in the box with centre c and radius r, then W x + b lies in the box with centre W c + b and radius |W| r. This is standard interval arithmetic, computed exactly. Sums and additions of boxes add centres and radii.
- **Margins at a linear output.** The margin for output j is (W_l − W_j)·x + (b_l − b_j), which is at least (W_l − W_j)·c − |W_l − W_j|·r + (b_l − b_j) for x in the box.
- **Margins when the output map sees one-hot sets directly**, as after a full contraction. The margin is the sum over positions of W[l, t_p] − W[j, t_p], plus the bias difference. It is therefore at least the sum over positions of min over t in the range of W[l, t] − W[j, t]. This is exact when all but one position is a single token.
- **Without a linear output map**, the margin is at least (c_l − r_l) − (c_j + r_j).

The leaf is certified only if every lower bound for j ≠ l is positive. So every input in it has a positive margin, and the network classifies it correctly.

## contract

Matrix multiplication is associative, so replacing consecutive linear maps x ↦ W_k(⋯(W_1 x + b_1)⋯) + b_k by the single map with weight W_k⋯W_1 and bias W_k⋯W_2 b_1 + ⋯ + b_k computes the same function. The chain must be consecutive, and every intermediate result must feed only the next map, so nothing else in the graph reads the values that disappear. The product is computed exactly.

## symmetry

Two conditions are checked, and both must hold.

- **The network ignores order.** The traced graph shows that the input is used only by a sum over the positions dimension. Then the output is g of the sum of the per-position vectors, for some g, so it is the same for every reordering of the positions.
- **The label ignores order.** The task declares `LABEL_SYMMETRIC = True`. The model-folder loader tests the declaration on random inputs, and the declaration itself is trusted model code.

Then an input is correct exactly when its sorted version is. Under symmetry, a leaf counts only its inputs with tokens in non-decreasing order, each weighted by its number of distinct reorderings. Every sorted input lies in exactly one leaf, because the leaves partition the inputs. Every input is a reordering of exactly one sorted input. So no input is counted twice, and the total never exceeds the number of inputs. A test compares the weighted count with enumeration for 1, 2 and 3 positions. The certified mask checks it again on every proof.

## approximate

For a linear map with weight W, the proof supplies P. The checker computes eps = max over all entries of |W − P|, exactly. Then W[l, t] − W[j, t] ≥ P[l, t] − P[j, t] − 2 eps for every t, so for any range the minimum over t of the left side is at least the minimum of P[l, t] − P[j, t], minus 2 eps. The forms differ only in how that minimum is bounded.

- **dense:** computed directly.
- **row_segments:** rows l and j are constant on known pieces, so the minimum over a range is the minimum over the overlapping pieces. It is computed by merging the two rows' segments, exactly.
- **low_rank** (P = F G): P[l, t] − P[j, t] = Σ_k c_k G[k, t] with c = F_l − F_j. For every t this is at least Σ_k c_k · (G_min[k] if c_k ≥ 0 else G_max[k]), where G_min and G_max are taken over the range.

All three are tested against exact brute force on random leaves and random approximations, including poor ones.

## skip

It certifies nothing.
