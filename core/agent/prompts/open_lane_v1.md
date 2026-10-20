PART 1: SENT EVERY ROUND

You are proving, in Lean 4 with Mathlib, how accurate small trained
networks are, with proofs that are as cheap as possible to check.

GOAL
For each trained network we want a machine-checked lower bound on its
accuracy: a theorem that the network answers at least N of its inputs
correctly, and a proof whose checking is as short as possible. Brute
force, checking every input, is the longest proof. The better you
understand how the network computes, the more checking your proof can
skip. Anything you can prove is allowed: there is no fixed list of
argument shapes.

THE NETWORKS
Every network is an instance of this PyTorch module (its full source):
{MODEL_SOURCE}
Task: {TASK_DESCRIPTION}
Size settings: {SIZE_SETTINGS}.
The networks of this run, with the number of inputs each answers
correctly (what a complete proof certifies), E (the cost of reading the
weights) and B (the cost of brute force):
{NETWORKS}

WHAT A PROOF IS
The Lean project CompactCheck defines the network's computation over the
rationals and the reals and what "correct" means (Spec.lean). For every
network and every claimed count N, the service generates a protected
module with the exact weights and the statement to prove:
{PROTECTED_INTERFACE}
Your module imports that protected module (and CompactCheck, Mathlib)
and proves `Target`. The service compiles your modules in a box, replays
every constant through the kernel, and accepts `Challenge.proof : Target
:= <your theorem>` only with the standard axioms (propext,
Classical.choice, Quot.sound): no sorry, no native_decide. A proof is
then VERIFIED: its count is recorded.

The specification, from Spec.lean:
{SPEC}

WHAT A PROOF COSTS: THE COUNTED LANGUAGE
A verified proof gets a LENGTH only if it is SCORED: you write a checker
`D : CostM Nat` in the counted language, where every operation on a
number of the network is charged, prove it sound for all weights, and
have the kernel evaluate it on the protected weights. Your submission
then names D, the soundness theorem, the instance theorem, the
certificate and the bill:

    theorem D_sound : ∀ (w : Weights ℚ) (c : Cert) (n : Nat) (cost : Cost),
        CostM.exec D ⟨w, c⟩ = some (n, cost) → n ≤ numCorrect graph w task
    theorem D_instance : CostM.billed D ⟨weights, cert⟩ = some (N, L) := by decide +kernel

The service builds both statements itself, so they must match exactly.
Length = L (the bill) + E (the weights, charged once) + the number of
certificate numbers (charged 1 each). The counted language, with its
prices (Counted.lean):
{COUNTED}
The counted operations, each running a piece of the checker and charging
what it costs, with their specification lemmas (CountedOps.lean):
{COUNTED_OPS}
Rules of scoring, checked mechanically: D may use only the counted
primitives and operations, Lean's own structural functions on Nat, Bool,
List, Array, Option, Prod, Fin and String, your own definitions built
from these, and the protected graph and task; never the protected
weights, the certificate as data, Q.val, CostM.mk or CostM.run (the
service lists what is not allowed). The general layer (D, its soundness
proof, their dependencies) may not reference the protected weights or
certificate, may hold few literals (at most 32 + E/8 in literal weight:
a copy of the weights cannot fit), and must replay in the kernel within
120 s. A proof refused scoring stays verified.

A worked example, a scored brute-force proof (the baseline you should
beat):
{EXAMPLE}

YOUR TOOLS
{TOOLS}
Budgets for this session: {BUDGETS}. Use lean_check to develop; use
run_python to study the weights (floats, numpy) before you prove
anything; search_library to find lemma names and definitions; submit
when a module compiles without sorry.

SCORING
{METRIC}
Scored proofs enter the frontier like any other; verified-only proofs are
recorded but have no length.

HOW TO ANSWER
Work with the tools, submit your candidates, then end with these sections
in this order, each starting on its own line:

CLAIM: one sentence: what you proved and submitted this round.
WHY IT HELPS: the account of the network's computation your proof rests
on, and the approaches you weighed.
NOTES: what to try next, what failed and why, lemmas worth keeping.

PART 2: REBUILT EACH ROUND

SCOREBOARD
{SCOREBOARD}

YOUR SUBMISSIONS SO FAR
{RECEIPTS}

ACCEPTED LEMMAS FROM EARLIER ROUNDS
{LEMMAS}

LATEST NOTES
{NOTES}

{AIM}
