# Decisions

Design decisions of the open lane and its evaluation, copied here from the project's notes so that a reviewer
working from the repositories can read them. Each is dated; later decisions supersede earlier ones.

- [pilot-2-what-the-score-shows.md](pilot-2-what-the-score-shows.md) (22 October 2026): where pilot 2's
  Q_grid gain came from (fixed cost, input selection, the price list), what the metric rewards that is not
  understanding, the one loophole open on the tiny network, and how to read partial coverage.
- [frozen-transfer.md](frozen-transfer.md) (22 October 2026): the next evaluation. A submission is a method
  (size-parametric general layer plus a certificate recipe), scored by Q_exact on a fixed dense schedule of
  budgets, on held-out networks, with the frontier as the record and cost of finishing (exact fallback
  costs) as the tie-breaker; what is frozen; how held-out is enforced; what the claim is (the frozen method
  generalises) and is not (understanding); the cost model to freeze first (policy v4, then the matched
  baseline remeasured).
