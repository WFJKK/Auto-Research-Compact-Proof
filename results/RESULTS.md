# Results

One line per run. Raw run data stays outside the repository. Write-ups that discuss several runs: [max2, 13 October 2026](max2-2026-10-13/README.md).

| Run | Model | Commit | Backend | Development (median Q) |
| --- | --- | --- | --- | --- |
| [max2-rerun-q](max2-rerun-q/report.md) | models/max2 | 4e77876816 | claude_code | v64 Q 0.3933 (worst 0.3882); v128 Q 0.4036 (worst 0.4005) |
| [max2-rerun-cof](max2-rerun-cof/report.md) | models/max2 | 4e77876816 | claude_code | v64 Q 0.3250 (worst 0.3229); v128 Q 0.3477 (worst 0.3449) |
| [attn-max2-q-api-3](frontiers-2026-10-19/README.md) | models/attn-max2 | 3c26106e1c (not on GitHub) | api | v64 Q 0.1502 (worst 0.1447); v128 Q 0.1497 (worst 0.1466) |
| [attn-max2-q-api-tokens](frontiers-2026-10-19/README.md) | models/attn-max2 | 433de60f1f (not on GitHub) | api | v64 Q 0.3286 (worst 0.3286); v128 Q 0.2845 (worst 0.2821) |
| [attn-min2-q-api-tokens](frontiers-2026-10-19/README.md) | models/attn-min2 | 02642356cc (not on GitHub) | api | v64 Q 0.3316 (worst 0.3316); v128 Q 0.2687 (worst 0.2679) |
| [attn-max4-q-api-tokens](frontiers-2026-10-19/README.md) | models/attn-max4 | 02642356cc (not on GitHub) | api | v16 Q 0.4811 (worst 0.4702); v24 Q 0.4813 (worst 0.4774) |
| [attn-min4-q-api-tokens](frontiers-2026-10-19/README.md) | models/attn-min4 | 02642356cc (not on GitHub) | api | v16 Q 0.4762 (worst 0.4528); v24 Q 0.4951 (worst 0.4841) |
| [open-lane-pilot-tiny-3](open-lane-pilot-tiny-3/report.md) | tests/max2_tiny | 6608098b97 (a local merge of f929cfa, on the Mac) | claude_code | v8 Q 0.5101 (worst 0.5101); Q_grid 0.5375; full proof B/6.0 |
