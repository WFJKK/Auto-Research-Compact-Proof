# Held-out evaluation of max2-rerun-cof

3 frontier recipes, checked by Python.

## v64

| Metric | Development | Held out |
| --- | --- | --- |
| Q_median | 0.3250 | 0.3262 |
| Q_worst | 0.3229 | 0.3244 |
| cost_of_finishing_median | 9.1450 | 9.2191 |
| cost_of_finishing_worst | 8.9704 | 9.0665 |
| shortest_full_over_B_median | 0.1094 | 0.1085 |

| Held-out network | Q | Best certified accuracy | Real accuracy |
| --- | --- | --- | --- |
| max2-v64-s2 | 0.3269 | 0.99902 | 0.99902 |
| max2-v64-s6 | 0.3276 | 0.99927 | 0.99927 |
| max2-v64-s7 | 0.3254 | 0.99902 | 0.99902 |
| max2-v64-s9 | 0.3244 | 0.99976 | 0.99976 |

## v128

| Metric | Development | Held out |
| --- | --- | --- |
| Q_median | 0.3477 | 0.3476 |
| Q_worst | 0.3449 | 0.3468 |
| cost_of_finishing_median | 19.0799 | 19.0892 |
| cost_of_finishing_worst | 18.7714 | 18.8459 |
| shortest_full_over_B_median | 0.0524 | 0.0524 |

| Held-out network | Q | Best certified accuracy | Real accuracy |
| --- | --- | --- | --- |
| max2-v128-s2 | 0.3468 | 0.99982 | 0.99982 |
| max2-v128-s3 | 0.3500 | 0.99994 | 0.99994 |
| max2-v128-s4 | 0.3480 | 0.99994 | 0.99994 |
| max2-v128-s9 | 0.3471 | 0.99994 | 0.99994 |
