# Held-out evaluation of max2-rerun-q

10 frontier recipes, checked by Python.

## v64

| Metric | Development | Held out |
| --- | --- | --- |
| Q_median | 0.3933 | 0.3957 |
| Q_worst | 0.3882 | 0.3905 |
| cost_of_finishing_median | 9.1190 | 9.2206 |
| cost_of_finishing_worst | 9.0250 | 9.0402 |
| shortest_full_over_B_median | 0.1097 | 0.1085 |

| Held-out network | Q | Best certified accuracy | Real accuracy |
| --- | --- | --- | --- |
| max2-v64-s2 | 0.3997 | 0.99902 | 0.99902 |
| max2-v64-s6 | 0.3998 | 0.99927 | 0.99927 |
| max2-v64-s7 | 0.3905 | 0.99902 | 0.99902 |
| max2-v64-s9 | 0.3918 | 0.99976 | 0.99976 |

## v128

| Metric | Development | Held out |
| --- | --- | --- |
| Q_median | 0.4036 | 0.4041 |
| Q_worst | 0.4005 | 0.3984 |
| cost_of_finishing_median | 19.1301 | 19.2926 |
| cost_of_finishing_worst | 18.9669 | 18.9382 |
| shortest_full_over_B_median | 0.0523 | 0.0518 |

| Held-out network | Q | Best certified accuracy | Real accuracy |
| --- | --- | --- | --- |
| max2-v128-s2 | 0.3984 | 0.99982 | 0.99982 |
| max2-v128-s3 | 0.4056 | 0.99994 | 0.99994 |
| max2-v128-s4 | 0.4044 | 0.99994 | 0.99994 |
| max2-v128-s9 | 0.4038 | 0.99994 | 0.99994 |
