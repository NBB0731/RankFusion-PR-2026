# Replication 5 validation-weight diagnostics

Descriptive diagnostics derived only from validation-set bootstrap AP. No test labels or test scores are read.

| Expert | Mean validation AP | Mean fusion weight | Median weight | Weight q10–q90 | Top weight runs / 100 |
|---|---:|---:|---:|---:|---:|
| RBF-SVM | 0.5829 | 0.216 | 0.183 | [0.014, 0.449] | 38 |
| Random Forest | 0.6140 | 0.227 | 0.173 | [0.054, 0.506] | 16 |
| Extra Trees | 0.6241 | 0.244 | 0.185 | [0.109, 0.459] | 30 |
| HistGradientBoosting | 0.4881 | 0.062 | 0.023 | [0.000, 0.167] | 1 |
| RRFEN fuzzy-only | 0.5150 | 0.116 | 0.098 | [0.000, 0.240] | 9 |
| RRFEN-Hybrid | 0.5799 | 0.135 | 0.138 | [0.013, 0.226] | 6 |

Mean combined fuzzy-expert weight: 0.251; mean effective number of experts: 4.24.

Descriptive validation-only diagnostics; do not treat as independent evidence of test performance or as a post-hoc method selection criterion.
