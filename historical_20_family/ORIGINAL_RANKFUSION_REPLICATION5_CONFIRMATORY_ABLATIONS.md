# Replication 5: frozen AP-weighted rank-fusion confirmation

Protocol SHA-256: `c002b0b64f83a52fff24491db8b4d1cb64b316faa64bc88b5c08fbed58301484`

20 previously unused family-disjoint datasets; five seeds per family; family is the inference unit. AP is primary. All variants and contrasts were frozen before test labels were scored.

## Macro performance

| Method / variant | Macro AP | Between-family SD | 95% family-bootstrap CI | Macro AUROC |
|---|---:|---:|---:|---:|
| rank_uniform_6 | 0.6112 | 0.3059 | [0.4787, 0.7408] | 0.8755 |
| rank_meanAP_6 | 0.6423 | 0.3038 | [0.5108, 0.7675] | 0.8773 |
| rank_bootstrap_6 | 0.6411 | 0.3060 | [0.5059, 0.7671] | 0.8806 |
| rank_meanAP_4_no_fuzzy | 0.6339 | 0.3131 | [0.4965, 0.7629] | 0.8758 |
| zscore_uniform_6 | 0.6279 | 0.3035 | [0.4955, 0.7525] | 0.8772 |
| zscore_meanAP_6 | 0.6425 | 0.3025 | [0.5107, 0.7676] | 0.8772 |
| zscore_bootstrap_6 | 0.6401 | 0.3055 | [0.5045, 0.7679] | 0.8803 |
| rank_bootstrap_4_no_fuzzy | 0.6339 | 0.3128 | [0.4956, 0.7647] | 0.8786 |
| Extra Trees | 0.6139 | 0.3101 | [0.4775, 0.7429] | 0.8657 |
| PReNet | 0.5231 | 0.3384 | [0.3778, 0.6681] | 0.7497 |
| DevNet | 0.5039 | 0.3396 | [0.3611, 0.6496] | 0.8078 |
| six_expert_validation_selector | 0.6286 | 0.3151 | [0.4912, 0.7578] | 0.8626 |
| expanded_8_expert_validation_selector | 0.6285 | 0.3157 | [0.4911, 0.7584] | 0.8592 |

## Pre-specified paired contrasts

Δ AP is six-expert validation-AP-weighted percentile-rank fusion minus the comparator. Holm correction covers all six listed contrasts.

| Comparator | Mean Δ AP | 95% paired family-bootstrap CI | Wins/ties/losses | Holm p |
|---|---:|---:|---:|---:|
| Extra Trees | +0.0284 | [+0.0191, +0.0380] | 16/2/2 | 0.0019621 |
| six_expert_validation_selector | +0.0137 | [+0.0071, +0.0211] | 14/2/4 | 0.0068487 |
| expanded_8_expert_validation_selector | +0.0138 | [+0.0063, +0.0218] | 14/2/4 | 0.0098879 |
| rank_bootstrap_6 | +0.0012 | [-0.0002, +0.0037] | 8/2/10 | 0.84463 |
| rank_uniform_6 | +0.0311 | [+0.0103, +0.0577] | 16/2/2 | 0.0098879 |
| rank_meanAP_4_no_fuzzy | +0.0084 | [+0.0016, +0.0172] | 13/2/5 | 0.044462 |

## Limits

Interpret all outcomes according to the frozen protocol. The inferential unit is the dataset family; five seeds are averaged within family. This single confirmation cohort cannot on its own establish broad superiority.
