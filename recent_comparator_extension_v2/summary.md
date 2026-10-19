# Recent-comparator extension (post-hoc matched-split analysis)

Families: 19; seeds per family: 5; support: 50 labeled normals + 10 labeled anomalies.

This is a retrospective comparator extension: the candidate's Replication 5 outcomes were already examined. LandParcelValidation was used only as a code-feasibility pilot and excluded in full from inference.

| Method | Macro AP | Family SD | 95% family bootstrap CI |
|---|---:|---:|---:|
| Proposed AP-weighted rank fusion | 0.6343 | 0.3099 | [0.4969, 0.7661] |
| FairComb (adapted author procedure) | 0.5787 | 0.3140 | [0.4418, 0.7140] |
| WSAD-DT (official defaults) | 0.5550 | 0.3271 | [0.4128, 0.6969] |

| Paired comparison | Mean AP difference | 95% paired CI | Wins/ties/losses | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| AP-weighted rank fusion vs FairComb_test_AP | +0.0556 | [+0.0262, +0.0917] | 17/2/0 | 0.000293053 | 0.000586105 |
| AP-weighted rank fusion vs WSAD_DT_test_AP | +0.0792 | [+0.0298, +0.1505] | 15/2/2 | 0.000599119 | 0.000599119 |

Friedman omnibus across the three methods: chi-square=19.8824, p=4.81506e-05.
WSAD-DT mean runtime per family (averaged across seeds): 268.9s; median 130.1s.

The results are same-split and test-label-blind during method fitting/selection, but remain post-hoc relative to the primary study. They support only the tested tabular benchmark protocol.
