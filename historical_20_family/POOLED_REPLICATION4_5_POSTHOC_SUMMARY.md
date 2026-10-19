# Post-hoc pooled Replications 4–5 (40 families)

This combines the two fresh, family-disjoint cohorts evaluated with the same six-expert validation-AP-weighted percentile-rank fusion protocol. Five seeds are averaged within family. Replication 5 remains the prospective independent confirmation; this pooled analysis is supportive and post hoc.

| Comparator | Mean Δ AP | 95% stratified family-bootstrap CI | Wins/ties/losses | Holm p (6 contrasts) | Rep4 Δ | Rep5 Δ |
|---|---:|---:|---:|---:|---:|---:|
| Extra Trees | +0.0211 | [+0.0133, +0.0288] | 28/4/8 | 0.00042477 | +0.0138 | +0.0284 |
| Six-expert validation selector | +0.0107 | [+0.0059, +0.0162] | 26/4/10 | 0.00059248 | +0.0077 | +0.0137 |
| Expanded eight-expert validation selector | +0.0091 | [+0.0031, +0.0154] | 25/4/11 | 0.0055124 | +0.0044 | +0.0138 |
| Bootstrap-penalized rank fusion | +0.0009 | [-0.0002, +0.0023] | 17/4/19 | 0.76862 | +0.0005 | +0.0012 |
| Uniform rank fusion | +0.0360 | [+0.0177, +0.0596] | 34/2/4 | 1.2235e-05 | +0.0408 | +0.0311 |
| Four-classical-expert rank fusion | +0.0083 | [+0.0034, +0.0143] | 24/4/12 | 0.0085036 | +0.0082 | +0.0084 |

The inferential unit is the family; seed runs are not counted as independent observations. Bootstrap resampling was stratified by cohort. Holm correction covers the six listed paired contrasts.
