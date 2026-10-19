Reproducibility archive for the Pattern Recognition submission

This updated archive is synchronized to the selected manuscript version reporting 19 OddBench dataset families and macro AP 0.6343. It contains the frozen Replication 5 protocol and model code, retained-family prediction files, the exact 19-family reanalysis tables, and the retrospective matched-split comparator extension against an adapted FairComb procedure and WSAD-DT. All reported inferential units are dataset families after averaging five seeds.

Pilot exclusion and analysis boundary

The original metadata-frozen selection contained 20 families. LandParcelValidation (seed 1201) was used as a code-feasibility pilot; after outcomes had been inspected, the entire family was excluded from final inference. No LandParcelValidation prediction files are included in this final supplementary archive. The original 20-family aggregate is retained only under `historical_20_family/` to make the transition auditable; it is not the result reported in the manuscript. The final analysis is retrospective and must not be described as independent prospective confirmation.

Contents and reproduction

- `replication5/`: original frozen protocol, selection manifests, integrity audits, and environment record.
- `predictions/`: frozen internal-expert predictions for the 19 retained families; the source OddBench data are not bundled and remain governed by their original terms.
- `historical_20_family/`: original 20-family aggregate retained solely as audit history.
- `recent_comparator_extension_v2/`: 19-family, five-seed score files and per-run records for the candidate, FairComb adaptation, and WSAD-DT. The pilot directory is excluded.
- `final_reanalysis/FINAL_19_FAMILY_RESULTS.json`: exact 19-family metrics and comparison results.
- `code/architecture_v4/`: fitting/scoring scripts and a reaggregation script.
- `comparator_sources/`: the relevant WSAD-DT source snapshot and FairComb source routines used to document the adaptation.
- `SHA256SUMS.json`: SHA-256 hashes for every other file in this archive.

To reproduce

The frozen per-run predictions permit regeneration of all reported aggregate AP tables without refitting. From the extracted archive, install the packages listed in `replication5/environment.json` and run `python code/architecture_v4/reaggregate_19_family_from_frozen_summary.py` to regenerate the 19-family internal ablation summary from the original family-level results while excluding the pilot. The matched comparator results are stored with per-run JSON and prediction arrays; their aggregate summary is in `recent_comparator_extension_v2/summary.json`. To rerun model fitting, obtain the public MacrOData-CMU/OddBench NPZ files under their original terms, place them as described by the included runners, and use the frozen protocol.

All experiments used CPU; no GPU was used. FairComb is an adaptation of the public author procedure to the shared expert pool, not an exact reproduction of the original R package. WSAD-DT was run using the official implementation/default configuration recorded in the protocol. The direct comparisons were conducted after candidate outcomes had been examined and are retrospective.
