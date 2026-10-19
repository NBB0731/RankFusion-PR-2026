"""Family-level summary and inference for the frozen recent-comparator extension."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

HERE=Path(__file__).resolve().parent
ROOT=HERE/"macrodata_confirmatory"
P=json.loads((ROOT/"RECENT_COMPARATOR_EXTENSION_PROTOCOL.json").read_text(encoding="utf-8"))
PH=hashlib.sha256((ROOT/"RECENT_COMPARATOR_EXTENSION_PROTOCOL.json").read_bytes()).hexdigest()
OUT=ROOT/"recent_comparator_extension_v2"
FAMILIES=P["datasets"];SEEDS=P["seeds"]
METHODS=["candidate_test_AP","FairComb_test_AP","WSAD_DT_test_AP"]

def boot_ci(x,seed=70019,B=50000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    a=rng.choice(x,size=(B,len(x)),replace=True).mean(axis=1)
    return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]
def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));prev=0.
    for rank,i in enumerate(order):
        prev=max(prev,min(1.,(len(ps)-rank)*float(ps[i])));out[i]=prev
    return out.tolist()

records={}
for family in FAMILIES:
    vals={m:[] for m in METHODS};times=[]
    for seed in SEEDS:
        path=OUT/family/str(seed)/"result.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing expected run: {path}")
        r=json.loads(path.read_text(encoding="utf-8"))
        if r.get("protocol_sha256")!=PH or r.get("status")!="complete":
            raise RuntimeError(f"Protocol/status mismatch in {path}")
        for m in METHODS: vals[m].append(float(r["metrics"][m]))
        times.append(float(r["wsad_dt"]["fit_and_score_seconds"]))
    records[family]={m:float(np.mean(vals[m])) for m in METHODS}
    records[family]["WSAD_DT_seconds_mean"]=float(np.mean(times))

means={m:np.array([records[f][m] for f in FAMILIES],float) for m in METHODS}
summary={}
for m,x in means.items():
    summary[m]={"macro_AP":float(x.mean()),"family_sd":float(x.std(ddof=1)),
                "family_bootstrap_95CI":boot_ci(x),"per_family_AP":x.tolist()}
contrasts=[]
for cmp_name in ["FairComb_test_AP","WSAD_DT_test_AP"]:
    d=means["candidate_test_AP"]-means[cmp_name]
    p=float(wilcoxon(d,alternative="two-sided",zero_method="wilcox",method="auto").pvalue) if np.any(d) else 1.
    contrasts.append({"comparison":"AP-weighted rank fusion vs "+cmp_name,
       "mean_delta_AP":float(d.mean()),"median_delta_AP":float(np.median(d)),
       "paired_family_bootstrap_95CI":boot_ci(d,seed=70019+len(contrasts)),
       "wins_ties_losses":[int((d>0).sum()),int((d==0).sum()),int((d<0).sum())],
       "wilcoxon_p_raw":p})
for c,p in zip(contrasts,holm([x["wilcoxon_p_raw"] for x in contrasts])):
    c["wilcoxon_p_holm_2"]=p
fr=friedmanchisquare(*(means[m] for m in METHODS))
runtime=np.array([records[f]["WSAD_DT_seconds_mean"] for f in FAMILIES],float)
result={"status":"complete_posthoc_matched_comparator_extension","protocol_sha256":PH,
 "n_families":len(FAMILIES),"seeds_per_family":len(SEEDS),"family_inference_unit":True,
 "excluded_pilot":P["excluded_pilot"],"methods":summary,"family_level_contrasts":contrasts,
 "friedman":{"chi2":float(fr.statistic),"p":float(fr.pvalue)},
 "WSAD_DT_runtime_seconds":{"median_family_mean":float(np.median(runtime)),"mean_family_mean":float(runtime.mean()),"family_values":runtime.tolist()},
 "per_family":records,
 "interpretation_note":"All comparator-extension outputs are retrospective because the Replication 5 candidate outcomes had already been examined. Do not describe this as a new prospective confirmation."}
(OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
lines=["# Recent-comparator extension (post-hoc matched-split analysis)","",
 f"Families: {len(FAMILIES)}; seeds per family: {len(SEEDS)}; support: 50 labeled normals + 10 labeled anomalies.","",
 "This is a retrospective comparator extension: the candidate's Replication 5 outcomes were already examined. LandParcelValidation was used only as a code-feasibility pilot and excluded in full from inference.","",
 "| Method | Macro AP | Family SD | 95% family bootstrap CI |","|---|---:|---:|---:|"]
for m,label in zip(METHODS,["Proposed AP-weighted rank fusion","FairComb (adapted author procedure)","WSAD-DT (official defaults)"]):
    s=summary[m];ci=s["family_bootstrap_95CI"]
    lines.append(f"| {label} | {s['macro_AP']:.4f} | {s['family_sd']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
lines += ["","| Paired comparison | Mean AP difference | 95% paired CI | Wins/ties/losses | Raw p | Holm p |","|---|---:|---:|---:|---:|---:|"]
for c in contrasts:
    ci=c["paired_family_bootstrap_95CI"];w=c["wins_ties_losses"]
    lines.append(f"| {c['comparison']} | {c['mean_delta_AP']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {w[0]}/{w[1]}/{w[2]} | {c['wilcoxon_p_raw']:.6g} | {c['wilcoxon_p_holm_2']:.6g} |")
lines += ["",f"Friedman omnibus across the three methods: chi-square={fr.statistic:.4f}, p={fr.pvalue:.6g}.",
 f"WSAD-DT mean runtime per family (averaged across seeds): {runtime.mean():.1f}s; median {np.median(runtime):.1f}s.","",
 "The results are same-split and test-label-blind during method fitting/selection, but remain post-hoc relative to the primary study. They support only the tested tabular benchmark protocol."]
(OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(OUT/"summary.md")
