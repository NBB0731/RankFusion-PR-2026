"""Score the frozen replication-4 fusion ablations and planned contrasts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path(__file__).resolve().parent
ROOT=HERE/"macrodata_confirmatory"
PROTOCOL=ROOT/"FROZEN_GROUPCLEAN_REPLICATION5_PROTOCOL.json"
P=json.loads(PROTOCOL.read_text(encoding="utf-8"))
FAMILIES=P["selection"]["selected"]
SEEDS=P["protocol"]["seeds"]
EXPERTS=["RBF-SVM","Random Forest","Extra Trees","HistGradientBoosting","RRFEN fuzzy-only","RRFEN-Hybrid"]
CLASSICAL=EXPERTS[:4]
TEMP=.05
METHODS=["rank_uniform_6","rank_meanAP_6","rank_bootstrap_6","rank_meanAP_4_no_fuzzy","zscore_uniform_6","zscore_meanAP_6","zscore_bootstrap_6","rank_bootstrap_4_no_fuzzy","Extra Trees","PReNet","DevNet","six_expert_validation_selector","expanded_8_expert_validation_selector"]

def rank_pair(v,t):
    v=np.asarray(v,float); t=np.asarray(t,float)
    rv=(rankdata(v,method="average")-1)/max(len(v)-1,1)
    s=np.sort(v); rt=(np.searchsorted(s,t,"left")+np.searchsorted(s,t,"right")-1)/(2*max(len(s)-1,1))
    return rv,rt
def z_pair(v,t):
    mu=float(np.mean(v)); sd=float(np.std(v,ddof=1)); sd=sd if np.isfinite(sd) and sd>1e-12 else 1.
    return (v-mu)/sd,(t-mu)/sd
def weights(u):
    x=np.asarray(u,float)/TEMP; x-=x.max(); w=np.exp(x); return w/w.sum()
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0.]*len(ps); prev=0.
    for rank,i in enumerate(order): prev=max(prev,min(1.,(len(ps)-rank)*ps[i])); adj[i]=prev
    return adj
def boot_ci(x,rng,B=30000):
    x=np.asarray(x,float); m=rng.choice(x,size=(B,len(x)),replace=True).mean(axis=1)
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def main():
    by_family={m:[] for m in METHODS}; by_family_auc={m:[] for m in METHODS}; audit=[]
    for family in FAMILIES:
        per={m:[] for m in METHODS}; per_auc={m:[] for m in METHODS}
        for seed in SEEDS:
            folder=ROOT/"groupclean_replication5_predictions"/family/str(seed)
            deep=ROOT/"deepod_groupclean_replication5_predictions"/family/str(seed)
            with np.load(folder/"frozen_predictions.npz",allow_pickle=False) as z, np.load(deep/"predictions.npz",allow_pickle=False) as d:
                yv=z["validation_y"].astype(int); yt=z["y_test"].astype(int)
                val={m:z[f"validation__{m}"] for m in EXPERTS}; test={m:z[f"test__{m}"] for m in EXPERTS}
                rec=json.loads((folder/"run_record.json").read_text(encoding="utf-8")); boot=rec["bootstrap_selector"]
                vu=np.array([boot[m]["utility"] for m in EXPERTS]); vm=np.array([boot[m]["mean_AP"] for m in EXPERTS])
                def fuse(experts,cal,wt):
                    if wt=="uniform": w=np.full(len(experts),1/len(experts))
                    else:
                        ix=[EXPERTS.index(m) for m in experts]; w=weights((vm if wt=="meanAP" else vu)[ix])
                    f=np.zeros(len(yt),float)
                    for q,m in zip(w,experts):
                        _,s=(rank_pair if cal=="rank" else z_pair)(val[m],test[m]); f+=q*s
                    return f
                outs={
                    "rank_uniform_6":fuse(EXPERTS,"rank","uniform"),
                    "rank_meanAP_6":fuse(EXPERTS,"rank","meanAP"),
                    "rank_bootstrap_6":fuse(EXPERTS,"rank","bootstrap"),
                    "rank_meanAP_4_no_fuzzy":fuse(CLASSICAL,"rank","meanAP"),
                    "zscore_uniform_6":fuse(EXPERTS,"zscore","uniform"),
                    "zscore_meanAP_6":fuse(EXPERTS,"zscore","meanAP"),
                    "zscore_bootstrap_6":fuse(EXPERTS,"zscore","bootstrap"),
                    "rank_bootstrap_4_no_fuzzy":fuse(CLASSICAL,"rank","bootstrap"),
                    "Extra Trees":test["Extra Trees"],
                    "PReNet":d["prenet_test"],"DevNet":d["devnet_test"],
                }
                six=max(EXPERTS,key=lambda m:average_precision_score(yv,val[m]))
                outs["six_expert_validation_selector"]=test[six]
                deep_names=["PReNet","DevNet"]
                chosen=max([(average_precision_score(yv,val[m]),m) for m in EXPERTS]+[(average_precision_score(yv,d["prenet_validation"]),"PReNet"),(average_precision_score(yv,d["devnet_validation"]),"DevNet")])[1]
                outs["expanded_8_expert_validation_selector"]=test[chosen] if chosen in test else d["prenet_test" if chosen=="PReNet" else "devnet_test"]
                for m,s in outs.items():
                    per[m].append(float(average_precision_score(yt,s)))
                    per_auc[m].append(float(roc_auc_score(yt,s)))
            audit.append({"family":family,"seed":int(seed),"test_n":len(yt),"test_anomaly":int(yt.sum()),"six_selector":six,"expanded_selector":chosen})
        for m in METHODS:
            by_family[m].append(float(np.mean(per[m])))
            by_family_auc[m].append(float(np.mean(per_auc[m])))
    rng=np.random.default_rng(41027); metrics={}
    for m in METHODS:
        x=np.asarray(by_family[m],float); metrics[m]={"macro_AP":float(x.mean()),"sd_between_families":float(x.std(ddof=1)),"family_bootstrap_95CI":boot_ci(x,rng),"macro_AUROC":float(np.mean(by_family_auc[m])),"per_family_AP":x.tolist()}
    # AP-only validation weighting is the candidate; stability weighting is an explicit ablation.
    pairs=[("rank_meanAP_6",m) for m in ["Extra Trees","six_expert_validation_selector","expanded_8_expert_validation_selector","rank_bootstrap_6","rank_uniform_6","rank_meanAP_4_no_fuzzy"]]
    comparisons=[]
    for method,ref in pairs:
        delta=np.asarray(by_family[method])-np.asarray(by_family[ref])
        p=float(wilcoxon(delta,alternative="two-sided",zero_method="wilcox",method="auto").pvalue) if np.any(delta) else 1.
        comparisons.append({"contrast":method+" vs "+ref,"mean_delta_AP":float(delta.mean()),"paired_family_bootstrap_95CI":boot_ci(delta,rng),"wins_ties_losses":[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())],"wilcoxon_p_raw":p})
    adj=holm([c["wilcoxon_p_raw"] for c in comparisons])
    for c,p in zip(comparisons,adj): c["wilcoxon_p_holm_6"]=p
    result={"status":"frozen_replication5_confirmatory_test_scored_after_outputs_frozen","protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),"n_families":len(FAMILIES),"seeds_per_family":len(SEEDS),"inference_unit":"family after seed averaging","primary_metric":"AP","variant_definitions":{"rank":"validation empirical percentiles","zscore":"validation mean and sample SD normalization","meanAP":"softmax of mean stratified-bootstrap validation AP; temperature 0.05","bootstrap":"softmax of mean validation AP minus 2 bootstrap SD; temperature 0.05","no_fuzzy":"four classical experts only"},"macro_metrics":metrics,"pre_specified_paired_comparisons":comparisons,"family_level_AP":{m:{n:v for n,v in zip(FAMILIES,by_family[m])} for m in METHODS},"audit":audit,"interpretation":"All six listed contrasts were specified before scoring this cohort and Holm-corrected together. Positive results are evidence only for this frozen protocol and these families."}
    j=ROOT/"RANKFUSION_REPLICATION5_CONFIRMATORY_ABLATIONS.json"; md=ROOT/"RANKFUSION_REPLICATION5_CONFIRMATORY_ABLATIONS.md"
    j.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# Replication 5: frozen AP-weighted rank-fusion confirmation", "", f"Protocol SHA-256: `{result['protocol_sha256']}`", "", "20 previously unused family-disjoint datasets; five seeds per family; family is the inference unit. AP is primary. All variants and contrasts were frozen before test labels were scored.","", "## Macro performance", "", "| Method / variant | Macro AP | Between-family SD | 95% family-bootstrap CI | Macro AUROC |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        q=metrics[m]; lines.append(f"| {m} | {q['macro_AP']:.4f} | {q['sd_between_families']:.4f} | [{q['family_bootstrap_95CI'][0]:.4f}, {q['family_bootstrap_95CI'][1]:.4f}] | {q['macro_AUROC']:.4f} |")
    lines += ["", "## Pre-specified paired contrasts", "", "Δ AP is six-expert validation-AP-weighted percentile-rank fusion minus the comparator. Holm correction covers all six listed contrasts.","", "| Comparator | Mean Δ AP | 95% paired family-bootstrap CI | Wins/ties/losses | Holm p |", "|---|---:|---:|---:|---:|"]
    for c in comparisons:
        ci=c["paired_family_bootstrap_95CI"]; ref=c["contrast"].split(" vs ",1)[1]
        lines.append(f"| {ref} | {c['mean_delta_AP']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {'/'.join(map(str,c['wins_ties_losses']))} | {c['wilcoxon_p_holm_6']:.5g} |")
    lines += ["", "## Limits", "", "Interpret all outcomes according to the frozen protocol. The inferential unit is the dataset family; five seeds are averaged within family. This single confirmation cohort cannot on its own establish broad superiority.",""]
    md.write_text("\n".join(lines),encoding="utf-8")
    print("Wrote",md)

if __name__=="__main__": main()


