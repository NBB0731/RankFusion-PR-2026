from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"historical_20_family"/"ORIGINAL_RANKFUSION_REPLICATION5_CONFIRMATORY_ABLATIONS.json"
out=ROOT/"final_reanalysis"/"FINAL_19_FAMILY_RESULTS_REBUILT.json"
d=json.loads(source.read_text(encoding="utf-8")); maps=d["family_level_AP"]; excluded="LandParcelValidation"
families=[f for f in next(iter(maps.values())) if f!=excluded]
def boot(x,seed=70019,B=50000):
 x=np.asarray(x,float);r=np.random.default_rng(seed);a=r.choice(x,(B,len(x)),replace=True).mean(axis=1);return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]
metrics={}
for m,v in maps.items():
 x=np.asarray([v[f] for f in families],float);metrics[m]={"macro_AP":float(x.mean()),"between_family_SD":float(x.std(ddof=1)),"family_bootstrap_95CI":boot(x),"per_family_AP":{f:float(v[f]) for f in families}}
primary=maps["rank_meanAP_6"];cmp=["Extra Trees","six_expert_validation_selector","expanded_8_expert_validation_selector","rank_bootstrap_6","rank_uniform_6","rank_meanAP_4_no_fuzzy"]; contrasts=[];ps=[]
for i,c in enumerate(cmp):
 delta=np.asarray([primary[f]-maps[c][f] for f in families]);p=float(wilcoxon(delta,zero_method="wilcox",method="auto").pvalue);ps.append(p);contrasts.append({"comparator":c,"mean_delta_AP":float(delta.mean()),"paired_family_bootstrap_95CI":boot(delta,70019+i),"wins_ties_losses":[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())],"wilcoxon_p_raw":p})
order=np.argsort(ps);adj=np.empty(len(ps));prev=0
for rank,i in enumerate(order):prev=max(prev,min(1.,(len(ps)-rank)*ps[i]));adj[i]=prev
for c,p in zip(contrasts,adj):c["wilcoxon_p_holm_6"]=float(p)
result={"families":families,"excluded_pilot":excluded,"methods":metrics,"six_prespecified_primary_contrasts":contrasts,"inference_unit":"family after five-seed averaging"}
out.write_text(json.dumps(result,indent=2),encoding="utf-8");print(out)
