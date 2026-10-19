"""Frozen CPU-only OddBench confirmatory run. Test labels are never scored here."""
from __future__ import annotations
import hashlib, itertools, json, sys, time, traceback, warnings, os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.stats import bootstrap
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
ARC=ROOT.parent
V3=ROOT.parent/"architecture_v3"
sys.path.insert(0,str(V3)); sys.path.insert(0,str(ARC))
from rrfen import RRFEN, _prepare_component
from rrfen_hybrid import RRFENHybrid, candidate_grids

DATA=ROOT/"macrodata_confirmatory"
PROTOCOL_PATH=DATA/os.environ.get("ODDBENCH_PROTOCOL_FILE","FROZEN_CONFIRMATORY_PROTOCOL.json")
P=json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PH=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
MH=hashlib.sha256((DATA/"FROZEN_METADATA_MANIFEST.json").read_bytes()).hexdigest()
SCRIPT=Path(__file__)
SH=hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
OUT=DATA/os.environ.get("ODDBENCH_OUTPUT_FOLDER","predictions")
SEEDS=P["protocol"]["seeds"]
N_BOOT=500

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def ap(y,s): return float(average_precision_score(y,s))

def fit_grid(name,Xc,yc,Xv,yv,seed):
    if name=="RBF-SVM":
        grid=[{"C":c,"gamma":g/Xc.shape[1]} for c,g in itertools.product((.1,1.,10.),(.1,1.,10.))]
        maker=lambda p:SVC(C=p["C"],gamma=p["gamma"],class_weight="balanced",cache_size=256)
        score=lambda m,x:m.decision_function(x)
    elif name=="Random Forest":
        grid=[{"min_samples_leaf":l,"max_features":f} for l,f in itertools.product((1,5,20),(.5,.75,1.))]
        maker=lambda p:RandomForestClassifier(n_estimators=200,min_samples_leaf=p["min_samples_leaf"],max_features=p["max_features"],class_weight="balanced",random_state=seed,n_jobs=2)
        score=lambda m,x:m.predict_proba(x)[:,1]
    elif name=="Extra Trees":
        grid=[{"min_samples_leaf":l,"max_features":f} for l,f in itertools.product((1,3,10),(.5,1.))]
        maker=lambda p:ExtraTreesClassifier(n_estimators=300,min_samples_leaf=p["min_samples_leaf"],max_features=p["max_features"],class_weight="balanced",random_state=seed,n_jobs=2)
        score=lambda m,x:m.predict_proba(x)[:,1]
    elif name=="HistGradientBoosting":
        grid=[{"max_leaf_nodes":l,"l2_regularization":r} for l,r in itertools.product((7,15,31),(.1,1.,10.))]
        maker=lambda p:HistGradientBoostingClassifier(max_leaf_nodes=p["max_leaf_nodes"],l2_regularization=p["l2_regularization"],learning_rate=.05,max_iter=150,min_samples_leaf=20,class_weight="balanced",early_stopping=False,random_state=seed)
        score=lambda m,x:m.predict_proba(x)[:,1]
    else: raise ValueError(name)
    best=None; trials=[]; start=time.perf_counter()
    for p in grid:
        m=maker(p);m.fit(Xc,yc);sv=np.asarray(score(m,Xv),float); va=ap(yv,sv)
        trials.append({"params":p,"validation_AP":va})
        if best is None or va>best[0]:best=(va,p,m,sv)
    fit_grid_seconds=time.perf_counter()-start
    start=time.perf_counter();m=maker(best[1]).fit(Xc,yc)
    refit_seconds=time.perf_counter()-start
    return {"params":best[1],"validation_score":best[3],"validation_AP":best[0],"model":m,"trials":trials,
            "score_fn":score,"fit_seconds":fit_grid_seconds+refit_seconds,"refit_seconds":refit_seconds}

def prepare(path,seed):
    z=np.load(path,allow_pickle=False)
    tr=np.asarray(z["train"],float);te=np.asarray(z["test"],float);yt=np.asarray(z["test_labels"],int).ravel()
    tr[~np.isfinite(tr)]=np.nan;te[~np.isfinite(te)]=np.nan
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(tr));n_u=int(np.floor(.8*len(tr)));u=perm[:n_u];normal_pool=perm[n_u:]
    normal_cal=rng.choice(normal_pool,size=min(50,len(normal_pool)),replace=False)
    pos=np.flatnonzero(yt==1);neg=np.flatnonzero(yt==0)
    support=rng.choice(pos,size=10,replace=False)
    remaining=np.setdiff1d(np.arange(len(yt)),support,assume_unique=False)
    v,t=train_test_split(remaining,test_size=.5,random_state=seed,stratify=yt[remaining])
    yc=np.r_[np.zeros(len(normal_cal),dtype=int),np.ones(len(support),dtype=int)]
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(tr[u])
    scaler=StandardScaler().fit(imp.transform(tr[u]))
    Xu=scaler.transform(imp.transform(tr[u]))
    Xc=scaler.transform(imp.transform(np.vstack([tr[normal_cal],te[support]])))
    Xv=scaler.transform(imp.transform(te[v]));Xt=scaler.transform(imp.transform(te[t]))
    return Xu,Xc,yc,Xv,yt[v],Xt,yt[t],v,t,support,normal_cal,{
        "n_train":len(tr),"n_test":len(te),"d":tr.shape[1],"support_pos":len(support),"calibration_normal":len(normal_cal),
        "validation_n":len(v),"validation_pos":int(yt[v].sum()),"test_n":len(t),"test_pos":int(yt[t].sum()),
        "test_rows_not_scored_until_aggregate":True}

def run_one(name,seed):
    out=OUT/name/str(seed);out.mkdir(parents=True,exist_ok=True)
    pred=out/"frozen_predictions.npz";meta=out/"run_record.json"
    if pred.exists() and meta.exists():
        old=json.loads(meta.read_text(encoding="utf-8"))
        if old.get("protocol_sha256")==PH and old.get("script_sha256")==SH:return old
        raise RuntimeError(f"Stale frozen output exists: {pred}")
    path=DATA/"public"/(name+".npz")
    Xu,Xc,yc,Xv,yv,Xt,yt,v,t,support,normal_cal,counts=prepare(path,seed)
    vals={};tests={};records={}
    for name_m in ("RBF-SVM","Random Forest","Extra Trees","HistGradientBoosting"):
        r=fit_grid(name_m,Xc,yc,Xv,yv,seed)
        vals[name_m]=r["validation_score"]
        start=time.perf_counter();tests[name_m]=np.asarray(r["score_fn"](r["model"],Xt),float);infer=time.perf_counter()-start
        records[name_m]={"params":r["params"],"validation_AP":r["validation_AP"],"trials":r["trials"],"fit_seconds":r["fit_seconds"],"inference_seconds":infer}

    cache={(k,q):_prepare_component(Xu,min(k,len(Xu)-1),q) for k in (5,10,20) for q in (.8,.9,.98)}
    probe=RRFEN(topology_ks=(5,10,20),topology_qs=(.8,.9,.98));probe.topo_models=cache
    topo_v=probe.topology_evidence(Xv);topo_t=probe.topology_evidence(Xt)
    fuzzy_best=None;fuzzy_trials=[];fuzzy_search_start=time.perf_counter()
    for proto,local_k in itertools.product((8,16,32),(5,15,25)):
        m=RRFEN(normal_centers=proto,anomaly_centers=proto,local_k=local_k,topology_ks=(5,10,20),topology_qs=(.8,.9,.98))
        m.fit(Xu,Xc,yc,topology_cache=cache,learn_gate=False,random_state=seed)
        sv=np.asarray(m.predict_score(Xv,topo_v),float);score_ap=ap(yv,sv)
        fuzzy_trials.append({"proto":proto,"local_k":local_k,"validation_AP":score_ap})
        if fuzzy_best is None or score_ap>fuzzy_best[0]:fuzzy_best=(score_ap,proto,local_k,m,sv)
    fuzzy_search_time=time.perf_counter()-fuzzy_search_start
    vals["RRFEN fuzzy-only"]=fuzzy_best[4]
    start=time.perf_counter();tests["RRFEN fuzzy-only"]=np.asarray(fuzzy_best[3].predict_score(Xt,topo_t),float);fuzzy_infer=time.perf_counter()-start
    records["RRFEN fuzzy-only"]={"params":{"prototype":fuzzy_best[1],"local_k":fuzzy_best[2]},"validation_AP":fuzzy_best[0],"trials":fuzzy_trials,"fit_seconds":fuzzy_search_time,"inference_seconds":fuzzy_infer}

    # Refit fuzzy settings and two selected raw-feature learners inside the
    # hybrid, using only choices already selected on the validation set.
    svm_params=records["RBF-SVM"]["params"]
    rf_params=records["Random Forest"]["params"]
    model=RRFENHybrid(proto=fuzzy_best[1],local_k=fuzzy_best[2],gate_C=.1,
                      svm_params=svm_params,rf_params={"min_leaf":rf_params["min_samples_leaf"],"max_features":rf_params["max_features"]},
                      topology_ks=(5,10,20),topology_qs=(.8,.9,.98))
    start=time.perf_counter();model.fit(Xu,Xc,yc,seed=seed);hyb_fit=time.perf_counter()-start
    start=time.perf_counter();vals["RRFEN-Hybrid"]=np.asarray(model.decision_function(Xv,topology=topo_v),float);tests["RRFEN-Hybrid"]=np.asarray(model.decision_function(Xt,topology=topo_t),float);hyb_infer=time.perf_counter()-start
    records["RRFEN-Hybrid"]={"params":{"prototype":fuzzy_best[1],"local_k":fuzzy_best[2],"gate_C":.1,"RBF_SVM":svm_params,"Random_Forest":rf_params},"validation_AP":ap(yv,vals["RRFEN-Hybrid"]),"fit_seconds":hyb_fit,"inference_seconds":hyb_infer}

    # Validation-only stratified bootstrap stability selector; labels and test
    # predictions remain absent from selection logic.
    rng=np.random.default_rng(seed+33019);boot={}
    for meth,sv in vals.items():
        by=[]
        for _ in range(N_BOOT):
            i0=rng.choice(np.flatnonzero(yv==0),size=int((yv==0).sum()),replace=True)
            i1=rng.choice(np.flatnonzero(yv==1),size=int((yv==1).sum()),replace=True)
            by.append(ap(np.r_[yv[i0],yv[i1]],np.r_[sv[i0],sv[i1]]))
        boot[meth]={"mean_AP":float(np.mean(by)),"sd_AP":float(np.std(by,ddof=1)),"utility":float(np.mean(by)-2*np.std(by,ddof=1))}
    selected=max(boot,key=lambda k:boot[k]["utility"])
    tests["Stability selector"]=tests[selected].copy()
    # Direct validation-AP selection is an explicit comparator in the new
    # untouched-family rank-fusion cohort.
    validation_best=max(vals,key=lambda k:ap(yv,vals[k]))
    tests["Validation-AP maximum selector"]=tests[validation_best].copy()
    # Bootstrap-weighted rank fusion: fixed alpha=2 and temperature=.05 from
    # leave-one-family-out ADBench development, then percentile-map each test
    # score against its own validation-score distribution.
    experts=list(vals)
    utility=np.array([boot[m]["utility"] for m in experts],float)
    utility-=utility.max()
    weights=np.exp(utility/.05);weights/=weights.sum()
    fused_v=np.zeros(len(yv),float);fused_t=np.zeros(len(yt),float)
    for w,m in zip(weights,experts):
        sv=np.asarray(vals[m],float);st=np.asarray(tests[m],float)
        rv=(rankdata(sv,method="average")-1)/max(len(sv)-1,1)
        ref=np.sort(sv)
        rt=(np.searchsorted(ref,st,"left")+np.searchsorted(ref,st,"right")-1)/(2*max(len(ref)-1,1))
        fused_v+=w*rv;fused_t+=w*rt
    vals["Bootstrap-weighted rank fusion"]=fused_v
    tests["Bootstrap-weighted rank fusion"]=fused_t
    boot["rank_fusion_weights"]={m:float(w) for m,w in zip(experts,weights)}
    boot["rank_fusion_fixed_alpha"]=2.0;boot["rank_fusion_fixed_temperature"]=.05
    frozen={"test_scores":tests,"selected_expert":selected}
    np.savez_compressed(pred,y_test=yt,validation_y=yv,**{f"validation__{k}":x for k,x in vals.items()},**{f"test__{k}":x for k,x in tests.items()})
    rec={"status":"predictions_frozen_test_metrics_not_computed","dataset":name,"seed":int(seed),"protocol_sha256":PH,"metadata_manifest_sha256":MH,"script_sha256":SH,
         "source_sha256":sha(path),"counts":counts,"validation_records":records,"bootstrap_selector":boot,"selected_expert":selected,
         "validation_ap_max_expert":validation_best,
         "test_prediction_sha256":sha(pred),"prediction_file":str(pred.relative_to(DATA)),"cpu_only":True}
    meta.write_text(json.dumps(rec,indent=2,allow_nan=False),encoding="utf-8")
    print(name,seed,"predictions frozen; val selected",selected,"val_AP",{k:round(ap(yv,s),4) for k,s in vals.items()},flush=True)
    return rec

def main():
    expected=set(P["selection"]["selected"])
    import argparse
    pa=argparse.ArgumentParser();pa.add_argument("--datasets",nargs="*",default=sorted(expected));pa.add_argument("--seeds",nargs="*",type=int,default=SEEDS);args=pa.parse_args()
    if set(args.datasets)-expected:raise ValueError("Dataset outside frozen metadata manifest")
    if set(args.seeds)-set(SEEDS):raise ValueError("Seed outside frozen protocol")
    (DATA/"environment.json").write_text(json.dumps({"protocol_sha256":PH,"metadata_sha256":MH,"script_sha256":SH,"thread_limit":2,"python":sys.version},indent=2),encoding="utf-8")
    with threadpool_limits(limits=2):
        for name in args.datasets:
            for seed in args.seeds:
                try:run_one(name,seed)
                except Exception:
                    p=OUT/name/str(seed);p.mkdir(parents=True,exist_ok=True);(p/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8");raise
    print("All requested score-blind prediction outputs are frozen; do not edit protocol or rerun with changed choices.",flush=True)

if __name__=="__main__":main()
