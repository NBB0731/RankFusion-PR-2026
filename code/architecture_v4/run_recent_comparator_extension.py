"""Frozen post-hoc matched-split comparison against FairComb and WSAD-DT."""
from __future__ import annotations
import hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.special import erf, ndtr
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE / "macrodata_confirmatory"
PROTOCOL = ROOT / "RECENT_COMPARATOR_EXTENSION_PROTOCOL.json"
P = json.loads(PROTOCOL.read_text(encoding="utf-8"))
PH = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
OUT = ROOT / "recent_comparator_extension_v2"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = HERE / "comparator_sources"
sys.path.insert(0, str(SOURCE / "WSAD-DT"))
sys.path.insert(0, str(HERE / "baseline_packages"))
sys.path.insert(0, str(HERE))
import torch
torch.set_num_threads(2)
from wsad_dt import WSADDT
import run_macrodata_groupclean_replication5 as split_runner

EXPERTS = ["RBF-SVM", "Random Forest", "Extra Trees", "HistGradientBoosting",
           "RRFEN fuzzy-only", "RRFEN-Hybrid"]

def znorm(v, t):
    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12: sd = 1.0
    return ndtr((v-mu)/sd), ndtr((t-mu)/sd)

def wcorr(x, y, w):
    w = np.asarray(w, float); w = w/w.sum()
    x=np.asarray(x,float); y=np.asarray(y,float)
    dx=x-np.sum(w*x); dy=y-np.sum(w*y)
    den=np.sqrt(np.sum(w*dx*dx)*np.sum(w*dy*dy))
    return float(np.sum(w*dx*dy)/den) if den > 1e-15 else 0.0

def faircomb_select(v, k):
    n,m=v.shape; k=min(max(int(k),1),max(n-1,1))
    z=np.column_stack([znorm(v[:,j],v[:,j])[0] for j in range(m)])
    top=np.zeros((n,m),bool)
    for j in range(m): top[np.argsort(-v[:,j],kind="mergesort")[:k],j]=True
    target=np.any(top,axis=1).astype(float); no=int(target.sum()); ni=n-no
    if no==0 or ni==0: return list(range(m))
    w=np.where(target>0,1/(2*no),1/(2*ni))
    selected=[int(np.argmax([wcorr(z[:,j],target,w) for j in range(m)]))]
    cur=z[:,selected].mean(axis=1); curc=wcorr(cur,target,w)
    remain=set(range(m))-set(selected)
    while remain:
        order=sorted(remain,key=lambda j:wcorr(z[:,j],cur,w)); added=False
        for j in order:
            trial=z[:,selected+[j]].mean(axis=1); tc=wcorr(trial,target,w)
            remain.remove(j)
            if tc>curc+1e-12:
                selected.append(j);cur=trial;curc=tc;added=True;break
        if not added: break
    return selected

def wtsr(raw, norm, alpha=.5):
    n=len(raw); ranks=rankdata(-np.asarray(raw,float),method="average")
    return alpha*norm+(1-alpha)*(n-ranks+1)/max(n,1)

def member_mask(v, selected, top=None):
    n=len(v); mask=np.zeros((n,len(selected)),bool)
    for c,j in enumerate(selected):
        if top is None:
            s=v[:,j];sd=float(np.std(s,ddof=1));sd=sd if np.isfinite(sd) and sd>1e-12 else 1.
            mask[:,c]=np.maximum(erf((s-float(np.mean(s)))/(sd*np.sqrt(2))),0)>.1
        else:
            k=max(1,int(np.floor(n*top)));mask[np.argsort(-v[:,j],kind="mergesort")[:k],c]=True
    return mask

def score_parity(norm_scores, members):
    means=[]; mean_sd=[]
    for j in range(members.shape[1]):
        z=norm_scores[members[:,j],j]
        if len(z):
            means.append(float(z.mean()))
            mean_sd.append(float(z.mean()+z.std(ddof=1 if len(z)>1 else 0)))
    return max(max(means)-min(means),max(mean_sd)-min(mean_sd)) if len(means)>1 else 0.0

def ep(agg, members):
    order=np.argsort(-agg,kind="mergesort"); ranks=np.empty(len(order),float)
    ranks[order]=np.arange(1,len(order)+1)
    ratios=[]
    for j in range(members.shape[1]):
        ids=np.flatnonzero(members[:,j])
        if len(ids): ratios.append(len(ids)*(len(ids)+1)/2/float(ranks[ids].sum()))
    return float(max(ratios)-min(ratios)) if len(ratios)>1 else 0.

def faircomb(v,t,k):
    selected=faircomb_select(v,k); vr=v[:,selected]; tr=t[:,selected]
    vz=[];tz=[]
    for j in range(vr.shape[1]):
        a,b=znorm(vr[:,j],tr[:,j]);vz.append(a);tz.append(b)
    vz=np.column_stack(vz);tz=np.column_stack(tz)
    orig_v=vz.mean(axis=1);orig_t=tz.mean(axis=1)
    members=member_mask(v,selected);sp=score_parity(vz,members)
    corr_v=np.column_stack([wtsr(vr[:,j],vz[:,j]) for j in range(len(selected))]).mean(axis=1)
    # FairComb computes EP from the per-observation maximum over detector
    # scores (its Max-EP default), while the final returned score is an average.
    ep0=ep(vz.max(axis=1),members);ep1=ep(np.column_stack([wtsr(vr[:,j],vz[:,j]) for j in range(len(selected))]).max(axis=1),members)
    accepted=sp>=.04 and ep1<ep0;fallback=None
    if sp>=.04 and not accepted:
        for top in (.05,.10,.15):
            fixed=member_mask(v,selected,top)
            sp_fixed=score_parity(vz,fixed)
            if sp_fixed < .04: continue
            ep2=ep(np.column_stack([wtsr(vr[:,j],vz[:,j]) for j in range(len(selected))]).max(axis=1),fixed)
            if ep2<ep0: accepted=True;fallback=top;members=fixed;ep1=ep2;break
    if accepted:
        out=np.column_stack([wtsr(tr[:,j],tz[:,j]) for j in range(len(selected))]).mean(axis=1);variant="WtSR"
    else: out=orig_t;variant="Z-Norm mean"
    return out,{"selected_experts":[EXPERTS[j] for j in selected],"selected_indices":selected,
        "validation_top_k":int(k),"score_parity":float(sp),"ep_before":float(ep0),"ep_after":float(ep1),
        "correction_applied":bool(accepted),"fallback_fixed_topk":fallback,"variant":variant,
        "ep_combination_rule":"max over selected expert scores","final_combination":"mean"}

def run_one(dataset, seed):
    folder=OUT/dataset/str(seed);folder.mkdir(parents=True,exist_ok=True);rp=folder/"result.json"
    if rp.exists():
        old=json.loads(rp.read_text(encoding="utf-8"))
        if old.get("protocol_sha256")==PH:return old
        raise RuntimeError(f"Stale result: {rp}")
    pred=ROOT/"groupclean_replication5_predictions"/dataset/str(seed)/"frozen_predictions.npz"
    recp=ROOT/"groupclean_replication5_predictions"/dataset/str(seed)/"run_record.json"
    data=ROOT/"public"/f"{dataset}.npz"
    with np.load(pred,allow_pickle=False) as z:
        yv=z["validation_y"].astype(int);yt=z["y_test"].astype(int)
        v=np.column_stack([z[f"validation__{m}"] for m in EXPERTS]).astype(float)
        t=np.column_stack([z[f"test__{m}"] for m in EXPERTS]).astype(float)
    rec=json.loads(recp.read_text(encoding="utf-8"));boot=rec["bootstrap_selector"]
    u=np.array([boot[m]["mean_AP"] for m in EXPERTS]);u-=u.max();w=np.exp(u/.05);w/=w.sum()
    rv=[];rt=[]
    for j in range(len(EXPERTS)):
        a=v[:,j];b=t[:,j];rv.append((rankdata(a,method="average")-1)/max(len(a)-1,1))
        s=np.sort(a);rt.append((np.searchsorted(s,b,"left")+np.searchsorted(s,b,"right")-1)/(2*max(len(s)-1,1)))
    cand_v=np.column_stack(rv)@w;cand_t=np.column_stack(rt)@w
    fair_t,fmeta=faircomb(v,t,int(yv.sum()))
    fmeta["gselect_z_norm_validation_AP"]=float(average_precision_score(yv,np.column_stack([znorm(v[:,j],v[:,j])[0] for j in fmeta["selected_indices"]]).mean(axis=1)))
    Xu,Xc,yc,Xv,yv2,Xt,yt2,*_=split_runner.prepare_group(data,seed)
    if not(np.array_equal(yv,yv2) and np.array_equal(yt,yt2)):raise RuntimeError("split differs from frozen prediction labels")
    Xtr=np.vstack([Xu,Xc]);yw=np.r_[np.zeros(len(Xu),int),yc.astype(int)]
    mm=MinMaxScaler().fit(Xtr);Xtr=mm.transform(Xtr).astype(np.float32)
    Xv=mm.transform(Xv).astype(np.float32);Xt=mm.transform(Xt).astype(np.float32)
    tic=time.perf_counter();model=WSADDT(n_ensemble=5,seed=seed,batch_size=64,parallel=False).fit(Xtr,yw)
    ws_v=model.decision_function(Xv);ws_t=model.decision_function(Xt);secs=time.perf_counter()-tic
    result={"status":"complete","dataset":dataset,"seed":int(seed),"protocol_sha256":PH,
      "official_source_commits":{"WSAD-DT":"b59c4aa4595fd24462e9c6852c1fc792a0dc7d0d","FairComb":"1309b7780e61c96998246135348bbfdfb2939b01"},
      "metrics":{"candidate_validation_AP":float(average_precision_score(yv,cand_v)),
       "candidate_test_AP":float(average_precision_score(yt,cand_t)),"FairComb_test_AP":float(average_precision_score(yt,fair_t)),
       "WSAD_DT_validation_AP":float(average_precision_score(yv,ws_v)),"WSAD_DT_test_AP":float(average_precision_score(yt,ws_t))},
      "faircomb":fmeta,"wsad_dt":{"official_defaults":{"n_ensemble":5,"batch_size":64,"epochs":100,"seed":int(seed),"parallel":False},
       "fit_and_score_seconds":float(secs),"train_rows":int(len(Xtr))},
      "counts":{"unlabeled_reference":int(len(Xu)),"known_normal_support":int((yc==0).sum()),
       "known_anomaly_support":int((yc==1).sum()),"validation":int(len(yv)),"test":int(len(yt))},
      "test_labels_used_in_model_or_fusion_selection":False}
    np.savez_compressed(folder/"predictions.npz",y_test=yt,candidate_test=cand_t,faircomb_test=fair_t,
      wsad_dt_test=ws_t,y_validation=yv,candidate_validation=cand_v,wsad_dt_validation=ws_v)
    result["prediction_sha256"]=hashlib.sha256((folder/"predictions.npz").read_bytes()).hexdigest()
    result["runner_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(f"{dataset} {seed}: AP cand={result['metrics']['candidate_test_AP']:.4f}, FairComb={result['metrics']['FairComb_test_AP']:.4f}, WSAD-DT={result['metrics']['WSAD_DT_test_AP']:.4f}; WSAD-DT {secs:.1f}s",flush=True)
    return result

def main():
    import argparse
    pa=argparse.ArgumentParser();pa.add_argument("--datasets",nargs="*",default=P["datasets"]);pa.add_argument("--seeds",nargs="*",type=int,default=P["seeds"]);a=pa.parse_args()
    if set(a.datasets)-set(P["datasets"]) or set(a.seeds)-set(P["seeds"]):raise ValueError("Outside frozen cohort")
    env={"protocol_sha256":PH,"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "wsad_dt_source_commit":"b59c4aa4595fd24462e9c6852c1fc792a0dc7d0d","faircomb_source_commit":"1309b7780e61c96998246135348bbfdfb2939b01",
      "torch_version":torch.__version__,"cuda_available":bool(torch.cuda.is_available()),"torch_threads":torch.get_num_threads(),"python":sys.version}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    failures=[]
    with threadpool_limits(limits=2):
        for ds in a.datasets:
            for seed in a.seeds:
                try:run_one(ds,seed)
                except Exception:
                    f=OUT/ds/str(seed);f.mkdir(parents=True,exist_ok=True);(f/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
                    failures.append({"dataset":ds,"seed":seed});print("FAILED",ds,seed,flush=True)
    (OUT/"failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    print(f"Finished. failures={len(failures)}",flush=True)
if __name__=="__main__":main()
