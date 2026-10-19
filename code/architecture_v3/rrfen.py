"""Reliability-routed fuzzy evidence network (development implementation).

This is a candidate architecture, not yet frozen or a validated result.
The three evidence channels are class-conditional fuzzy prototypes, a
Beta-smoothed labelled-neighbour posterior, and persistent graph isolation.
Each channel is mapped through an explicit confidence-conditioned Sugeno
rule table; a confidence-weighted simplex combines the channel scores.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def _risk_membership(s):
    s=np.clip(np.asarray(s,float),0,1)
    low=np.clip(1-2*s,0,1)
    mid=np.clip(1-2*np.abs(s-.5),0,1)
    high=np.clip(2*s-1,0,1)
    return np.stack([low,mid,high],axis=1)


def _fuzzy_source_score(score,reliability):
    """Two-confidence-term, three-risk-term product Sugeno inference."""
    rm=_risk_membership(score)
    conf=np.clip(np.asarray(reliability,float),0,1)
    cm=np.stack([1-conf,conf],axis=1)
    # Low confidence shrinks the risk toward indifference; high-confidence
    # evidence is mapped monotonically to low/medium/high risk.
    consequent=np.array([[.45,.05],[.50,.50],[.55,.95]])
    firing=rm[:,:,None]*cm[:,None,:]
    return np.sum(firing*consequent[None,:,:],axis=(1,2))/np.maximum(firing.sum(axis=(1,2)),1e-12)


def _prepare_component(reference,k,q,decay=.02):
    k=min(int(k),len(reference)-1)
    if k<1: raise ValueError('topology k must be positive')
    nn=NearestNeighbors(n_neighbors=k+1,algorithm='auto',n_jobs=1).fit(reference)
    dr,ir=nn.kneighbors(reference)
    # Remove the actual row index, not simply column zero (duplicates exist).
    keep=ir!=np.arange(len(reference))[:,None]
    kth=np.empty((len(reference),k)); nbr=np.empty((len(reference),k),int)
    for i in range(len(reference)):
        kth[i]=dr[i,keep[i]][:k];nbr[i]=ir[i,keep[i]][:k]
    radius=float(np.quantile(kth[:,-1],q,method='linear'))
    mask=kth<=radius
    rows=np.broadcast_to(np.arange(len(reference))[:,None],nbr.shape)[mask]
    graph=csr_matrix((np.ones(len(rows)),(rows,nbr[mask])),shape=(len(reference),len(reference)))
    graph=graph.maximum(graph.T)
    _,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels)
    nq=NearestNeighbors(n_neighbors=k,algorithm='auto',n_jobs=1).fit(reference)
    return nq, radius, labels, sizes, np.sort(kth[:,-1]), decay


def _score_component(query,prepared):
    nq,radius,labels,sizes,ref,decay=prepared
    dq,iq=nq.kneighbors(query)
    connected=dq<=radius
    component=np.where(connected,sizes[labels[iq]],1)
    # Match the defined topology evidence: the query inherits its largest
    # reached component, so a single tiny component cannot dominate a query
    # that is also connected to a large, well-supported component.
    largest_reached=component.max(axis=1)
    topological=np.exp(-decay*(largest_reached-1))
    # Empirical tail rank of the query k-distance, with midranks for ties.
    d=dq[:,-1]
    left=np.searchsorted(ref,d,'left');right=np.searchsorted(ref,d,'right')
    geometric=np.clip((left+right-1)/(2*max(len(ref)-1,1)),0,1)
    return np.column_stack([geometric,topological])


class RRFEN:
    """Reliability-routed fuzzy evidence with an out-of-fold learned gate."""
    def __init__(self,normal_centers=16,anomaly_centers=16,local_k=15,
                 topology_ks=(5,10,20),topology_qs=(.8,.9,.98),
                 beta_prior=1.0,confidence_floor=.10):
        self.normal_centers=normal_centers;self.anomaly_centers=anomaly_centers
        self.local_k=local_k;self.topology_ks=tuple(topology_ks)
        self.topology_qs=tuple(topology_qs);self.beta_prior=beta_prior
        self.confidence_floor=confidence_floor

    def fit(self,Xu,Xc,yc,topology_cache=None,learn_gate=True,random_state=177):
        self.Xu=np.asarray(Xu,float);self.Xc=np.asarray(Xc,float);self.yc=np.asarray(yc,int).ravel()
        if set(np.unique(self.yc))!={0,1}:raise ValueError('both classes required in calibration set')
        centers=[];center_y=[]
        for cls,k in [(0,self.normal_centers),(1,self.anomaly_centers)]:
            z=self.Xc[self.yc==cls]
            k=min(int(k),len(z))
            km=MiniBatchKMeans(n_clusters=k,n_init=3,batch_size=512,random_state=random_state+cls).fit(z)
            centers.append(km.cluster_centers_);center_y.extend([cls]*k)
        self.centers=np.vstack(centers);self.center_y=np.asarray(center_y)
        normal=self.centers[self.center_y==0]
        self.sigma=max(float(np.median(np.sqrt(cdist(self.Xc[self.yc==0],normal,'sqeuclidean').min(axis=1)))),1e-6)
        fc=self._proto_features(self.Xc)
        self.feature_scaler=StandardScaler().fit(fc)
        fc=self.feature_scaler.transform(fc)
        self.proto=LogisticRegression(C=1.0,class_weight='balanced',max_iter=3000,random_state=random_state).fit(fc,self.yc)
        self.local_nn=NearestNeighbors(n_neighbors=min(max(self.local_k,25),len(self.Xc)),algorithm='auto',n_jobs=1).fit(self.Xc)
        if topology_cache is None:
            topology_cache={}
            for k in sorted(set(self.topology_ks)):
                kk=min(k,len(self.Xu)-1)
                for q in self.topology_qs:topology_cache[(k,q)]=_prepare_component(self.Xu,kk,q)
        self.topo_models=topology_cache
        self.gate=None
        if learn_gate and min(np.bincount(self.yc,minlength=2))>=2:
            n_splits=min(5,int(np.bincount(self.yc,minlength=2).min()))
            oof=np.zeros((len(self.yc),12),float)
            cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
            topo_c=self.topology_evidence(self.Xc)
            for it,iv in cv.split(self.Xc,self.yc):
                base=RRFEN(normal_centers=self.normal_centers,anomaly_centers=self.anomaly_centers,
                           local_k=self.local_k,topology_ks=self.topology_ks,
                           topology_qs=self.topology_qs,beta_prior=self.beta_prior,
                           confidence_floor=self.confidence_floor)
                base.fit(self.Xu,self.Xc[it],self.yc[it],topology_cache=self.topo_models,
                         learn_gate=False,random_state=random_state)
                e=base.predict_evidence(self.Xc[iv],(topo_c[0][iv],topo_c[1][iv]))
                oof[iv]=self._gate_features(*e)
            self.gate=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,
                                         random_state=random_state).fit(oof,self.yc)
        return self

    def _proto_features(self,X):
        logu=-cdist(np.asarray(X,float),self.centers,'sqeuclidean')/(2*self.sigma**2)
        logu-=logu.max(axis=1,keepdims=True)
        a=np.exp(logu);return a/np.maximum(a.sum(axis=1,keepdims=True),1e-15)

    def _local(self,X):
        d,ix=self.local_nn.kneighbors(X,n_neighbors=min(self.local_k,len(self.Xc)))
        scale=np.maximum(d[:,-1:],1e-12)
        w=np.exp(-d/scale)
        ya=self.yc[ix]
        p=(np.sum(w*ya,axis=1)+self.beta_prior)/(np.sum(w,axis=1)+2*self.beta_prior)
        neff=(w.sum(axis=1)**2)/np.maximum((w*w).sum(axis=1),1e-12)
        entropy=-(p*np.log(np.clip(p,1e-12,1))+(1-p)*np.log(np.clip(1-p,1e-12,1)))/np.log(2)
        rel=(neff/(neff+5))*np.clip(1-entropy,0,1)
        return p,rel

    def topology_evidence(self,X):
        evidence=np.asarray([_score_component(X,self.topo_models[(k,q)])
                             for k in self.topology_ks for q in self.topology_qs])
        # Keep distance extremeness and graph isolation as distinct channels;
        # averaging them can erase a useful distance ordering when a graph is
        # globally connected at one or more scales.
        score=np.median(evidence,axis=0)
        reliability=np.exp(-4*np.std(evidence,axis=0))
        return score,reliability

    def predict_evidence(self,X,topology_evidence=None):
        X=np.asarray(X,float)
        act=self._proto_features(X)
        pproto=self.proto.predict_proba(self.feature_scaler.transform(act))[:,1]
        ent=-(act*np.log(np.clip(act,1e-12,1))).sum(axis=1)/np.log(max(act.shape[1],2))
        proto_rel=np.clip((1-ent)*(2*np.abs(pproto-.5)),0,1)
        plocal,local_rel=self._local(X)
        if topology_evidence is None:topology_evidence=self.topology_evidence(X)
        topo_score,topo_rel=topology_evidence
        scores=np.column_stack([pproto,plocal,topo_score[:,0],topo_score[:,1]])
        rels=np.column_stack([proto_rel,local_rel,topo_rel[:,0],topo_rel[:,1]])
        return scores,rels

    def predict_score(self,X,topology_evidence=None):
        scores,rels=self.predict_evidence(X,topology_evidence)
        transformed=np.column_stack([_fuzzy_source_score(scores[:,j],rels[:,j]) for j in range(scores.shape[1])])
        if self.gate is not None:return self.gate.decision_function(self._gate_features(scores,rels))
        weights=self.confidence_floor+(1-self.confidence_floor)*rels
        weights/=weights.sum(axis=1,keepdims=True)
        return np.sum(weights*transformed,axis=1)

    def predict_equal_score(self,X,topology_evidence=None):
        scores,rels=self.predict_evidence(X,topology_evidence)
        return self._gate_features(scores,rels)[:,:scores.shape[1]].mean(axis=1)

    @staticmethod
    def _gate_features(scores,rels):
        fuzzy=np.column_stack([_fuzzy_source_score(scores[:,j],rels[:,j]) for j in range(scores.shape[1])])
        return np.column_stack([fuzzy,rels,fuzzy*rels])
