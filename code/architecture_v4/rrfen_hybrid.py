"""Cross-fitted fuzzy-evidence and feature-expert fusion (development candidate).

The four fuzzy evidence channels are retained, while RBF-SVM and Random
Forest provide direct feature-level experts. A regularized logistic router is
trained only on out-of-fold predictions from the labeled calibration set.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rrfen import RRFEN, _prepare_component, _fuzzy_source_score


def _rank_fit(x):
    x = np.asarray(x, float)
    return np.sort(x, axis=0)


def _rank_transform(x, sorted_reference):
    x = np.asarray(x, float)
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        ref = sorted_reference[:, j]
        out[:, j] = (np.searchsorted(ref, x[:, j], side="right") +
                     np.searchsorted(ref, x[:, j], side="left")) / (2 * max(len(ref), 1))
    return np.clip(out, 0.0, 1.0)


def _make_features(svm_score, rf_score, fuzzy, reliability, rank_reference=None):
    raw = np.column_stack([svm_score, rf_score, fuzzy, reliability])
    if rank_reference is None:
        ref = _rank_fit(raw)
        return _rank_transform(raw, ref), ref
    return _rank_transform(raw, rank_reference), rank_reference


def _fit_svm(X, y, params):
    return SVC(C=float(params["C"]), gamma=float(params["gamma"]),
               class_weight="balanced", cache_size=256).fit(X, y)


def _fit_rf(X, y, params, seed):
    return RandomForestClassifier(n_estimators=200,
        min_samples_leaf=int(params["min_leaf"]),
        max_features=float(params["max_features"]), class_weight="balanced",
        random_state=int(seed), n_jobs=2).fit(X, y)


def _scores(model, X, method):
    if method == "svm":
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def _fuzzy_features(model, X, topology):
    score, rel = model.predict_evidence(X, topology)
    fs = np.column_stack([_fuzzy_source_score(score[:, j], rel[:, j])
                          for j in range(score.shape[1])])
    weights = model.confidence_floor + (1-model.confidence_floor)*rel
    aggregate = np.sum(weights*fs, axis=1)/np.maximum(weights.sum(axis=1), 1e-12)
    return aggregate, fs, rel


class RRFENHybrid:
    """RRFEN evidence channels routed jointly with raw-feature experts."""
    def __init__(self, proto=16, local_k=15, gate_C=.1, svm_params=None,
                 rf_params=None, topology_ks=(5, 10, 20),
                 topology_qs=(.8, .9, .98)):
        self.proto = int(proto)
        self.local_k = int(local_k)
        self.gate_C = float(gate_C)
        self.svm_params = svm_params or {"C": 1., "gamma": .1}
        self.rf_params = rf_params or {"min_leaf": 5, "max_features": .75}
        self.topology_ks = tuple(topology_ks)
        self.topology_qs = tuple(topology_qs)

    def fit(self, Xu, Xc, yc, seed=0):
        Xu, Xc, yc = np.asarray(Xu, float), np.asarray(Xc, float), np.asarray(yc, int)
        self.Xu, self.Xc, self.yc = Xu, Xc, yc
        self.topology_cache = {(k, q): _prepare_component(Xu, min(k, len(Xu)-1), q)
            for k in self.topology_ks for q in self.topology_qs}
        probe = RRFEN(normal_centers=self.proto, anomaly_centers=self.proto,
            local_k=self.local_k, topology_ks=self.topology_ks,
            topology_qs=self.topology_qs)
        probe.topo_models = self.topology_cache
        topo_c = probe.topology_evidence(Xc)

        n_pos = int(np.bincount(yc, minlength=2).min())
        folds = min(5, n_pos)
        if folds < 2:
            raise ValueError("At least two examples per class are required for cross-fitting")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        # Two feature-level classifier experts, one fuzzy aggregate,
        # four fuzzy channel scores, and four reliability descriptors.
        oof = np.zeros((len(yc), 11), dtype=float)
        for tr, va in skf.split(Xc, yc):
            svm = _fit_svm(Xc[tr], yc[tr], self.svm_params)
            rf = _fit_rf(Xc[tr], yc[tr], self.rf_params, seed+len(tr))
            fuzzy_model = RRFEN(normal_centers=self.proto, anomaly_centers=self.proto,
                local_k=self.local_k, topology_ks=self.topology_ks,
                topology_qs=self.topology_qs).fit(Xu, Xc[tr], yc[tr],
                    topology_cache=self.topology_cache, learn_gate=False,
                    random_state=int(seed)+17)
            sf, fs, rel = _fuzzy_features(fuzzy_model, Xc[va],
                (topo_c[0][va], topo_c[1][va]))
            oof[va] = np.column_stack([_scores(svm, Xc[va], "svm"),
                _scores(rf, Xc[va], "rf"), sf, fs, rel])

        self.rank_reference = _rank_fit(oof)
        oof_rank = _rank_transform(oof, self.rank_reference)
        self.meta_scaler = StandardScaler().fit(oof_rank)
        self.gate = LogisticRegression(C=self.gate_C, class_weight="balanced",
            max_iter=3000, random_state=int(seed)).fit(
                self.meta_scaler.transform(oof_rank), yc)

        self.svm = _fit_svm(Xc, yc, self.svm_params)
        self.rf = _fit_rf(Xc, yc, self.rf_params, seed)
        self.fuzzy_model = RRFEN(normal_centers=self.proto, anomaly_centers=self.proto,
            local_k=self.local_k, topology_ks=self.topology_ks,
            topology_qs=self.topology_qs).fit(Xu, Xc, yc,
                topology_cache=self.topology_cache, learn_gate=False,
                random_state=int(seed)+17)
        return self

    def decision_function(self, X, topology=None):
        X = np.asarray(X, float)
        if topology is None:
            topology = self.fuzzy_model.topology_evidence(X)
        sf, fs, rel = _fuzzy_features(self.fuzzy_model, X, topology)
        raw = np.column_stack([_scores(self.svm, X, "svm"),
            _scores(self.rf, X, "rf"), sf, fs, rel])
        ranked = _rank_transform(raw, self.rank_reference)
        return self.gate.decision_function(self.meta_scaler.transform(ranked))


def candidate_grids(n_features):
    return ([{"C": c, "gamma": g / n_features}
             for c in (.1, 1., 10.) for g in (.1, 1., 10.)],
            [{"min_leaf": leaf, "max_features": frac}
             for leaf in (1, 5, 20) for frac in (.5, .75, 1.)])
