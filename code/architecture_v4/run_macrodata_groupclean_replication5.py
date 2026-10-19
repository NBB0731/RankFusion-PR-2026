"""Run the frozen OddBench protocol using unique, feature-group-disjoint rows."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
os.environ["ODDBENCH_PROTOCOL_FILE"] = "FROZEN_GROUPCLEAN_REPLICATION5_PROTOCOL.json"
os.environ["ODDBENCH_OUTPUT_FOLDER"] = "groupclean_replication5_predictions"
sys.path.insert(0, str(HERE))
import run_macrodata_confirmatory as base

BASE_PROTOCOL = base.P
PROTOCOL_HASH = base.PH
METADATA_HASH = base.MH
DATA = base.DATA
SEEDS = BASE_PROTOCOL["protocol"]["seeds"]
DATASETS = BASE_PROTOCOL["selection"]["selected"]
OUT = DATA / "groupclean_replication5_predictions"

def row_key(row):
    a = np.asarray(row, dtype="<f8").copy()
    a[np.isinf(a)] = np.nan
    a[a == 0] = 0.0
    a[np.isnan(a)] = np.float64(np.nan)
    return a.tobytes()

def unique_group_rows(X, labels=None):
    groups = {}
    for i, row in enumerate(X):
        k = row_key(row)
        if k not in groups:
            groups[k] = [i, set()]
        if labels is not None:
            groups[k][1].add(int(labels[i]))
    rows, ys, keys = [], [], []
    for k, (idx, labs) in groups.items():
        if labels is not None and len(labs) != 1:
            continue
        rows.append(X[idx])
        if labels is not None:
            ys.append(next(iter(labs)))
        keys.append(k)
    return np.asarray(rows, dtype=float), (np.asarray(ys, dtype=int) if labels is not None else None), keys

def prepare_group(path, seed):
    z = np.load(path, allow_pickle=False)
    tr = np.asarray(z["train"], float)
    te = np.asarray(z["test"], float)
    yt_all = np.asarray(z["test_labels"], int).ravel()
    tr[~np.isfinite(tr)] = np.nan
    te[~np.isfinite(te)] = np.nan
    ytr = np.asarray(z["train_labels"], int).ravel() if "train_labels" in z.files else np.zeros(len(tr), dtype=int)
    if np.any(ytr != 0):
        raise ValueError("Group-clean protocol requires an all-normal official training split")

    Xu_raw, _, train_keys = unique_group_rows(tr)
    test_label_sets = defaultdict(set)
    for row, label in zip(te, yt_all):
        test_label_sets[row_key(row)].add(int(label))
    conflict_groups = {k for k, labs in test_label_sets.items() if len(labs) > 1}
    test_groups, yt, test_keys = unique_group_rows(te, yt_all)
    train_key_set = set(train_keys)
    eligible = np.array([i for i, k in enumerate(test_keys) if k not in train_key_set], dtype=int)
    test_groups, yt = test_groups[eligible], yt[eligible]
    test_keys = [test_keys[i] for i in eligible]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xu_raw))
    n_u = int(np.floor(.8 * len(Xu_raw)))
    u = perm[:n_u]
    normal_pool = perm[n_u:]
    normal_cal = rng.choice(normal_pool, size=min(50, len(normal_pool)), replace=False)
    pos = np.flatnonzero(yt == 1)
    support = rng.choice(pos, size=10, replace=False)
    remaining = np.setdiff1d(np.arange(len(yt)), support, assume_unique=False)
    v, t = train_test_split(remaining, test_size=.5, random_state=seed, stratify=yt[remaining])

    yc = np.r_[np.zeros(len(normal_cal), dtype=int), np.ones(len(support), dtype=int)]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xu_raw[u])
    scaler = StandardScaler().fit(imp.transform(Xu_raw[u]))
    Xu = scaler.transform(imp.transform(Xu_raw[u]))
    Xc = scaler.transform(imp.transform(np.vstack([Xu_raw[normal_cal], test_groups[support]])))
    Xv = scaler.transform(imp.transform(test_groups[v]))
    Xt = scaler.transform(imp.transform(test_groups[t]))
    counts = {
        "n_train_rows_raw": int(len(tr)), "n_train_unique_groups": int(len(Xu_raw)),
        "n_test_rows_raw": int(len(te)), "n_test_conflict_groups_removed": int(len(conflict_groups)),
        "n_test_train_overlap_groups_removed": int(len(set(row_key(r) for r in te) & train_key_set)),
        "n_test_unique_groups_after_filter": int(len(test_groups)), "d": int(tr.shape[1]),
        "support_pos": int(len(support)), "calibration_normal": int(len(normal_cal)),
        "validation_n": int(len(v)), "validation_pos": int(yt[v].sum()),
        "test_n": int(len(t)), "test_pos": int(yt[t].sum()),
        "test_predictions_group_unique": True,
    }
    return Xu, Xc, yc, Xv, yt[v], Xt, yt[t], v, t, support, normal_cal, counts

base.prepare = prepare_group
base.SCRIPT = Path(__file__)
base.OUT = OUT
base.SH = hashlib.sha256(base.SCRIPT.read_bytes()).hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()
    if set(args.datasets) - set(DATASETS) or set(args.seeds) - set(SEEDS):
        raise ValueError("Requested run is outside the frozen group-clean protocol")
    with threadpool_limits(limits=2):
        for name in args.datasets:
            for seed in args.seeds:
                try:
                    base.run_one(name, int(seed))
                except Exception:
                    folder = OUT / name / str(seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    raise
    print("All group-disjoint prediction outputs are frozen; test metrics remain uncomputed.", flush=True)



