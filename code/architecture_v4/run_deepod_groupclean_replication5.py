"""CPU deep few-shot baselines for the prospectively frozen replication-4 split.

Only support anomalies and unlabeled/normal reference rows enter fitting.
Validation labels are used only for model selection; test labels are sealed
until all baseline and deep predictions have been frozen and integrity-audited.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PKG = HERE / "baseline_packages"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
import run_macrodata_groupclean_replication5 as gc
from deepod.models.tabular import PReNet, DevNet

OUT = HERE / "macrodata_confirmatory" / "deepod_groupclean_replication5_predictions"
DATASETS = gc.DATASETS
SEEDS = gc.SEEDS
EPOCHS = 30
BATCH = 64

def fit_predict(name, seed):
    Xu, Xc, yc, Xv, yv, Xt, yt, *_ = gc.prepare_group(gc.DATA / "public" / (name + ".npz"), seed)
    # Xc contains the frozen 50-normal calibration subset followed by 10 known
    # anomalies. Both weakly supervised methods use 0=unlabeled and 1=anomaly.
    Xfit = np.vstack([Xu, Xc])
    yfit = np.r_[np.zeros(len(Xu), dtype=np.int64), yc.astype(np.int64)]
    models = {
        "PReNet": PReNet(epochs=EPOCHS, batch_size=BATCH, device="cpu", verbose=0,
                         random_state=int(seed), prt_steps=EPOCHS),
        "DevNet": DevNet(epochs=EPOCHS, batch_size=BATCH, device="cpu", verbose=0,
                         random_state=int(seed), prt_steps=EPOCHS),
    }
    scores = {}
    fit_secs = {}
    pred_secs = {}
    for model_name, model in models.items():
        t0 = time.perf_counter()
        model.fit(Xfit.astype(np.float32), yfit)
        fit_secs[model_name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        val = np.asarray(model.decision_function(Xv.astype(np.float32)), float).ravel()
        test = np.asarray(model.decision_function(Xt.astype(np.float32)), float).ravel()
        pred_secs[model_name] = time.perf_counter() - t0
        if len(test) != len(yt) or not np.all(np.isfinite(test)):
            raise RuntimeError(f"Bad scores: {model_name}, {name}, {seed}")
        scores[model_name] = {"validation": val, "test": test}
    return scores, {"fit_seconds": fit_secs, "val_test_inference_seconds": pred_secs,
                    "n_fit_unlabeled": int(len(Xu) + (yc == 0).sum()),
                    "n_known_anomaly_support": int((yc == 1).sum()),
                    "n_validation": int(len(yv)), "n_test": int(len(yt)),
                    "epochs": EPOCHS, "batch_size": BATCH, "device": "cpu"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    if set(args.datasets) - set(DATASETS) or set(args.seeds) - set(SEEDS):
        raise ValueError("Requested run outside frozen group-clean cohort")
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in args.seeds:
            folder = OUT / dataset / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / "predictions.npz"
            if output.exists():
                continue
            try:
                with threadpool_limits(limits=2):
                    scores, metadata = fit_predict(dataset, seed)
                np.savez_compressed(output,
                    prenet_validation=scores["PReNet"]["validation"],
                    prenet_test=scores["PReNet"]["test"],
                    devnet_validation=scores["DevNet"]["validation"],
                    devnet_test=scores["DevNet"]["test"])
                metadata["prediction_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                metadata["protocol_sha256"] = gc.PROTOCOL_HASH
                metadata["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print(f"DONE {dataset} seed={seed} fit={metadata['fit_seconds']}", flush=True)
            except Exception:
                (folder / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
    print("DeepOD predictions frozen; no AP scoring was run by this script.", flush=True)

if __name__ == "__main__":
    main()



