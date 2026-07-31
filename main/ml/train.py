"""First-generation transport surrogate: gradient-boosted trees.

Model: one HistGradientBoostingRegressor per target (dx_km, dy_km,
dspread_km). Evaluation: the SAME leave-one-block-out protocol as
baselines.py — a learned model only earns its place by beating persistence,
passive advection AND 1-NN under that protocol.

Reproducibility: fixed seed, dataset SHA-256, sklearn version and the full
metric table stored next to the model artefact.

Usage (repo root, opendrift env):
    python -m main.ml.train            # LOBO evaluation + final fit + save
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.ml.baselines import (  # noqa: E402
    leave_one_block_out as baseline_lobo,
    mae_km,
    predict_advection,
)

DATASET = ROOT / "main" / "outputs" / "ml" / "patch_dataset.npz"
MODEL_OUT = ROOT / "main" / "outputs" / "ml" / "surrogate_hgb.joblib"
META_OUT = ROOT / "main" / "outputs" / "ml" / "surrogate_hgb.json"
MODEL_RES_OUT = ROOT / "main" / "outputs" / "ml" / "surrogate_hgb_residual.joblib"
META_RES_OUT = ROOT / "main" / "outputs" / "ml" / "surrogate_hgb_residual.json"
SEED = 42

HGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_depth=None,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=SEED,
)


def fit_model(X: np.ndarray, Y: np.ndarray) -> list:
    """One booster per target dimension."""
    models = []
    for k in range(Y.shape[1]):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X, Y[:, k])
        models.append(m)
    return models


def predict(models: list, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)


def lobo_evaluate(X, Y, blocks, base: np.ndarray | None = None) -> dict:
    """Leave-one-block-out MAE for the learned model.

    With ``base`` (residual mode), the model is trained on Y - base and the
    evaluated prediction is base + model(X): the surrogate learns only the
    CORRECTION over passive advection, so a useless correction degrades to
    the advection baseline instead of compounding its own bias in rollout
    (the failure mode the 2024 blind test exposed for the direct model).
    """
    target = Y if base is None else Y - base
    rows = []
    for held in sorted(set(blocks.tolist())):
        te = blocks == held
        models = fit_model(X[~te], target[~te])
        pred = predict(models, X[te])
        if base is not None:
            pred = pred + base[te]
        rows.append(mae_km(Y[te], pred))
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import sklearn

    p = argparse.ArgumentParser(description="Train the transport surrogate.")
    p.add_argument("--residual", action="store_true",
                   help="Learn the correction over passive advection instead "
                        "of the full displacement (robust in rollout).")
    args = p.parse_args()

    d = np.load(DATASET)
    X, Y, blocks = d["X"], d["Y"], d["block"]
    dt = float(d["dt_hours"])
    ds_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()[:16]

    adv = predict_advection(X, dt) if args.residual else None
    mode = "residual (advecção + correção)" if args.residual else "direto"
    print(f"Treino surrogate HGB [{mode}] — {len(X)} amostras, "
          f"{len(set(blocks.tolist()))} blocos, DT={dt:.0f} h, seed={SEED}")

    print("\n[1/3] Baselines (leave-one-block-out)…", flush=True)
    base = baseline_lobo(X, Y, blocks, dt)

    print("[2/3] Surrogate (leave-one-block-out)…", flush=True)
    model_m = lobo_evaluate(X, Y, blocks, base=adv)

    hdr = f"{'modelo':12s} {'err_desloc(km)':>14s} {'mae_dspread':>12s}"
    print("\n" + hdr)
    for name, m in list(base.items()) + [("HGB", model_m)]:
        print(f"{name:12s} {m['mean_disp_err_km']:14.2f} {m['mae_dspread']:12.2f}")

    best_base = min(v["mean_disp_err_km"] for v in base.values())
    beats = model_m["mean_disp_err_km"] < best_base
    print(f"\nSurrogate {'VENCE' if beats else 'NAO vence'} o melhor baseline "
          f"({model_m['mean_disp_err_km']:.2f} vs {best_base:.2f} km).")

    print("\n[3/3] Fit final em todos os blocos + salvando artefato…", flush=True)
    models = fit_model(X, Y - adv if adv is not None else Y)
    out_model = MODEL_RES_OUT if args.residual else MODEL_OUT
    out_meta = META_RES_OUT if args.residual else META_OUT
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_model)
    out_meta.write_text(json.dumps({
        "mode": "residual_over_advection" if args.residual else "direct",
        "trained": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256_16": ds_hash,
        "n_samples": int(len(X)),
        "n_blocks": int(len(set(blocks.tolist()))),
        "dt_hours": dt,
        "seed": SEED,
        "sklearn": sklearn.__version__,
        "params": HGB_PARAMS,
        "features": d["feature_names"].tolist(),
        "targets": d["target_names"].tolist(),
        "lobo_metrics": {"baselines": base, "hgb": model_m},
        "beats_best_baseline": bool(beats),
    }, indent=2))
    print(f"[OK] modelo -> {out_model.name} | metadados -> {out_meta.name}")


if __name__ == "__main__":
    main()
