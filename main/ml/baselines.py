"""Baselines for the patch-transport surrogate + block-wise evaluation.

Any learned model must beat ALL of these, evaluated with leave-one-block-out
(block = field x season), before it earns a place in the project:

  persistence : the patch does not move            (dx = dy = dspread = 0)
  advection   : passive drift with local forcing   (currents + 3% wind, DT)
  nearest     : copy the target of the most similar training sample
                (brute-force nearest neighbour in standardised feature space)

Usage (repo root, opendrift env):
    python -m main.ml.baselines        # evaluates main/outputs/ml/patch_dataset.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "main" / "outputs" / "ml" / "patch_dataset.npz"
WIND_DRIFT_FACTOR = 0.03      # matches seed:wind_drift_factor in the runs
KM_PER_H_PER_MS = 3.6         # 1 m/s sustained for 1 h = 3.6 km


def predict_persistence(X: np.ndarray) -> np.ndarray:
    return np.zeros((len(X), 3), np.float32)


def predict_advection(X: np.ndarray, dt_hours: float) -> np.ndarray:
    """Passive transport: currents + 3% wind over the horizon; spread frozen."""
    u = X[:, 4] + WIND_DRIFT_FACTOR * X[:, 6]
    v = X[:, 5] + WIND_DRIFT_FACTOR * X[:, 7]
    out = np.zeros((len(X), 3), np.float32)
    out[:, 0] = u * KM_PER_H_PER_MS * dt_hours
    out[:, 1] = v * KM_PER_H_PER_MS * dt_hours
    return out


def predict_nearest(X_train, Y_train, X_test) -> np.ndarray:
    """1-NN in z-scored feature space (brute force — fine below ~1e5 rows)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    A = (X_train - mu) / sd
    B = (X_test - mu) / sd
    out = np.empty((len(B), Y_train.shape[1]), np.float32)
    for i, b in enumerate(B):
        j = int(np.argmin(((A - b) ** 2).sum(axis=1)))
        out[i] = Y_train[j]
    return out


def mae_km(Y_true, Y_pred) -> dict:
    err = np.abs(Y_true - Y_pred)
    disp = np.sqrt((Y_true[:, 0] - Y_pred[:, 0]) ** 2
                   + (Y_true[:, 1] - Y_pred[:, 1]) ** 2)
    return {"mae_dx": float(err[:, 0].mean()),
            "mae_dy": float(err[:, 1].mean()),
            "mean_disp_err_km": float(disp.mean()),
            "mae_dspread": float(err[:, 2].mean())}


def leave_one_block_out(X, Y, blocks, dt_hours: float) -> dict:
    """Evaluate each baseline with strict block-wise splits."""
    names = sorted(set(blocks.tolist()))
    agg = {b: [] for b in ["persistence", "advection", "nearest"]}
    for held in names:
        te = blocks == held
        tr = ~te
        preds = {
            "persistence": predict_persistence(X[te]),
            "advection": predict_advection(X[te], dt_hours),
            "nearest": predict_nearest(X[tr], Y[tr], X[te]),
        }
        for k, p in preds.items():
            agg[k].append(mae_km(Y[te], p))
    out = {}
    for k, rows in agg.items():
        out[k] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = np.load(DATASET)
    X, Y, blocks = d["X"], d["Y"], d["block"]
    dt = float(d["dt_hours"])
    print(f"Baselines (leave-one-block-out, {len(set(blocks.tolist()))} blocos, "
          f"{len(X)} amostras, DT={dt:.0f} h)\n")
    res = leave_one_block_out(X, Y, blocks, dt)
    hdr = f"{'baseline':12s} {'err_desloc(km)':>14s} {'mae_dx':>8s} {'mae_dy':>8s} {'mae_dspread':>12s}"
    print(hdr)
    for k, m in res.items():
        print(f"{k:12s} {m['mean_disp_err_km']:14.2f} {m['mae_dx']:8.2f} "
              f"{m['mae_dy']:8.2f} {m['mae_dspread']:12.2f}")
    print("\nRegra do projeto: um modelo aprendido so entra se vencer TODOS "
          "os baselines acima nesta mesma avaliacao por blocos.")


if __name__ == "__main__":
    main()
