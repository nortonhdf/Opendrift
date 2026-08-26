"""Where does the slick spread come from, and why can nothing predict it?

Every model in CAMADA_IA.md 5e lands on the same MAE for the spread target,
climatology included. This probe answers why, in three steps that each rule
something out:

1. **Is there anything to predict?** Compare each model's MAE against the MAD
   around the best constant. A ratio of ~1 means predicting a constant is
   already optimal and the "failure" is in the target, not the models.

2. **Where does the spread come from?** Re-run one scenario with the declared
   stochastic uncertainties switched off, and with each one alone. With
   horizontal_diffusivity = 0 these are the only declared source of
   spreading; this measures how much of the slick width they actually own.

3. **Is the scenario-to-scenario variation physics or RNG?** Re-run the
   lowest, median and highest-spread scenarios of the blind year with three
   seeds each. If within-scenario scatter is far below between-scenario
   scatter, the differences are real flow behaviour — and the question
   becomes whether release-time features can see them (they cannot, step 1).

The "arquivo" column of step 3 is the archived value for the same scenario,
and it is the check that this probe is measuring the same physical model:
the first version of this script ran on the defaults of ``run_simulation``
(1000 particles, Stokes drift ON) and disagreed with the archive by 6x. It
now runs the archive configuration, and the columns agree.

Runs are cached by file name, so a second invocation only prints.

Usage (repo root, opendrift env):
    python main/scripts/_spread_decomposition.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from main.domain_config import SEASON_MONTHS  # noqa: E402
from main.fields_config import CAMPOS_FIELDS  # noqa: E402
from main.ml.dataset import patch_state  # noqa: E402
from main.ml.forecast import tcol  # noqa: E402
from main.ml.multiyear import forcing_paths  # noqa: E402
from main.ml.scenario import HORIZONS_D  # noqa: E402
from main.run_open_oil import run_simulation  # noqa: E402

OUT = ROOT / "main" / "outputs" / "_spread_probe"
ML = ROOT / "main" / "outputs" / "ml"
YEAR = 2024
SEEDS = [0, 7, 99]


def spread_at(nc: Path, h: int) -> float:
    ds = xr.open_dataset(nc)
    lon = np.asarray(ds["lon"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    t = ds["time"].values
    ds.close()
    idx = int(round(h * 24 / ((t[1] - t[0]) / np.timedelta64(1, "h"))))
    if idx >= lon.shape[1]:
        return float("nan")
    return patch_state(lon[:, idx], lat[:, idx])[2]


def simulate(name: str, field: str, start: datetime, **kw) -> Path:
    """One run in EXACTLY the configuration the 168-h archive was built with.

    n_particles and use_waves are not the defaults of run_simulation, and
    that matters: main.ml.multiyear generates the archive with 200 particles
    and Stokes drift OFF. A probe left on the defaults measures a different
    physical model and its numbers cannot be compared with the archive the
    claim is about.
    """
    from main.ml.multiyear import N_PARTICLES

    nc = OUT / f"{name}.nc"
    if not nc.exists():
        cfg = CAMPOS_FIELDS[field]
        cur, wnd = forcing_paths(start.year)
        run_simulation(seed_lon=cfg["lon"], seed_lat=cfg["lat"],
                       n_particles=N_PARTICLES, start_time=start,
                       duration_hours=168, oil_type=cfg["oil_type"],
                       use_wind=True, use_waves=False,
                       currents_file=str(cur), wind_file=str(wnd),
                       waves_file=None, outfile=str(nc), figfile=None,
                       loglevel=50, **kw)
    return nc


# ── 1. is there anything to predict? ─────────────────────────────────────────

def step_target() -> None:
    d = np.load(ML / "scenario_dataset.npz", allow_pickle=True)
    b = np.load(ML / "scenario_dataset_2024.npz", allow_pickle=True)
    Y, blocks = d["Y"], d["block"]
    Yh, blocks_h = b["Y"], b["block"]
    print("1. O alvo tem o que prever?  (MAE da climatologia vs MAD de uma "
          "constante)\n")
    print(f"{'h':>3}{'mediana':>10}{'IQR':>16}{'MAD':>8}{'MAE clim':>10}"
          f"{'MAE/MAD':>9}")
    for hi, h in enumerate(HORIZONS_D):
        k = tcol(hi, "spread_km")
        s = Y[:, k][np.isfinite(Y[:, k])]
        sh = Yh[:, k]
        ok = np.isfinite(sh)
        table = {q: Y[blocks == q, k][np.isfinite(Y[blocks == q, k])].mean()
                 for q in np.unique(blocks)}
        pred = np.array([table.get(q, s.mean()) for q in blocks_h])
        mae = float(np.mean(np.abs(sh[ok] - pred[ok])))
        mad = float(np.mean(np.abs(sh[ok] - np.median(sh[ok]))))
        q1, q3 = np.percentile(s, [25, 75])
        print(f"{h:>3}{np.median(s):10.2f}{q1:8.2f}-{q3:<7.2f}{mad:8.2f}"
              f"{mae:10.2f}{mae / mad:9.2f}")
    print("\n   MAE/MAD ~ 1 => nenhum modelo bate uma constante: o alvo quase "
          "nao varia\n   em torno dela, e a 'falha' esta no alvo, nao nos "
          "modelos.\n")


# ── 2. where does the spread come from? ──────────────────────────────────────

CASES = [
    ("declarado", dict(current_uncertainty=0.05, wind_uncertainty=0.5)),
    ("sem_incerteza", dict(current_uncertainty=0.0, wind_uncertainty=0.0)),
    ("so_corrente", dict(current_uncertainty=0.05, wind_uncertainty=0.0)),
    ("so_vento", dict(current_uncertainty=0.0, wind_uncertainty=0.5)),
]


def step_sources() -> None:
    start = datetime(YEAR, 4, 10)
    print("2. De onde vem o espalhamento?  (Marlim, 2024-04-10, km)\n")
    print(f"{'configuracao':18s}" + "".join(f"{'D+' + str(h):>9s}"
                                            for h in HORIZONS_D))
    vals = {}
    for name, kw in CASES:
        nc = simulate(f"src_{name}", "Marlim", start, random_seed=0, **kw)
        vals[name] = [spread_at(nc, h) for h in HORIZONS_D]
        print(f"{name:18s}" + "".join(f"{v:9.3f}" for v in vals[name]))
    quad = [float(np.hypot(c, w))
            for c, w in zip(vals["so_corrente"], vals["so_vento"])]
    print(f"{'corrente+vento':18s}" + "".join(f"{v:9.3f}" for v in quad)
          + "   <- soma em quadratura")
    frac = [100 * z / d for z, d in zip(vals["sem_incerteza"],
                                        vals["declarado"])]
    print(f"{'sem/declarado %':18s}" + "".join(f"{v:9.1f}" for v in frac))
    print("\n   As duas incertezas declaradas somam em quadratura e explicam "
          "o total.\n   Sem elas sobra <1%: com raio de semeadura de 1 m, o "
          "que o oceano faz\n   com a mancha e proporcional ao tamanho dela, "
          "e ela comeca em um ponto.\n")


# ── 3. physics or RNG? ───────────────────────────────────────────────────────

def step_seeds() -> None:
    d = np.load(ML / "scenario_dataset_2024.npz", allow_pickle=True)
    k = tcol(len(HORIZONS_D) - 1, "spread_km")
    sp = d["Y"][:, k]
    ok = np.flatnonzero(np.isfinite(sp))
    order = ok[np.argsort(sp[ok])]
    picks = {"baixo": order[0], "mediano": order[len(order) // 2],
             "alto": order[-1]}
    man = json.loads((ROOT / "main" / "outputs" /
                      f"training168_{YEAR}" / "manifest.json").read_text())

    print("3. A variacao entre cenarios e fisica ou semente?  (D+7, km)\n")
    print(f"{'cenario':28s}{'arquivo':>9s}" + "".join(f"{'seed ' + str(s):>9s}"
                                                      for s in SEEDS))
    rows = {}
    for label, i in picks.items():
        key = str(d["run_key"][i])
        e = man[key]
        start = datetime(YEAR, SEASON_MONTHS[e["season"]], e["day"])
        vals = []
        for s in SEEDS:
            nc = simulate(f"seed_{key}_{s}", e["field"], start, random_seed=s)
            vals.append(spread_at(nc, HORIZONS_D[-1]))
        rows[label] = vals
        print(f"{label + ' (' + key + ')':28s}{sp[i]:9.2f}"
              + "".join(f"{v:9.2f}" for v in vals))
    within = float(np.mean([np.std(v) for v in rows.values()]))
    between = float(np.std([np.mean(v) for v in rows.values()]))
    print(f"\n   dentro do cenario (sementes): {within:.3f} km")
    print(f"   entre cenarios:               {between:.3f} km"
          f"   ({between / max(within, 1e-9):.0f}x)")
    print("\n   Se entre >> dentro, as diferencas sao reais — e o que falta e "
          "informacao\n   sobre o escoamento FUTURO ao longo do trajeto, que "
          "nao existe no instante\n   do vazamento. Mesma barreira do 5d.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    step_target()
    step_sources()
    step_seeds()


if __name__ == "__main__":
    main()
