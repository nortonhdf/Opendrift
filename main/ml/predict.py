"""One forecast for one release — the entry point the app tab calls.

Everything the ML layer proved is behind two artefacts and a feature vector:

    forecast_product.joblib   centroid displacement + conformal band  (§5e)
    footprint_plume.joblib    the shape drawn around that path        (§5f)

This module holds them open, builds the feature row with the SAME function
the training set was built with (main.ml.scenario.feature_row — that is the
point of it), and returns something a map can draw.

What it will NOT do quietly:

- forecast outside the forcing box, or for a year with no forcing file. The
  antecedent features come from those files; without them there is no
  forecast, and inventing one is worse than refusing.
- hide a short lookback. A release in early January has no 90-day window
  inside its year, so those features are NaN — HGB consumes NaN natively and
  the forecast is still valid, but it is a thinner forecast and the caller is
  told so.

Usage:
    from main.ml.predict import Predictor
    p = Predictor()
    out = p.forecast(lon=-40.1, lat=-22.4, api=28.0, water_depth_m=1000,
                     when=datetime(2024, 3, 10))
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.domain_config import (  # noqa: E402
    FORCING_LAT_MAX, FORCING_LAT_MIN, FORCING_LON_MAX, FORCING_LON_MIN,
    SEASON_MONTHS,
)
from main.ml import forecast as fc  # noqa: E402
from main.ml import footprint_forecast as ff  # noqa: E402
from main.ml.dataset import KM_PER_DEG  # noqa: E402
from main.ml.multiyear import forcing_paths  # noqa: E402
from main.ml.scenario import AntecedentSampler, HORIZONS_D, feature_row  # noqa: E402

SEARCH_YEARS = range(2015, 2036)


def available_years() -> list:
    """Years with a currents file — the only ones a forecast can stand on."""
    return [y for y in SEARCH_YEARS if forcing_paths(y)[0].exists()]


def season_of(month: int) -> str:
    """Nearest of the four representative months, cyclically.

    The footprint carries one climatology per season, used only where the
    predicted displacement is too small to define a frame. Picking the
    nearest month keeps that fallback meaningful for a release in, say,
    November.
    """
    def gap(m):
        d = abs(m - month)
        return min(d, 12 - d)
    return min(SEASON_MONTHS, key=lambda s: gap(SEASON_MONTHS[s]))


def offset_to_lonlat(lon0: float, lat0: float, dx_km, dy_km):
    """Km offsets in the release frame back to degrees."""
    return (lon0 + np.asarray(dx_km) / (KM_PER_DEG * np.cos(np.radians(lat0))),
            lat0 + np.asarray(dy_km) / KM_PER_DEG)


def in_domain(lon: float, lat: float) -> bool:
    return (FORCING_LON_MIN <= lon <= FORCING_LON_MAX
            and FORCING_LAT_MIN <= lat <= FORCING_LAT_MAX)


class Predictor:
    """Products + forcing samplers, held open across calls."""

    def __init__(self, years=None):
        import joblib

        for path, how in ((fc.PRODUCT, "python -m main.ml.forecast --export"),
                          (ff.PRODUCT,
                           "python -m main.ml.footprint_forecast --export")):
            if not path.exists():
                raise FileNotFoundError(
                    f"Produto ausente: {path.name}. Gere com: {how}")
        self.fc = joblib.load(fc.PRODUCT)
        self.fp = joblib.load(ff.PRODUCT)
        self.years = list(years) if years else available_years()
        if not self.years:
            raise FileNotFoundError(
                "Nenhum arquivo de correntes em main/inputs/ — sem forçante "
                "não há features antecedentes e portanto não há previsão.")
        self._sampler = AntecedentSampler(self.years)
        self.horizons = list(self.fc["horizons_d"])

    def close(self):
        self._sampler.close()

    # ── features ─────────────────────────────────────────────────────────────

    def features(self, lon, lat, api, water_depth_m, when) -> np.ndarray:
        year = when.year if isinstance(when, datetime) else int(str(when)[:4])
        if year not in self.years:
            raise ValueError(
                f"{year} não tem forçante em main/inputs/ (disponíveis: "
                f"{self.years}). Baixe com main/scripts/download_*.py.")
        if not in_domain(lon, lat):
            raise ValueError(
                f"({lat:.2f}, {lon:.2f}) está fora da caixa de forçante "
                f"lon {FORCING_LON_MIN}..{FORCING_LON_MAX} / "
                f"lat {FORCING_LAT_MIN}..{FORCING_LAT_MAX}.")
        return feature_row(lon, lat, api, water_depth_m,
                           np.datetime64(when), self._sampler)

    def _coverage_warnings(self, x: np.ndarray) -> list:
        """Which antecedent windows fell outside the forcing year."""
        names = list(self.fc["feature_names"])
        out = []
        for n in names:
            if n.startswith("coverage_") and np.isfinite(x[names.index(n)]):
                cov = float(x[names.index(n)])
                if cov < 0.99:
                    out.append(f"janela de {n.split('_')[1]} coberta "
                               f"{cov:.0%} — a feature entra como NaN")
        return out

    # ── the forecast ─────────────────────────────────────────────────────────

    def forecast(self, lon: float, lat: float, api: float,
                 water_depth_m: float, when, model: str = None) -> dict:
        """Predicted track with its calibrated band, plus the footprint field.

        The band is on the DISTANCE travelled (that is what the conformal
        correction was calibrated on), so it is a range along the predicted
        bearing — never a disc around the predicted point. What carries the
        spatial uncertainty is the footprint probability field.
        """
        x = self.features(lon, lat, api, water_depth_m, when)
        season = season_of(when.month if isinstance(when, datetime)
                           else int(str(when)[5:7]))
        band = fc.predict_scenario(self.fc, x)

        track, foot = {}, {}
        for h in self.horizons:
            b = band[h]
            plon, plat = offset_to_lonlat(lon, lat, b["dx_km"], b["dy_km"])
            track[h] = {**b, "lon": float(plon), "lat": float(plat)}
            f = ff.predict_footprint(self.fp, self.fc, x, h, lon, lat, season,
                                     model=model)
            foot[h] = {"lon": f["lon"], "lat": f["lat"], "prob": f["prob"],
                       "threshold": f["threshold"], "model": f["model"]}
        return {
            "features": x, "season": season,
            "release": {"lon": lon, "lat": lat, "api": api,
                        "water_depth_m": water_depth_m, "when": when},
            "track": track, "footprint": foot,
            "warnings": self._coverage_warnings(x),
            "horizons_d": self.horizons,
        }


def main() -> None:
    """Smoke check from the command line: one release, printed."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse

    ap = argparse.ArgumentParser(description="Forecast one release.")
    ap.add_argument("--lon", type=float, default=-40.1)
    ap.add_argument("--lat", type=float, default=-22.4)
    ap.add_argument("--api", type=float, default=28.0)
    ap.add_argument("--depth", type=float, default=1000.0)
    ap.add_argument("--date", default="2024-03-10")
    a = ap.parse_args()

    p = Predictor()
    out = p.forecast(a.lon, a.lat, a.api, a.depth,
                     datetime.fromisoformat(a.date))
    print(f"Vazamento em ({a.lat}, {a.lon}) em {a.date} — estação de "
          f"referência {out['season']}")
    for w in out["warnings"]:
        print(f"  [aviso] {w}")
    print(f"{'':6s}{'distância (km)':>22s}   {'posição prevista':>20s}"
          f"   {'células p≥limiar':>18s}")
    for h in out["horizons_d"]:
        t = out["track"][h]
        f = out["footprint"][h]
        n = int((f["prob"] >= f["threshold"]).sum())
        print(f"D+{h:<4d}{t['dist_km']:8.0f} "
              f"[{t['dist_lo_km']:.0f}–{t['dist_hi_km']:.0f}]"
              f"{'':>4s}{t['lat']:8.2f}, {t['lon']:.2f}{n:15d}")
    p.close()


if __name__ == "__main__":
    main()
