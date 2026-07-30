"""Evaluation metrics for the dispersion surrogate.

- Liu & Weisberg (2011) skill score: THE standard trajectory metric since the
  Deepwater Horizon hindcasts (JGR 116, C09013, doi:10.1029/2010JC006837).
- IoU on the analysis grid: patch-shape agreement.
- Brier score: probabilistic cell-hit quality for summary-statistics models.
"""

from __future__ import annotations

import numpy as np

EARTH_KM_PER_DEG = 111.32


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km (inputs in degrees, arrays broadcast)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def liu_weisberg_ss(obs_lon, obs_lat, mod_lon, mod_lat, n: float = 1.0) -> float:
    """Liu-Weisberg skill score for one trajectory pair.

    Trajectories are same-length arrays sampled at the same instants
    (index 0 = common origin). s = sum(d_i) / sum(l_i), where d_i is the
    separation at step i and l_i the cumulative length of the OBSERVED
    trajectory up to step i. SS = max(0, 1 - s/n); 1 = perfect, 0 = worthless.
    """
    obs_lon = np.asarray(obs_lon, float)
    obs_lat = np.asarray(obs_lat, float)
    mod_lon = np.asarray(mod_lon, float)
    mod_lat = np.asarray(mod_lat, float)
    if obs_lon.shape != mod_lon.shape or obs_lon.ndim != 1:
        raise ValueError("expected matching 1-D trajectories")

    d = haversine_km(obs_lon, obs_lat, mod_lon, mod_lat)[1:]          # per step
    seg = haversine_km(obs_lon[:-1], obs_lat[:-1], obs_lon[1:], obs_lat[1:])
    l_cum = np.cumsum(seg)                                            # observed length
    denom = float(l_cum.sum())
    if denom == 0.0:
        # Stationary observed trajectory: SS undefined; call it perfect only
        # if the model is also (numerically) stationary on top of it.
        return 1.0 if float(d.max(initial=0.0)) < 1e-6 else 0.0
    s = float(d.sum()) / denom
    return float(max(0.0, 1.0 - s / n))


def centroid_error_km(obs_lon, obs_lat, mod_lon, mod_lat) -> float:
    """Distance between the centroids of two particle sets (deg in, km out)."""
    return float(haversine_km(np.nanmean(obs_lon), np.nanmean(obs_lat),
                              np.nanmean(mod_lon), np.nanmean(mod_lat)))


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two boolean grids. Empty∪empty -> 1.0."""
    a = np.asarray(mask_a, bool)
    b = np.asarray(mask_b, bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error between predicted probability and 0/1 outcome."""
    p = np.asarray(prob, float)
    o = np.asarray(outcome, float)
    return float(np.mean((p - o) ** 2))
