"""Single source of truth for shared spatial/temporal constants.

The audit (finding #11) found the grid domain, resolution and season dates
independently re-declared in compute_risk_grids, compute_beaching, app and
precompute_scenarios — they agreed by discipline only. Everything that must
stay consistent across download, batch and app lives here.
"""

from __future__ import annotations

from datetime import datetime

# ── Forcing download box (audit-approved 2026-07-29; the old 3.5°×3.5° box
#    lost ~16% of particles over its boundary within 120 h) ────────────────
FORCING_LON_MIN, FORCING_LON_MAX = -45.0, -36.0
FORCING_LAT_MIN, FORCING_LAT_MAX = -27.0, -19.0

# ── Analysis grid for risk/beaching products (matches the forcing box) ─────
GRID_LON_MIN, GRID_LON_MAX = FORCING_LON_MIN, FORCING_LON_MAX
GRID_LAT_MIN, GRID_LAT_MAX = FORCING_LAT_MIN, FORCING_LAT_MAX
GRID_RES = 0.1  # degrees

# ── Seasons: four representative months of the forcing year ────────────────
FORCING_YEAR = 2025
SEASON_MONTHS = {"jan": 1, "apr": 4, "jul": 7, "oct": 10}
SEASONS = list(SEASON_MONTHS)

# Ensemble start days: one member per day, 1..28 (28 members/month).
# The audit convergence experiment (docs/auditoria/REGENERACAO.md) showed the
# old 10-member ensemble was NOT converged (IoU 0.63-0.73 for 5-vs-10
# prob_any); 28 daily members sits in the recommended >=20-30 range.
ENSEMBLE_DAYS = list(range(1, 29))


def season_date(season: str, year: int = FORCING_YEAR, day: int = 15) -> datetime:
    """Canonical scenario start date for a season key ('jan'/'apr'/'jul'/'oct')."""
    return datetime(year, SEASON_MONTHS[season], day)


def ensemble_dates(season: str, year: int = FORCING_YEAR) -> list[datetime]:
    """The member start dates for one season."""
    m = SEASON_MONTHS[season]
    return [datetime(year, m, d) for d in ENSEMBLE_DAYS]
