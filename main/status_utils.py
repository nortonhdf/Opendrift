"""Decode OpenDrift element-status flags safely.

OpenDrift records the meaning of the integer ``status`` codes *per output
file*, in the ``flag_values`` / ``flag_meanings`` attributes of the ``status``
variable, in order of first occurrence within that run. The same code can
therefore mean ``stranded`` in one file and ``missing_data`` in another —
codes must NEVER be hard-coded (audit finding #1, docs/auditoria/DIAGNOSTICO.md).
The only invariant is that ``active`` is always 0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ACTIVE = "active"
STRANDED = "stranded"
MISSING_DATA = "missing_data"


def status_map(status_var) -> dict[int, str]:
    """Return {code: meaning} for one dataset's ``status`` variable.

    ``status_var`` is an xarray DataArray (or anything with ``.attrs``), or a
    plain attrs dict. Falls back to {0: 'active'} when the attributes are
    absent (a run in which no element was ever deactivated).
    """
    attrs = status_var if isinstance(status_var, dict) else status_var.attrs
    meanings = attrs.get("flag_meanings")
    if not meanings:
        return {0: ACTIVE}
    names = str(meanings).split()
    values = attrs.get("flag_values")
    if values is None:
        codes = list(range(len(names)))
    else:
        codes = np.atleast_1d(np.asarray(values)).astype(int).tolist()
    return dict(zip(codes, names))


def code_of(status_var, meaning: str) -> Optional[int]:
    """Return the integer code for ``meaning`` in this file, or None."""
    for code, name in status_map(status_var).items():
        if name == meaning:
            return code
    return None


def last_valid_index(lon: np.ndarray) -> np.ndarray:
    """Per-particle index of the last finite position; -1 if none.

    ``lon`` has shape (n_particles, n_timesteps). OpenDrift pads positions
    with NaN after an element is deactivated, so the last finite index is the
    step at which the element was last alive (its stranding point, for
    stranded elements).
    """
    valid = np.isfinite(lon)
    nt = lon.shape[1]
    any_valid = valid.any(axis=1)
    last = nt - 1 - np.argmax(valid[:, ::-1], axis=1)
    return np.where(any_valid, last, -1)


def final_status(lon: np.ndarray, status_values: np.ndarray,
                 smap: dict[int, str]) -> np.ndarray:
    """Per-particle status meaning at its last valid timestep.

    Returns an object array of meaning strings ('active', 'stranded',
    'missing_data', ... or 'none' for particles with no valid position).
    """
    idx = last_valid_index(lon)
    n = lon.shape[0]
    out = np.full(n, "none", dtype=object)
    ok = idx >= 0
    codes = status_values[np.arange(n)[ok], idx[ok]].astype(int)
    out[ok] = [smap.get(c, f"code{c}") for c in codes]
    return out


def final_status_counts(ds) -> dict[str, int]:
    """Count particles by final status meaning for an OpenDrift output dataset."""
    lon = np.asarray(ds["lon"].values, dtype=float)
    smap = status_map(ds["status"])
    finals = final_status(lon, ds["status"].values, smap)
    meanings, counts = np.unique(finals.astype(str), return_counts=True)
    return dict(zip(meanings.tolist(), counts.tolist()))
