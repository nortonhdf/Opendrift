"""
Campos Basin Oil Spill Dispersion — Streamlit App

Run with:
    cd C:\\Users\\nbkon\\Git\\opendrift
    conda activate opendrift
    streamlit run main/app.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.fields_config import CAMPOS_FIELDS
from main.run_open_oil import run_simulation

SCENARIOS_DIR = ROOT / "main" / "outputs" / "scenarios"
MANIFEST_PATH = SCENARIOS_DIR / "manifest.json"

RISK_DIR  = ROOT / "main" / "outputs" / "risk_grids"
GRID_RES  = 0.1  # degrees — matches compute_risk_grids.py

SEASON_LABELS = {"jan": "January", "apr": "April", "jul": "July", "oct": "October"}
SEASON_DATES  = {
    "jan": datetime(2025, 1, 15),
    "apr": datetime(2025, 4, 15),
    "jul": datetime(2025, 7, 15),
    "oct": datetime(2025, 10, 15),
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Campos Basin Oil Spill Dispersion",
    page_icon="🛢",
    layout="wide",
)

st.title("🛢 Campos Basin — Oil Spill Dispersion Model")
st.caption("Powered by OpenDrift / OpenOil · ERA5 wind & waves · CMEMS currents")

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_nc(path: str) -> dict:
    ds  = xr.open_dataset(path)
    out = {
        "lon":    ds["lon"].values,
        "lat":    ds["lat"].values,
        "status": ds["status"].values,
        "times":  ds["time"].values,
    }
    ds.close()
    return out


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def get_data_range(nc_path: str) -> tuple[datetime | None, datetime | None]:
    try:
        ds    = xr.open_dataset(nc_path)
        t     = ds["time"].values
        start = datetime.utcfromtimestamp(int(t[0].astype("int64")) // 1_000_000_000)
        end   = datetime.utcfromtimestamp(int(t[-1].astype("int64")) // 1_000_000_000)
        ds.close()
        return start, end
    except Exception:
        return None, None


def build_animated_figure(data: dict, field: dict, field_name: str, show_density: bool) -> go.Figure:
    lons   = data["lon"]      # (particles, timesteps)
    lats   = data["lat"]
    status = data["status"]
    times  = data["times"]

    n_p, n_t = lons.shape

    # Subsample particles for display performance
    sample = min(n_p, 300)
    rng    = np.random.default_rng(42)
    idx    = rng.choice(n_p, sample, replace=False)

    # Subsample timesteps for animation (max 60 frames)
    step   = max(1, n_t // 60)
    t_idxs = list(range(0, n_t, step))

    # ── Static trajectory lines ──
    traces = []
    for i in idx:
        traces.append(go.Scattermapbox(
            lon=lons[i].tolist(), lat=lats[i].tolist(),
            mode="lines",
            line=dict(color="rgba(80,120,200,0.15)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # ── Seed marker ──
    traces.append(go.Scattermapbox(
        lon=[field["lon"]], lat=[field["lat"]],
        mode="markers",
        marker=dict(size=14, color="#4CAF50"),
        name=f"{field_name} (seed)",
    ))

    # ── Density heatmap (final positions, all particles) ──
    if show_density:
        active_mask = status[:, -1] == 0
        traces.append(go.Densitymapbox(
            lon=lons[active_mask, -1].tolist(),
            lat=lats[active_mask, -1].tolist(),
            radius=18,
            colorscale="YlOrRd",
            opacity=0.6,
            showscale=True,
            name="Density",
        ))

    # ── Animated particle positions ──
    def make_scatter(t_idx: int) -> go.Scattermapbox:
        active  = status[idx, t_idx] == 0
        stranded = ~active
        colors = np.where(active, "#2196F3", "#f44336")
        return go.Scattermapbox(
            lon=lons[idx, t_idx].tolist(),
            lat=lats[idx, t_idx].tolist(),
            mode="markers",
            marker=dict(size=5, color=colors.tolist()),
            name="Particles",
            showlegend=True,
        )

    # Initial animated trace (frame 0)
    traces.append(make_scatter(0))
    animated_trace_idx = len(traces) - 1

    # Build frames
    frames = []
    for t_idx in t_idxs:
        ts  = times[t_idx]
        label = str(ts)[:16].replace("T", " ")
        frames.append(go.Frame(
            data=[make_scatter(t_idx)],
            traces=[animated_trace_idx],
            name=label,
        ))

    # Slider steps
    slider_steps = [
        dict(args=[[f.name], {"frame": {"duration": 80, "redraw": True}, "mode": "immediate"}],
             label=f.name[5:],   # show "MM-DD HH:MM"
             method="animate")
        for f in frames
    ]

    layout = go.Layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=field["lon"], lat=field["lat"]),
            zoom=6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.01, y=0.06, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 80, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate"}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, currentvalue=dict(prefix="Time: ", visible=True, xanchor="center"),
            pad=dict(t=50), steps=slider_steps,
        )],
    )

    fig = go.Figure(data=traces, frames=frames, layout=layout)
    return fig


@st.cache_data(show_spinner=False)
def load_risk_grid(npz_path: str) -> dict:
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def build_risk_figure(
    grid: dict, field: dict, field_name: str,
    metric: str, threshold: float,
) -> go.Figure:
    lons = grid["lons"]
    lats = grid["lats"]
    prob = grid[metric]  # (n_lat, n_lon)

    # Cell centres
    clon = lons + GRID_RES / 2
    clat = lats + GRID_RES / 2
    lon_g, lat_g = np.meshgrid(clon, clat)
    lon_f = lon_g.flatten()
    lat_f = lat_g.flatten()
    prob_f = prob.flatten()

    mask = prob_f >= threshold
    traces = []

    if mask.any():
        traces.append(go.Densitymapbox(
            lon=lon_f[mask].tolist(),
            lat=lat_f[mask].tolist(),
            z=prob_f[mask].tolist(),
            radius=22,
            colorscale="YlOrRd",
            zmin=threshold,
            zmax=1.0,
            opacity=0.70,
            showscale=True,
            colorbar=dict(
                title="Probability",
                tickformat=".0%",
                x=1.0,
            ),
            name="Risk",
        ))

    traces.append(go.Scattermapbox(
        lon=[field["lon"]], lat=[field["lat"]],
        mode="markers",
        marker=dict(size=14, color="#4CAF50", symbol="circle"),
        name=f"{field_name} (source)",
    ))

    layout = go.Layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=field["lon"], lat=field["lat"]),
            zoom=6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return go.Figure(data=traces, layout=layout)


def build_stats(data: dict) -> dict:
    status = data["status"]
    lons   = data["lon"]
    lats   = data["lat"]
    final  = status[:, -1]
    total  = len(final)
    active = int((final == 0).sum())
    stranded = total - active
    active_mask = final == 0
    if active_mask.any():
        lon_range = (float(lons[active_mask, -1].min()), float(lons[active_mask, -1].max()))
        lat_range = (float(lats[active_mask, -1].min()), float(lats[active_mask, -1].max()))
    else:
        lon_range = lat_range = (None, None)
    return dict(total=total, active=active, stranded=stranded,
                lon_range=lon_range, lat_range=lat_range)


def show_results(data: dict, field: dict, field_name: str, cfg: dict) -> None:
    col_map, col_info = st.columns([3, 1])

    show_density = st.toggle("Show density heatmap", value=False,
                             help="Overlay showing concentration of final particle positions")

    with col_map:
        st.plotly_chart(
            build_animated_figure(data, field, field_name, show_density),
            use_container_width=True,
        )

    with col_info:
        stats = build_stats(data)
        st.subheader(f"{cfg.get('field', field_name)}")
        if cfg.get("start"):
            st.caption(f"Start: {cfg['start']}  |  {cfg.get('duration_h')}h")
        st.metric("Total particles", stats["total"])
        st.metric("Active (at sea)", stats["active"],
                  delta=f"{stats['active']/stats['total']*100:.0f}%")
        st.metric("Stranded / inactive", stats["stranded"],
                  delta=f"-{stats['stranded']/stats['total']*100:.0f}%",
                  delta_color="inverse")
        if stats["lon_range"][0] is not None:
            st.caption("**Final active extent**")
            st.caption(f"Lon: {stats['lon_range'][0]:.2f}° → {stats['lon_range'][1]:.2f}°")
            st.caption(f"Lat: {stats['lat_range'][0]:.2f}° → {stats['lat_range'][1]:.2f}°")
        st.divider()
        st.caption(f"Wind: {'ON' if cfg.get('use_wind', True) else 'OFF'}")
        st.caption(f"Stokes drift: {'ON' if cfg.get('use_waves') else 'OFF'}")
        nc_path = ROOT / cfg.get("nc", "main/outputs/openoil_run.nc")
        if nc_path.exists():
            with st.expander("Download"):
                with open(nc_path, "rb") as f:
                    st.download_button("NetCDF trajectory", f,
                                       file_name=nc_path.name,
                                       mime="application/octet-stream")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_pre, tab_risk, tab_custom = st.tabs(["📦 Pre-computed Scenarios", "🗺️ Risk Maps", "⚙️ Custom Run"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pre-computed scenarios
# ════════════════════════════════════════════════════════════════════════════

with tab_pre:
    manifest = load_manifest()

    if not manifest:
        st.info(
            "No pre-computed scenarios found. Run the batch script first:\n\n"
            "```\npython main/scripts/precompute_scenarios.py\n```\n\n"
            "This takes ~50 minutes and generates 48 scenarios (6 fields × 4 seasons × wind on/off)."
        )
    else:
        ready_fields  = sorted({v["field"]  for v in manifest.values()})
        ready_seasons = sorted({v["season"] for v in manifest.values()},
                               key=list(SEASON_LABELS.keys()).index)

        col1, col2, col3 = st.columns(3)
        with col1:
            sel_field  = st.selectbox("Field",  ready_fields,  key="pre_field")
        with col2:
            sel_season = st.selectbox("Season", ready_seasons,
                                      format_func=lambda s: SEASON_LABELS[s],
                                      key="pre_season")
        with col3:
            sel_wind   = st.selectbox("Wind forcing", ["wind_on", "wind_off"],
                                      format_func=lambda w: "On" if w == "wind_on" else "Off",
                                      key="pre_wind")

        key = f"{sel_field.lower().replace(' ', '_')}_{sel_season}_{sel_wind}"

        if key in manifest:
            entry   = manifest[key]
            nc_path = ROOT / entry["nc"]
            field   = CAMPOS_FIELDS[sel_field]

            st.caption(
                f"📂 Pre-computed · {sel_field} · {SEASON_LABELS[sel_season]} 2025 · "
                f"wind {'on' if sel_wind == 'wind_on' else 'off'} · "
                f"computed {entry['computed'][:10]}"
            )

            data = load_nc(str(nc_path))
            show_results(data, field, sel_field,
                         cfg=dict(field=sel_field,
                                  start=str(SEASON_DATES[sel_season])[:10],
                                  duration_h=120,
                                  use_wind=sel_wind == "wind_on",
                                  use_waves=False,
                                  nc=entry["nc"]))
        else:
            st.warning(
                f"Scenario **{key}** not computed yet. "
                "Run `python main/scripts/precompute_scenarios.py` to generate it."
            )

        # Progress summary
        with st.expander(f"Scenario coverage ({len(manifest)}/48 ready)"):
            rows = []
            for field_n in CAMPOS_FIELDS:
                row = {"Field": field_n}
                for s in SEASON_LABELS:
                    for w in ["wind_on", "wind_off"]:
                        k = f"{field_n.lower().replace(' ', '_')}_{s}_{w}"
                        row[f"{SEASON_LABELS[s]} {'💨' if w == 'wind_on' else '🌫'}"] = "✅" if k in manifest else "⬜"
                rows.append(row)
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Field"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk Maps
# ════════════════════════════════════════════════════════════════════════════

with tab_risk:
    risk_manifest_path = RISK_DIR / "manifest.json"

    if not risk_manifest_path.exists():
        st.info(
            "No risk grids computed yet. Run the two-step ensemble pipeline first:\n\n"
            "**Step 1** — run ensemble simulations (~4–8 h):\n"
            "```\npython main/scripts/run_ensemble.py\n```\n\n"
            "**Step 2** — aggregate into probability grids (~5 min):\n"
            "```\npython main/scripts/compute_risk_grids.py\n```"
        )
    else:
        risk_manifest = json.loads(risk_manifest_path.read_text())

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_field = st.selectbox(
                "Field",
                [v["field"] for v in risk_manifest.values()
                 if v["season"] == list(risk_manifest.values())[0]["season"]],
                key="risk_field",
            )
        with rc2:
            available_seasons = sorted(
                {v["season"] for v in risk_manifest.values() if v["field"] == r_field},
                key=list(SEASON_LABELS.keys()).index,
            )
            r_season = st.selectbox(
                "Season",
                available_seasons,
                format_func=lambda s: SEASON_LABELS[s],
                key="risk_season",
            )
        with rc3:
            r_metric = st.radio(
                "Probability type",
                ["prob_any", "prob_final"],
                format_func=lambda m: "Exposure (any time in 120 h)" if m == "prob_any"
                                      else "Persistence (end of simulation)",
                key="risk_metric",
                horizontal=True,
            )

        r_threshold = st.slider(
            "Minimum probability to display",
            min_value=0.05, max_value=0.50, value=0.10, step=0.05,
            help="Grid cells with probability below this value are hidden",
        )

        grid_key = f"{r_field.lower().replace(' ', '_')}_{r_season}"
        if grid_key in risk_manifest:
            entry    = risk_manifest[grid_key]
            npz_path = str(ROOT / entry["npz"])
            grid     = load_risk_grid(npz_path)
            field_cfg = CAMPOS_FIELDS[r_field]

            prob = grid[r_metric]
            n_members = int(grid["n_members"])

            # ── Map + stats ──────────────────────────────────────────────
            map_col, info_col = st.columns([3, 1])

            with map_col:
                st.plotly_chart(
                    build_risk_figure(grid, field_cfg, r_field, r_metric, r_threshold),
                    use_container_width=True,
                )

            with info_col:
                st.subheader(f"{r_field}")
                st.caption(f"{SEASON_LABELS[r_season]} 2025 · {n_members} ensemble members")
                st.divider()

                cells_at_risk = int((prob >= r_threshold).sum())
                cell_area_km2 = (GRID_RES * 111) * (GRID_RES * 111 * np.cos(np.radians(field_cfg["lat"])))
                area_km2 = cells_at_risk * cell_area_km2

                st.metric("Peak probability", f"{prob.max() * 100:.0f}%")
                st.metric(
                    f"Area at risk (P ≥ {r_threshold:.0%})",
                    f"{area_km2:,.0f} km²",
                )
                st.metric("Ensemble members", n_members)

                # Centroid of risk area
                lons_g, lats_g = grid["lons"], grid["lats"]
                clon = lons_g + GRID_RES / 2
                clat = lats_g + GRID_RES / 2
                lon_g, lat_g = np.meshgrid(clon, clat)
                mask = prob >= r_threshold
                if mask.any():
                    risk_lon = float(np.average(lon_g[mask], weights=prob[mask]))
                    risk_lat = float(np.average(lat_g[mask], weights=prob[mask]))
                    st.caption("**Risk centroid**")
                    st.caption(f"{risk_lat:.2f}°N  {risk_lon:.2f}°E")
                    dist_km = (
                        ((risk_lon - field_cfg["lon"]) * 111 * np.cos(np.radians(field_cfg["lat"])))**2
                        + ((risk_lat - field_cfg["lat"]) * 111)**2
                    ) ** 0.5
                    bearing_deg = np.degrees(np.arctan2(
                        risk_lon - field_cfg["lon"],
                        risk_lat - field_cfg["lat"],
                    )) % 360
                    dirs = ["N","NE","E","SE","S","SW","W","NW"]
                    direction = dirs[int((bearing_deg + 22.5) / 45) % 8]
                    st.caption(f"{dist_km:.0f} km {direction} of source")

                st.divider()
                st.caption(
                    "Each ensemble member uses a different start date within "
                    "the month to capture natural variability in wind and current patterns."
                )
        else:
            st.warning(f"Risk grid for **{grid_key}** not found in manifest.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Custom run (live simulation)
# ════════════════════════════════════════════════════════════════════════════

with tab_custom:
    with st.sidebar:
        st.header("Custom run settings")

        field_name = st.selectbox("Oil field", list(CAMPOS_FIELDS.keys()))
        field      = CAMPOS_FIELDS[field_name]

        st.markdown(f"""
**{field_name}**
{field['description']}
API: **{field['api']}°** · {field['operator']} · {field['water_depth_m']} m
""")
        st.divider()

        # Data coverage
        curr_start, curr_end = get_data_range(str(ROOT / "main/inputs/currents.nc"))
        wind_start, wind_end = get_data_range(str(ROOT / "main/inputs/wind_cf.nc"))

        if curr_start and wind_start:
            data_start = max(curr_start, wind_start)
            data_end   = min(curr_end,   wind_end)
            st.caption(f"📂 Data: **{data_start.strftime('%b %d')} – {data_end.strftime('%b %d, %Y')}**")
        else:
            data_start = datetime(2025, 1, 1)
            data_end   = datetime(2025, 1, 7)
            st.warning("Could not read data files.")

        max_start = data_end - timedelta(hours=12)

        start_date = st.date_input(
            "Start date",
            value=data_start.date(),
            min_value=data_start.date(),
            max_value=max_start.date(),
        )
        start_hour = st.slider("Start hour (UTC)", 0, 23, 0)

        sel_start    = datetime(start_date.year, start_date.month, start_date.day, start_hour)
        max_dur      = max(12, min(120, int((data_end - sel_start).total_seconds() // 3600)))
        duration_h   = st.slider("Duration (hours)", 12, max_dur, min(120, max_dur), step=12)

        st.divider()
        use_wind   = st.toggle("Wind forcing",        value=True)
        use_waves  = st.toggle("Stokes drift (waves)", value=False,
                               help="Requires waves_cf.nc — run scripts/download_era5_waves.py first")
        n_particles = st.slider("Particles", 100, 2000, 1000, step=100)

        run_btn = st.button("▶ Run simulation", type="primary", use_container_width=True)

    CUSTOM_NC  = ROOT / "main" / "outputs" / "openoil_run.nc"
    CUSTOM_FIG = ROOT / "main" / "outputs" / "tracks.png"

    if run_btn:
        with st.spinner(f"Running {duration_h}h simulation for {field_name}…"):
            t0 = time.time()
            try:
                run_simulation(
                    seed_lon=field["lon"], seed_lat=field["lat"],
                    n_particles=n_particles, start_time=sel_start,
                    duration_hours=duration_h, oil_type=field["oil_type"],
                    use_wind=use_wind, use_waves=use_waves,
                    outfile=str(CUSTOM_NC), figfile=str(CUSTOM_FIG), loglevel=40,
                )
                st.session_state["custom_cfg"] = dict(
                    field=field_name, start=sel_start.isoformat(),
                    duration_h=duration_h, use_wind=use_wind, use_waves=use_waves,
                    nc=str(CUSTOM_NC.relative_to(ROOT)),
                )
                load_nc.clear()
                st.success(f"Done in {time.time()-t0:.0f}s")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    if CUSTOM_NC.exists():
        cfg  = st.session_state.get("custom_cfg", dict(
            field=field_name, start="", duration_h=duration_h,
            use_wind=use_wind, use_waves=use_waves,
            nc=str(CUSTOM_NC.relative_to(ROOT)),
        ))
        data = load_nc(str(CUSTOM_NC))
        show_results(data, field, cfg.get("field", field_name), cfg)
    else:
        st.info("Configure parameters in the sidebar and press **▶ Run simulation**.")
