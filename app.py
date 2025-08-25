# app.py — PV Climatology Viewer
import calendar
from datetime import date
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from NINJA_DATA import fetch_renewables_15y, pv_climatologies

st.set_page_config(page_title="FHP", layout="wide")
st.title("PV Climatology Viewer")
st.caption("Yearly shape + monthly 24h shapes. Computes raw/peak/energy once; no full time series plotting.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    lat = st.number_input("Latitude", value=48.14, format="%.5f")
    lon = st.number_input("Longitude", value=11.58, format="%.5f")
    years = st.slider("Years of history", min_value=3, max_value=30, value=15)
    end_date = st.date_input("End date", value=date(2023, 12, 31))
    dataset = st.selectbox("Ninja dataset", ["merra2", "era5"], index=0)
    capacity = st.number_input("Capacity (kW for scaling)", value=1.0, step=0.5)
    tilt = st.number_input("Tilt (°)", value=35)
    azim = st.number_input("Azimuth (°; 180 = south)", value=180)
    system_loss = st.number_input("System loss (0–1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05)
    tracking = st.selectbox("Tracking", options=[0, 1, 2], index=0, help="0=fixed, 1=NS, 2=EW")
    tz = st.selectbox("Local timezone", ["Europe/Berlin", "UTC"], index=0)

    view = st.radio(
        "Monthly 24h profile type",
        ["Peak-normalized", "Energy-normalized", "Raw"],
        index=0,
        help="Switch without refetching; all three are precomputed."
    )

    run = st.button("Run")

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def _fetch_pv(tech, lat, lon, end_date, years, dataset, capacity, system_loss, tracking, tilt, azim):
    return fetch_renewables_15y(
        tech, lat, lon,
        end_date=end_date, years=years, dataset=dataset,
        capacity=capacity, system_loss=system_loss,
        tracking=tracking, tilt=tilt, azim=azim
    )

@st.cache_data(show_spinner=False)
def _compute_clim(pv_df, tz):
    # returns: monthly_mean, yearly_shape, monthly_hour_profile, monthly_hour_peak, monthly_hour_energy
    return pv_climatologies(pv_df, value_col="electricity", tz=tz)

@st.cache_data(show_spinner=False)
def _doy_hour_profile(pv_df, tz):
    """Average 365×24 profile across years (raw), in local time — for CSV download."""
    s = pv_df["electricity"].copy()
    idx = pd.to_datetime(pv_df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    s.index = idx.tz_convert(tz)
    s = s.sort_index()  # sort the SERIES

    df = s.to_frame("pv")
    df["month_day"] = df.index.strftime("%m-%d")
    df["hour"] = df.index.hour
    # drop Feb 29 so we get 365 unique days
    df = df[df["month_day"] != "02-29"]
    mat = df.groupby(["month_day", "hour"])["pv"].mean().unstack("hour")
    # reorder to calendar
    cal = pd.date_range("2001-01-01", "2001-12-31", freq="D").strftime("%m-%d")
    mat = mat.reindex(cal)
    mat.index.name = "month_day"
    return mat  # 365×24

def _fig_yearly_shape(yearly_shape):
    fig, ax = plt.subplots(figsize=(5.5, 2.7))   # compact
    x = yearly_shape.index.astype(int).tolist()
    y = yearly_shape.values.tolist()
    bars = ax.bar(x, y)
    ax.set_title("PV Yearly Shape (monthly / annual mean)")
    ax.set_xlabel("Month"); ax.set_ylabel("Relative level")
    ax.set_xticks(range(len(x))); ax.set_xticklabels(x)
    for r, v in zip(bars, y):
        ax.text(r.get_x()+r.get_width()/2, r.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    return fig

def _fig_month(series24, month_name, ylabel="Relative", suffix="(peak-normalized)"):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))  # smaller to fit better
    x = list(series24.index.astype(int))
    y = series24.values.tolist()
    bars = ax.bar(x, y)
    ax.set_title(f"{month_name} {suffix}")
    ax.set_xlabel("Hour"); ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x))); ax.set_xticklabels(x)
    for r, v in zip(bars, y):
        ax.text(r.get_x()+r.get_width()/2, r.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    return fig

def _csv_button(label, df_or_series, filename):
    if isinstance(df_or_series, pd.Series):
        data = df_or_series.to_csv(header=["value"]).encode()
    else:
        data = df_or_series.to_csv().encode()
    st.download_button(label, data=data, file_name=filename, mime="text/csv")

# ---------------- Main ----------------
if run:
    with st.spinner("Fetching PV and computing climatologies…"):
        pv_de = _fetch_pv(
            "Solar", lat, lon,
            end_date=end_date.strftime("%Y-%m-%d"),
            years=years, dataset=dataset,
            capacity=capacity, system_loss=system_loss,
            tracking=tracking, tilt=tilt, azim=azim
        )
        clim = _compute_clim(pv_de, tz)
        doy = _doy_hour_profile(pv_de, tz)  # 365×24 raw “average day by day”

    # ---------- one compact yearly plot ----------
    st.subheader("Yearly Shape")
    st.pyplot(_fig_yearly_shape(clim["yearly_shape"]), clear_figure=True, use_container_width=True)

    # Downloads for yearly
    with st.expander("Download yearly shape (CSV)"):
        _csv_button("Download yearly_shape.csv", clim["yearly_shape"], "pv_yearly_shape.csv")

    # ---------- choose which monthly matrix to show (precomputed) ----------
    if view == "Peak-normalized":
        monthly_matrix = clim["monthly_hour_peak"]; ylabel = "Relative"; suffix = "(peak-normalized)"
    elif view == "Energy-normalized":
        monthly_matrix = clim["monthly_hour_energy"]; ylabel = "Relative"; suffix = "(energy-normalized)"
    else:
        monthly_matrix = clim["monthly_hour_profile"]; ylabel = "PV"; suffix = "(raw)"

    st.subheader(f"Monthly 24h Shapes {suffix}")
    cols = st.columns(4)  # 4 columns to fit on screen
    for m in range(1, 13):
        if m not in monthly_matrix.columns:
            continue
        s = monthly_matrix[m]
        fig = _fig_month(s, calendar.month_name[m], ylabel=ylabel, suffix=suffix)
        with cols[(m-1) % 4]:
            st.pyplot(fig, clear_figure=True, use_container_width=True)

    # Downloads for monthly matrix
    with st.expander("Download monthly 24h matrices (CSV)"):
        _csv_button("Selected view (24×12)", monthly_matrix, "pv_monthly_24h_selected.csv")
        _csv_button("Raw (24×12)", clim["monthly_hour_profile"], "pv_monthly_24h_raw.csv")
        _csv_button("Peak-normalized (24×12)", clim["monthly_hour_peak"], "pv_monthly_24h_peak.csv")
        _csv_button("Energy-normalized (24×12)", clim["monthly_hour_energy"], "pv_monthly_24h_energy.csv")

    # ---------- day-of-year raw hourly profile (no plotting) ----------
    st.subheader("Day-of-Year Hourly Profile (raw, no plot)")
    st.caption("Average 365×24 profile across years, in local time. Download below.")
    c1, c2 = st.columns(2)
    with c1:
        _csv_button("365×24 hourly profile (CSV)", doy, "pv_doy_365x24.csv")
    with c2:
        _csv_button("Daily totals (CSV)", doy.sum(axis=1).rename("daily_total"),
                    "pv_doy_daily_totals.csv")

    st.success("Done. All views were computed once; switching types does not re-fetch.")
