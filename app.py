import calendar
from datetime import date
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from NINJA_DATA import fetch_renewables_15y,pv_climatologies,plot_yearly_shape

st.set_page_config(page_title="PV Climatology Viewer", layout="wide")

st.title("PV Climatology Viewer")
st.caption("Yearly shape + average 24h monthly shapes (no raw time series plotted)")

# ---------------- Sidebar controls ----------------
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
    tracking = st.selectbox("Tracking", options=[0, 1, 2], index=0, help="0=fixed, 1=NS, 2=EW (per Ninja)")

    tz = st.selectbox("Local timezone", ["Europe/Berlin", "UTC"], index=0)

    profile_kind = st.radio(
        "Monthly 24h profile type",
        ["Peak-normalized", "Energy-normalized", "Raw"],
        index=0,
        help="How to display monthly shapes."
    )

    run = st.button("Run")

# --------------- Helpers for plotting in Streamlit ---------------
def plot_month_profile(series24: pd.Series, month_name: str, ylabel="Relative", suffix="(peak-normalized)"):
    fig, ax = plt.subplots()
    x = list(series24.index.astype(int))  # 0..23
    y = series24.values.tolist()
    bars = ax.bar(x, y)
    ax.set_title(f"Average 24h Shape – {month_name} {suffix}".strip())
    ax.set_xlabel("Hour"); ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x))); ax.set_xticklabels(x)
    for rect, val in zip(bars, y):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)
    return fig

# ---------------- Main action ----------------
if run:
    with st.spinner("Fetching PV and computing climatologies…"):
        pv_de = fetch_renewables_15y(
            "Solar", lat, lon,
            end_date=end_date.strftime("%Y-%m-%d"),
            years=years,
            dataset=dataset,
            capacity=capacity,
            system_loss=system_loss,
            tracking=tracking,
            tilt=tilt,
            azim=azim,
        )

        clim = pv_climatologies(pv_de, value_col="electricity", tz=tz)

    # ------- 1) ONE PLOT: YEARLY SHAPE -------
    st.subheader("Yearly Shape")
    fig_year = plot_yearly_shape(clim["yearly_shape"])
    st.pyplot(fig_year, clear_figure=True)

    # Decide which monthly profile matrix to use
    if profile_kind == "Peak-normalized":
        monthly_matrix = clim["monthly_hour_peak"]
        ylabel = "Relative"
        suffix = "(peak-normalized)"
    elif profile_kind == "Energy-normalized":
        monthly_matrix = clim["monthly_hour_energy"]
        ylabel = "Relative"
        suffix = "(energy-normalized)"
    else:
        monthly_matrix = clim["monthly_hour_profile"]
        ylabel = "PV"
        suffix = "(raw)"

    # ------- 2) TWELVE PLOTS: MONTHLY 24h SHAPES -------
    st.subheader("Monthly 24h Shapes")
    # 3 columns x 4 rows grid
    cols = st.columns(3)
    for m in range(1, 13):
        s = monthly_matrix[m]  # 24 hourly values for month m
        fig = plot_month_profile(s, calendar.month_name[m], ylabel=ylabel, suffix=suffix)
        with cols[(m-1) % 3]:
            st.pyplot(fig, clear_figure=True)

    st.success("Done. Raw full-year time series not plotted (by design).")
