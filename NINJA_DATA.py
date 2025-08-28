import time, json, requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os, calendar
import numpy as np
import matplotlib.pyplot as plt

API_BASE = "https://www.renewables.ninja/api"
API_TOKEN = "d8347c8ab55103acfae94d65de8d36803a2a69b6"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

def fetch_renewables_15y(technology: str, lat: float, lon: float,end_date: str = "2023-12-31",  years: int = 30,dataset: str = "merra2",capacity: float = 1.0,system_loss: float = 0.0,tracking: int = 0, tilt: int = 35, azim: int = 180,height: int = 80, turbine: str = "Vestas V80 2000",sleep_s: float = 1.1,) -> pd.DataFrame:
    tech = technology.lower().strip()
    if tech == "solar":
        url = f"{API_BASE}/data/pv"
        extra = {"system_loss": system_loss, "tracking": tracking, "tilt": tilt, "azim": azim}
    elif tech in ("onshore wind", "offshore wind", "wind"):
        url = f"{API_BASE}/data/wind"
        extra = {"height": height, "turbine": turbine}
    else:
        raise ValueError("technology must be 'Solar', 'Onshore Wind', or 'Offshore Wind'")

    end_dt = pd.to_datetime(end_date).normalize()
    start_dt = end_dt - relativedelta(years=years) + timedelta(days=1) 

    all_frames = []
    y0 = start_dt.year
    y1 = end_dt.year

    session = requests.Session()
    session.headers.update(HEADERS)

    def fetch_one(y_start: datetime, y_end: datetime) -> pd.DataFrame:
        params = {
            "lat": lat, "lon": lon,
            "date_from": y_start.strftime("%Y-%m-%d"),
            "date_to":   y_end.strftime("%Y-%m-%d"),
            "dataset": dataset,
            "capacity": capacity,
            "format": "json",
            "header": "true",      
            "interpolate": "true"}
        params.update(extra)

        backoff = 1.0
        for attempt in range(6):
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                payload = r.json()
                df = pd.read_json(json.dumps(payload["data"]), orient="index")
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.rename_axis("time").sort_index()
                return df
            if r.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 10)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError("Rate limit: gave up after multiple retries.")

    cur_start = start_dt
    while cur_start <= end_dt:
        cur_end = min(cur_start + relativedelta(years=1) - timedelta(days=1), end_dt)
        df_chunk = fetch_one(cur_start, cur_end)
        all_frames.append(df_chunk)
        time.sleep(sleep_s)  
        cur_start = cur_end + timedelta(days=1)

    out = pd.concat(all_frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out



def pv_climatologies(pv_df: pd.DataFrame,value_col: str = "electricity",tz: str = "Europe/Berlin",drop_feb29: bool = True):
    if value_col not in pv_df.columns:
        num_cols = pv_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric column found; specify value_col explicitly.")
        value_col = num_cols[0]

    s = pv_df[value_col].copy()
    idx = pd.to_datetime(pv_df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    s.index = idx

    s = s.tz_convert(tz).sort_index()

    series_local = s.rename("pv")

    df = s.to_frame("pv")
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    if drop_feb29:
        md = df.index.strftime("%m-%d")
        df = df[md != "02-29"]

    monthly_mean = df.groupby("month")["pv"].mean()
    annual_mean = monthly_mean.mean()
    yearly_shape_norm = monthly_mean / annual_mean if annual_mean != 0 else monthly_mean * 0
    monthly_peak = monthly_mean.max()
    yearly_shape_peak = monthly_mean / monthly_peak if monthly_peak != 0 else monthly_mean * 0

    monthly_hour_profile = (df.groupby(["month", "hour"])["pv"].mean().unstack("month").sort_index())
    monthly_hour_profile.index.name = "hour"
    monthly_hour_profile.columns.name = "month"

    monthly_hour_peak = monthly_hour_profile / monthly_hour_profile.max(axis=0)
    monthly_hour_energy = monthly_hour_profile.div(monthly_hour_profile.sum(axis=0), axis=1)

    return {
        "monthly_mean": monthly_mean,                 # 12 numbers
        "yearly_shape_norm": yearly_shape_norm,       # normalized by annual mean
        "yearly_shape_peak": yearly_shape_peak,       # normalized by monthly max
        "monthly_hour_profile": monthly_hour_profile, # 24 x 12
        "monthly_hour_peak": monthly_hour_peak,       # 24 x 12 (peak=1 per month)
        "monthly_hour_energy": monthly_hour_energy,   # 24 x 12 (sum=1 per month)
        "all_hours_raw": series_local                 # full raw hourly PV series
    }

def bar_with_labels(x, y, title, xlabel, ylabel, rotation=0, fmt="{:.2f}", save_path=None):
    fig, ax = plt.subplots()
    bars = ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=rotation)
    for rect, val in zip(bars, y):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                fmt.format(val), ha="center", va="bottom", fontsize=8)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_yearly_shape(yearly_shape: pd.Series, save_path: str | None = None):
    x = yearly_shape.index.astype(int).tolist()     
    y = yearly_shape.values.tolist()
    bar_with_labels(x, y,"PV Yearly Shape (monthly / annual mean)", "Month", "Relative level",fmt="{:.2f}", save_path=save_path)

def plot_monthly_shapes(monthly_profile: pd.DataFrame,label: str = "Relative",title_suffix: str = "",out_dir: str | None = None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for m in range(1, 13):
        if m not in monthly_profile.columns:
            continue
        s = monthly_profile[m]           
        x = list(s.index.astype(int))     
        y = s.values.tolist()
        title = f"Average 24h Shape – {calendar.month_name[m]} {title_suffix}".strip()
        save_path = os.path.join(out_dir, f"pv_shape_{m:02d}.png") if out_dir else None
        bar_with_labels(x, y, title, "Hour", label, fmt="{:.2f}", save_path=save_path)

