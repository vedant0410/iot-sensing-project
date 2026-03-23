"""
MobiSense Dashboard — Script 07
Interactive Streamlit dashboard for 7-day iPhone sensor + weather analysis.

Pages
-----
1. Overview          — KPI cards, 7-day timeline, GPS trace
2. Sensor Explorer   — Interactive time series per sensor
3. DSP Analysis      — Filter outputs, PSD, spectrogram, step detection
4. Correlations      — Spearman heatmap, OLS regression, significance panel
5. Day Explorer      — Deep dive into any single recording day

Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # /Users/vedant/Downloads/IOT PROJECT/
PROC = BASE / "processed"
FIGS = BASE / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MobiSense · IoT Sensor Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 1.6rem; font-weight: 700; }
    /* Sidebar nav */
    section[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }
    /* Section headers */
    .section-header { color: #3B82F6; font-weight: 700; font-size: 1.05rem;
                      border-bottom: 1px solid #334155; padding-bottom: 4px; margin-top: 1rem; }
    /* Callout box */
    .callout { background: #1E3A5F; border-left: 4px solid #3B82F6;
               padding: 10px 14px; border-radius: 4px; font-size: 0.9rem; }
    .callout-green { background: #14532D; border-left: 4px solid #22C55E;
                     padding: 10px 14px; border-radius: 4px; font-size: 0.9rem; }
    .callout-amber { background: #451A03; border-left: 4px solid #F59E0B;
                     padding: 10px 14px; border-radius: 4px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Shared footer ─────────────────────────────────────────────────────────────
def show_footer():
    st.divider()
    with st.expander("ℹ️ About this study & methodology"):
        fc1, fc2, fc3 = st.columns(3)
        fc1.markdown("""
**Data Collection**
- Device: iPhone 16 Pro Max
- App: Sensor Logger (store-and-forward)
- Period: 13–20 March 2026 (7.21 days)
- Location: West London (51.51°N, −0.24°E)
- Sensors: Accelerometer, Gyroscope, Barometer, Pedometer, Activity, Location, Battery, Network
- Weather: Open-Meteo Archive API (hourly, ECMWF reanalysis)
""")
        fc2.markdown("""
**Analysis Pipeline**
1. `01_concatenate.py` — merge daily CSVs
2. `02_fetch_weather.py` — Open-Meteo API
3. `03_clean.py` — preprocessing & cleaning
4. `04_dsp.py` — DSP (LPF, RMS, PSD, STFT, MA)
5. `05_crosscorrelation.py` — Spearman + OLS
6. `06_insights.py` — figures & summaries
7. `07_dashboard/app.py` — this dashboard

**Tools:** Python · Pandas · NumPy · SciPy · Statsmodels · Streamlit · Plotly
""")
        fc3.markdown("""
**Statistical Methods**
- Spearman rank correlation (n=7 daily, non-parametric)
- Pearson correlation (n=173 hourly, instrument validation)
- OLS multiple linear regression (n=173, 5 predictors)
- Welch PSD (step cadence detection)
- Butterworth LPF (barometer denoising)
- STFT spectrogram (temporal frequency analysis)

**Ethics:** All data self-collected by the researcher. No third-party personal data used.

**Submitted for:** ELEC70126 · MSc AI Applications & Innovation · Imperial College London
""")
        st.markdown("---")
        st.markdown("**📊 System Architecture & Data Flow Diagram**")
        st.markdown("""
<style>
.dfd-wrap { font-family: sans-serif; padding: 8px 0; }
.dfd-layer { display: flex; align-items: stretch; gap: 10px; margin-bottom: 4px; }
.dfd-label { width: 130px; min-width: 130px; display: flex; align-items: center;
             font-size: 0.72rem; font-weight: 700; color: #64748B;
             text-transform: uppercase; letter-spacing: 0.06em; }
.dfd-nodes { display: flex; flex: 1; gap: 8px; align-items: stretch; }
.dfd-node { flex: 1; border-radius: 8px; padding: 9px 10px; font-size: 0.78rem;
            line-height: 1.4; border: 1px solid rgba(255,255,255,0.08); }
.dfd-node b { display: block; font-size: 0.82rem; margin-bottom: 2px; }
.dfd-node.source  { background: #1E3A5F; border-color: #3B82F6; color: #BAE6FD; }
.dfd-node.raw     { background: #1C2333; border-color: #475569; color: #94A3B8; }
.dfd-node.script  { background: #1A2E1A; border-color: #22C55E; color: #86EFAC; }
.dfd-node.output  { background: #2D1B00; border-color: #F59E0B; color: #FCD34D; }
.dfd-node.dash    { background: #2D1040; border-color: #A855F7; color: #D8B4FE; }
.dfd-arrow { display: flex; justify-content: center; align-items: center;
             color: #475569; font-size: 1.1rem; margin: 0; padding: 0; line-height:1; }
</style>
<div class="dfd-wrap">

  <div class="dfd-layer">
    <div class="dfd-label">① Data<br>Sources</div>
    <div class="dfd-nodes">
      <div class="dfd-node source">
        <b>📱 iPhone 16 Pro Max</b>
        Accelerometer · Gyroscope · Barometer · Pedometer<br>
        Activity · Location (GPS) · Battery · Network<br>
        <span style="font-size:0.71rem;color:#7DD3FC">via Sensor Logger app · store-and-forward to Mac</span>
      </div>
      <div class="dfd-node source">
        <b>🌤️ Open-Meteo Archive API</b>
        Hourly atmospheric data for West London<br>
        Temperature · Pressure · Humidity · Wind · Rain<br>
        Sunshine · Solar radiation · Cloud cover · WMO codes
      </div>
    </div>
  </div>

  <div class="dfd-layer"><div class="dfd-label"></div>
    <div class="dfd-nodes"><div class="dfd-arrow" style="flex:1;text-align:center">▼ &nbsp; daily CSV export + API fetch &nbsp; ▼</div></div>
  </div>

  <div class="dfd-layer">
    <div class="dfd-label">② Raw<br>Storage</div>
    <div class="dfd-nodes">
      <div class="dfd-node raw">
        <b>Script 01 · concatenate.py</b>
        Merges daily session CSVs per sensor → 8 × combined.csv
        <span style="display:block;font-size:0.71rem;color:#64748B;margin-top:3px">accel 1.2 GB · gyro 1.2 GB · baro 102 MB · ...</span>
      </div>
      <div class="dfd-node raw">
        <b>Script 02 · fetch_weather.py</b>
        Calls Open-Meteo Archive API → weather_combined.csv
        <span style="display:block;font-size:0.71rem;color:#64748B;margin-top:3px">192 hourly rows · 10 atmospheric variables</span>
      </div>
    </div>
  </div>

  <div class="dfd-layer"><div class="dfd-label"></div>
    <div class="dfd-nodes"><div class="dfd-arrow" style="flex:1;text-align:center">▼ &nbsp; cleaning, outlier removal, feature engineering &nbsp; ▼</div></div>
  </div>

  <div class="dfd-layer">
    <div class="dfd-label">③ Analysis<br>Pipeline</div>
    <div class="dfd-nodes">
      <div class="dfd-node script">
        <b>Script 03 · clean.py</b>
        ±4σ clipping · sentinel removal · unit conversion
        daily_steps derivation · at_home flag · WMO labels
      </div>
      <div class="dfd-node script">
        <b>Script 04 · dsp.py</b>
        Butterworth LPF · RMS energy · Band-pass filter
        Welch PSD → 1.963 Hz · STFT spectrogram · Battery MA
      </div>
      <div class="dfd-node script">
        <b>Script 05 · crosscorrelation.py</b>
        Spearman ρ (n=7) · Pearson r (n=173)
        OLS regression · lag analysis · p-value testing
      </div>
      <div class="dfd-node script">
        <b>Script 06 · insights.py</b>
        7-day overview · steps vs weather · gyro by activity
        activity heatmap · collinearity proof (r=0.969)
      </div>
    </div>
  </div>

  <div class="dfd-layer"><div class="dfd-label"></div>
    <div class="dfd-nodes"><div class="dfd-arrow" style="flex:1;text-align:center">▼ &nbsp; processed CSVs + figures saved to disk &nbsp; ▼</div></div>
  </div>

  <div class="dfd-layer">
    <div class="dfd-label">④ Processed<br>Outputs</div>
    <div class="dfd-nodes">
      <div class="dfd-node output">
        <b>📊 15 Figures (fig01–fig15)</b>
        Barometer LPF · RMS · Band-pass · PSD · Spectrogram
        Battery MA · Baro validation · Steps vs weather
        Regression · Heatmap · 7-day overview · Activity
      </div>
      <div class="dfd-node output">
        <b>🗄️ Processed CSVs</b>
        8 × _clean.csv · weather_daily.csv · baro_1min/hourly
        accel_rms_1min/hourly · battery_1min · network_daily
        correlation_matrix_rho/pvalues · dataset_summary
      </div>
    </div>
  </div>

  <div class="dfd-layer"><div class="dfd-label"></div>
    <div class="dfd-nodes"><div class="dfd-arrow" style="flex:1;text-align:center">▼ &nbsp; loaded at runtime by the dashboard &nbsp; ▼</div></div>
  </div>

  <div class="dfd-layer">
    <div class="dfd-label">⑤ Visuali-<br>sation</div>
    <div class="dfd-nodes">
      <div class="dfd-node dash">
        <b>Script 07 · dashboard/app.py &nbsp;← You are here</b>
        Streamlit + Plotly · 5 interactive pages · GPS map · Day Explorer
        Spearman heatmap · OLS results · DSP figures · Step progress bars
        <span style="display:block;font-size:0.71rem;color:#C4B5FD;margin-top:3px">
        Deployable to Streamlit Cloud · also runs locally on localhost</span>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#475569; font-size:0.75rem; padding:6px'>MobiSense Dashboard · "
        "Vedant Parasrampuria (CID: 06053623) · ELEC70126 · Imperial College London · March 2026</div>",
        unsafe_allow_html=True,
    )

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#3B82F6"
GREEN  = "#22C55E"
AMBER  = "#F59E0B"
RED    = "#EF4444"
PURPLE = "#A855F7"
CYAN   = "#06B6D4"
SLATE  = "#94A3B8"

STEP_GOAL = 8_000   # WHO-recommended daily step threshold

ACTIVITY_COLORS = {
    "stationary": SLATE,
    "walking":    GREEN,
    "running":    RED,
    "automotive": AMBER,
    "cycling":    PURPLE,
}

DAY_LABELS = {
    1: "Day 1 · Fri 13 Mar",
    2: "Day 2 · Sat 14 Mar",
    3: "Day 3 · Sun 15 Mar",
    4: "Day 4 · Mon 16 Mar",
    5: "Day 5 · Tue 17 Mar",
    6: "Day 6 · Wed 18 Mar",
    7: "Day 7 · Thu 19 Mar",
}

# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_baro_hourly():
    df = pd.read_csv(PROC / "baro_hourly.csv", parse_dates=["local_time"])
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_baro_1min():
    df = pd.read_csv(PROC / "baro_1min.csv", parse_dates=["local_time"])
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_accel_hourly():
    df = pd.read_csv(PROC / "accel_rms_hourly.csv", parse_dates=["local_time"])
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_accel_1min():
    df = pd.read_csv(PROC / "accel_rms_1min.csv", parse_dates=["local_time"])
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_weather_clean():
    df = pd.read_csv(PROC / "weather_clean.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_weather_daily():
    return pd.read_csv(PROC / "weather_daily.csv")

@st.cache_data(show_spinner=False)
def load_battery_1min():
    df = pd.read_csv(PROC / "battery_1min.csv", parse_dates=["local_time"])
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_activity():
    df = pd.read_csv(PROC / "activity_clean.csv", low_memory=False)
    df["local_time"] = pd.to_datetime(df["local_time"], format="ISO8601", utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_location():
    df = pd.read_csv(PROC / "location_clean.csv", low_memory=False)
    df["local_time"] = pd.to_datetime(df["local_time"], format="ISO8601", utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_pedometer():
    df = pd.read_csv(PROC / "pedometer_clean.csv", low_memory=False)
    df["local_time"] = pd.to_datetime(df["local_time"], format="ISO8601", utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_network_daily():
    return pd.read_csv(PROC / "network_daily.csv")

@st.cache_data(show_spinner=False)
def load_corr_rho():
    df = pd.read_csv(PROC / "correlation_matrix_rho.csv", index_col=0)
    return df

@st.cache_data(show_spinner=False)
def load_corr_pval():
    df = pd.read_csv(PROC / "correlation_matrix_pvalues.csv", index_col=0)
    return df

@st.cache_data(show_spinner=False)
def load_dataset_summary():
    return pd.read_csv(PROC / "dataset_summary.csv")

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📱 MobiSense")
    st.caption("7-day iPhone 16 Pro Max · West London · March 2026")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Sensor Explorer", "🔬 DSP Analysis",
         "📈 Correlations", "🔍 Day Explorer"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Device:** iPhone 16 Pro Max")
    st.caption("**Sensors:** Accel · Gyro · Baro · Pedo · Activity · Location · Battery · Network")
    st.caption("**Weather:** Open-Meteo Archive API")
    st.caption("**Period:** 13–20 Mar 2026 (7.21 days)")
    st.caption("**Location:** West London (51.51°N, −0.24°E)")
    st.divider()
    st.markdown("""
<div style='font-size:0.78rem; color:#94A3B8; line-height:1.7'>
    <b style='color:#F1F5F9'>Vedant Parasrampuria</b><br>
    CID: 06053623<br>
    <a href='mailto:vedant.parasrampuria25@imperial.ac.uk'
       style='color:#3B82F6; text-decoration:none'>
       vedant.parasrampuria25@imperial.ac.uk
    </a><br><br>
    ELEC70126 · Mobile Sensing<br>
    MSc AI Applications &amp; Innovation<br>
    IX Programme · Imperial College London
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("<script>document.title = 'MobiSense · Overview'</script>", unsafe_allow_html=True)
    st.title("MobiSense · 7-Day IoT Sensor Study")
    st.markdown(
        "Continuous iPhone sensor data fused with Open-Meteo atmospheric records "
        "to explore how **weather shapes human movement and behaviour**."
    )
    st.markdown("""
<div style='background:#1E293B; border-left:4px solid #3B82F6; padding:10px 16px;
     border-radius:4px; font-size:0.9rem; color:#CBD5E1; margin-bottom:8px'>
    This is my interactive visualisation dashboard for my <b>Mobile Sensing coursework</b>
    at Imperial College London — built on 7 days of continuous iPhone sensor data
    collected and analysed by <b>Vedant Parasrampuria (CID: 06053623)</b>.
    Use the sidebar to explore sensor readings, DSP analysis, statistical correlations,
    and a day-by-day breakdown of movement and weather.
</div>
""", unsafe_allow_html=True)

    # ── How to use ────────────────────────────────────────────────────────────
    with st.expander("👆 How to use this dashboard"):
        hu1, hu2, hu3, hu4, hu5 = st.columns(5)
        for col, icon, title, desc in [
            (hu1, "🏠", "Overview",        "KPI cards, 7-day sensor timeline, GPS movement map, and daily weather snapshot"),
            (hu2, "📊", "Sensor Explorer", "Interactive time series for all 8 sensors + weather. Filter by day range."),
            (hu3, "🔬", "DSP Analysis",    "Digital signal processing: filter outputs, step detection, spectrogram"),
            (hu4, "📈", "Correlations",    "Spearman heatmap, OLS regression, significance testing"),
            (hu5, "🔍", "Day Explorer",    "Select any of 7 days for a full sensor + weather + GPS breakdown"),
        ]:
            col.markdown(f"""
<div style="background:#1E293B; border-radius:8px; padding:10px 12px;">
  <div style="font-size:1.1rem; margin-bottom:4px">{icon}</div>
  <div style="font-weight:700; font-size:0.88rem; color:#F1F5F9; margin-bottom:6px">{title}</div>
  <div style="font-size:0.78rem; color:#94A3B8; line-height:1.5">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    wd  = load_weather_daily()
    net = load_network_daily()
    ped = load_pedometer()

    daily_steps_ser = ped.groupby("day")["daily_steps"].first()
    total_steps     = int(daily_steps_ser.sum())
    mean_steps      = int(daily_steps_ser.mean())
    max_steps_day   = int(daily_steps_ser.idxmax())
    max_steps       = int(daily_steps_ser.max())
    pressure_rise   = round(wd["mean_pressure_hpa"].iloc[-1] - wd["mean_pressure_hpa"].iloc[0], 1)
    total_rain      = round(wd["total_precip_mm"].sum(), 1)
    sunny_days      = int((wd["total_sunshine_s"] > 20_000).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Steps",    f"{total_steps:,}",          "7 days")
    c2.metric("Daily Average",  f"{mean_steps:,} steps",     f"Peak: Day {max_steps_day}")
    c3.metric("Peak Day",       f"{max_steps:,} steps",      "Wed 18 Mar")

    c4, c5, c6 = st.columns(3)
    c4.metric("Pressure Rise",  f"+{pressure_rise} hPa",    "998 → 1027 hPa")
    c5.metric("Total Rainfall", f"{total_rain} mm",          "Days 1–3")
    c6.metric("Sunny Days",     f"{sunny_days} / 7",         "Days 2, 4–7")

    st.divider()

    # ── Daily step progress bars ──────────────────────────────────────────────
    st.markdown(f'<p class="section-header">Daily Step Count vs {STEP_GOAL:,}-Step Goal</p>', unsafe_allow_html=True)
    GOAL = STEP_GOAL
    daily_step_rows = daily_steps_ser.reset_index()
    daily_step_rows.columns = ["day", "steps"]
    pb_cols = st.columns(7)
    for i, (_, row) in enumerate(daily_step_rows.iterrows()):
        d, s = int(row["day"]), int(row["steps"])
        pct = min(s / GOAL, 1.0)
        goal_met = s >= GOAL
        with pb_cols[i]:
            st.markdown(
                f"<div style='text-align:center; font-size:0.75rem; color:#94A3B8; margin-bottom:2px'>"
                f"Day {d}</div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)
            colour = "#22C55E" if goal_met else "#F59E0B"
            label  = "✅ Goal!" if goal_met else f"{int(pct*100)}%"
            st.markdown(
                f"<div style='text-align:center; font-size:0.82rem; font-weight:700; color:{colour}'>"
                f"{s:,}<br><span style='font-size:0.7rem'>{label}</span></div>",
                unsafe_allow_html=True,
            )
    st.caption(f"Goal: {GOAL:,} steps/day (WHO-recommended threshold) · Green = goal achieved · Amber = below goal · 5/7 days met target")

    st.divider()

    # ── Dataset summary table ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Dataset Summary — All Sensors</p>', unsafe_allow_html=True)
    ds = load_dataset_summary()
    st.dataframe(ds, use_container_width=True, hide_index=True)
    st.caption("All data collected via Sensor Logger app on iPhone 16 Pro Max · West London · 13–20 March 2026")

    st.divider()

    # ── 7-day multi-panel overview ────────────────────────────────────────────
    st.markdown('<p class="section-header">7-Day Timeline</p>', unsafe_allow_html=True)

    baro_h  = load_baro_hourly()
    accel_h = load_accel_hourly()
    bat     = load_battery_1min()
    wc      = load_weather_clean()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.22, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[
            "Accelerometer RMS Energy (g) — Hourly",
            "Atmospheric Pressure (hPa) — Filtered",
            "Temperature (°C) & Precipitation (mm)",
            "Battery Level (%)",
        ],
    )

    # Panel 1: Accel RMS
    fig.add_trace(go.Scatter(
        x=accel_h["local_time"], y=accel_h["rms_g"],
        mode="lines", name="Accel RMS",
        line=dict(color=BLUE, width=1.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
    ), row=1, col=1)

    # Panel 2: Baro pressure
    fig.add_trace(go.Scatter(
        x=baro_h["local_time"], y=baro_h["pressure_hpa"],
        mode="lines", name="Pressure",
        line=dict(color=CYAN, width=1.5),
    ), row=2, col=1)

    # Panel 3: Temp
    fig.add_trace(go.Scatter(
        x=wc["timestamp_utc"], y=wc["temperature_c"],
        mode="lines", name="Temperature",
        line=dict(color=AMBER, width=1.5),
    ), row=3, col=1)
    # Rain bars
    rain_nonzero = wc[wc["precipitation_mm"] > 0]
    fig.add_trace(go.Bar(
        x=rain_nonzero["timestamp_utc"], y=rain_nonzero["precipitation_mm"],
        name="Precipitation", marker_color=f"rgba(6,182,212,0.5)",
        yaxis="y5",
    ), row=3, col=1)

    # Panel 4: Battery
    fig.add_trace(go.Scatter(
        x=bat["local_time"], y=bat["battery_pct"],
        mode="lines", name="Battery %",
        line=dict(color=GREEN, width=1),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
    ), row=4, col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#F1F5F9", size=11),
        margin=dict(l=60, r=20, t=40, b=20),
    )
    for i in range(1, 5):
        fig.update_yaxes(gridcolor="#1E293B", row=i, col=1)
        fig.update_xaxes(gridcolor="#1E293B", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── GPS trace map ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">GPS Movement Trace — All 7 Days</p>', unsafe_allow_html=True)
    st.markdown(
        "Every recorded GPS fix for the 7-day period. "
        "Colour encodes recording day (1=purple → 7=yellow). Home location marked ●."
    )

    loc = load_location()
    loc_valid = loc.dropna(subset=["latitude", "longitude"])
    loc_valid = loc_valid[
        (loc_valid["latitude"].between(51.3, 51.7)) &
        (loc_valid["longitude"].between(-0.5, 0.1))
    ].copy()

    fig_map = px.scatter_mapbox(
        loc_valid,
        lat="latitude", lon="longitude",
        color="day",
        color_continuous_scale="plasma",
        opacity=0.55,
        zoom=11,
        center=dict(lat=51.5128, lon=-0.2353),
        mapbox_style="open-street-map",
        hover_data={"day": True, "local_time": True, "latitude": ":.5f", "longitude": ":.5f"},
        labels={"day": "Day"},
    )
    # Home marker
    fig_map.add_trace(go.Scattermapbox(
        lat=[51.5128], lon=[-0.2353],
        mode="markers+text",
        marker=dict(size=14, color=RED, symbol="circle"),
        text=["🏠 Home"],
        textposition="top right",
        textfont=dict(size=12, color="white"),
        name="Home",
        showlegend=False,
    ))
    fig_map.update_layout(
        height=520,
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Day", tickvals=list(range(1, 8))),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Daily weather snapshot ────────────────────────────────────────────────
    st.markdown('<p class="section-header">Daily Weather & Activity Snapshot</p>', unsafe_allow_html=True)

    daily_steps_full = daily_steps_ser.reset_index()
    daily_steps_full.columns = ["day", "daily_steps"]
    snap = wd.merge(daily_steps_full, on="day")
    snap = snap.merge(net[["day", "pct_home"]], on="day")

    for _, row in snap.iterrows():
        sun_h = row["total_sunshine_s"] / 3600
        d = int(row["day"])
        icon = "☀️" if sun_h > 5 else ("🌤️" if sun_h > 2 else ("🌧️" if row["total_precip_mm"] > 1 else "☁️"))
        with st.expander(f"{icon}  {DAY_LABELS[d]}  —  {row['dominant_description']}  |  "
                         f"{int(row['daily_steps']):,} steps  |  "
                         f"{row['mean_temp_c']:.1f}°C  |  "
                         f"{sun_h:.1f}h sunshine"):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Steps",    f"{int(row['daily_steps']):,}")
            dc2.metric("Temp",     f"{row['mean_temp_c']:.1f}°C",  f"max {row['max_temp_c']:.1f}°C")
            dc3.metric("Rain",     f"{row['total_precip_mm']:.1f} mm")
            dc4, dc5 = st.columns(2)
            dc4.metric("Sunshine", f"{sun_h:.1f} h")
            dc5.metric("At Home",  f"{row['pct_home']:.0f}%")

    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SENSOR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sensor Explorer":
    st.markdown("<script>document.title = 'MobiSense · Sensor Explorer'</script>", unsafe_allow_html=True)
    st.title("Sensor Explorer")
    st.markdown("Interactive time series for every sensor. Use the controls to filter by sensor and zoom into any time window.")

    sensor = st.selectbox(
        "Select sensor",
        ["Accelerometer RMS Energy", "Gyroscope RMS Energy",
         "Atmospheric Pressure (Barometer)", "Battery Level",
         "Pedometer (Daily Steps)", "Activity Distribution",
         "Location (GPS Speed)", "Network (WiFi vs Cellular)",
         "Weather Variables"],
    )

    # Day range filter
    day_range = st.slider("Day range", 1, 7, (1, 7))
    d_start, d_end = day_range

    def day_mask_dt(df, time_col, d_start, d_end):
        """Return boolean mask for rows in day range (1-indexed from 2026-03-13)."""
        t0 = pd.Timestamp("2026-03-13", tz="UTC") + pd.Timedelta(days=d_start - 1)
        t1 = pd.Timestamp("2026-03-13", tz="UTC") + pd.Timedelta(days=d_end)
        return (df[time_col] >= t0) & (df[time_col] < t1)

    # ── Accelerometer RMS ─────────────────────────────────────────────────────
    if sensor == "Accelerometer RMS Energy":
        resolution = st.radio("Resolution", ["1-minute bins", "Hourly bins"], horizontal=True)
        df = load_accel_1min() if resolution == "1-minute bins" else load_accel_hourly()
        df = df[day_mask_dt(df, "local_time", d_start, d_end)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["local_time"], y=df["rms_g"],
            mode="lines", name="RMS (g)",
            line=dict(color=BLUE, width=1 if resolution == "1-minute bins" else 2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        ))

        # Shade activity periods
        act = load_activity()
        act = act[day_mask_dt(act, "local_time", d_start, d_end)]
        walking_bouts = act[act["activity"].isin(["walking", "running"])]

        # Overlay activity as colour bands (sample the 1-min resolution)
        fig.update_layout(
            title=f"Accelerometer RMS Energy — {resolution}",
            xaxis_title="Time (UTC)",
            yaxis_title="RMS Energy (g)",
            height=420,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean RMS",  f"{df['rms_g'].mean():.3f} g")
        col2.metric("Peak RMS",  f"{df['rms_g'].max():.3f} g")
        col3.metric("Std Dev",   f"{df['rms_g'].std():.3f} g")

        st.markdown(
            '<div class="callout">📐 <b>Method:</b> RMS = √(mean(x²)) over non-overlapping windows. '
            'Higher values indicate more vigorous movement. Peak at 5.53 g corresponds to running bouts.</div>',
            unsafe_allow_html=True,
        )

    # ── Barometer ─────────────────────────────────────────────────────────────
    elif sensor == "Atmospheric Pressure (Barometer)":
        resolution = st.radio("Resolution", ["1-minute filtered", "Hourly means"], horizontal=True)
        df = load_baro_1min() if resolution == "1-minute filtered" else load_baro_hourly()
        df = df[day_mask_dt(df, "local_time", d_start, d_end)]

        wc = load_weather_clean()
        wc_filt = wc[day_mask_dt(wc, "timestamp_utc", d_start, d_end)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["local_time"], y=df["pressure_hpa"],
            mode="lines", name="iPhone Barometer (LPF)",
            line=dict(color=CYAN, width=1.5),
        ))
        if resolution == "Hourly means":
            fig.add_trace(go.Scatter(
                x=wc_filt["timestamp_utc"], y=wc_filt["pressure_hpa"],
                mode="lines+markers", name="Open-Meteo Weather API",
                line=dict(color=AMBER, width=2, dash="dot"),
                marker=dict(size=5),
            ))

        fig.update_layout(
            title="Atmospheric Pressure — iPhone vs Open-Meteo API",
            xaxis_title="Time (UTC)",
            yaxis_title="Pressure (hPa)",
            height=420,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            legend=dict(bgcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)

        df_w = load_baro_hourly()
        st.markdown(
            '<div class="callout-green">✅ <b>Validation:</b> iPhone barometer vs Open-Meteo API '
            'Pearson r = 0.9968 (p = 1.93×10⁻¹⁸⁹). Mean offset = +1.09 hPa (~9 m altitude difference). '
            'Rising trend: 998.5 → 1026.6 hPa = incoming high-pressure system.</div>',
            unsafe_allow_html=True,
        )

    # ── Battery ───────────────────────────────────────────────────────────────
    elif sensor == "Battery Level":
        bat = load_battery_1min()
        bat = bat[day_mask_dt(bat, "local_time", d_start, d_end)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bat["local_time"], y=bat["battery_pct"],
            mode="lines", name="Battery %",
            line=dict(color=GREEN, width=1),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
        ))
        # 60-min rolling mean
        bat_sorted = bat.sort_values("local_time").copy()
        bat_sorted["ma60"] = bat_sorted["battery_pct"].rolling(60, center=True).mean()
        fig.add_trace(go.Scatter(
            x=bat_sorted["local_time"], y=bat_sorted["ma60"],
            mode="lines", name="60-min Moving Average",
            line=dict(color=AMBER, width=2.5),
        ))

        fig.add_hline(y=20, line_dash="dash", line_color=RED, annotation_text="Low battery")
        fig.update_layout(
            title="Battery Level with 60-Minute Moving Average",
            xaxis_title="Time (UTC)",
            yaxis_title="Battery (%)",
            yaxis_range=[0, 105],
            height=420,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            legend=dict(bgcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="callout">🔋 Sawtooth pattern: steady drain during active periods, '
            'vertical jumps when charging. 60-min moving average (orange) smooths rapid fluctuations '
            'to reveal daily usage rhythm.</div>',
            unsafe_allow_html=True,
        )

    # ── Activity distribution ─────────────────────────────────────────────────
    elif sensor == "Activity Distribution":
        act = load_activity()
        act_filt = act[act["day"].between(d_start, d_end)]

        counts = act_filt["activity"].value_counts().reset_index()
        counts.columns = ["activity", "count"]
        counts["pct"] = 100 * counts["count"] / counts["count"].sum()
        counts["color"] = counts["activity"].map(
            lambda a: ACTIVITY_COLORS.get(a, SLATE)
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels=counts["activity"],
                values=counts["count"],
                hole=0.45,
                marker_colors=counts["color"].tolist(),
                textinfo="label+percent",
                textfont_size=12,
            ))
            fig_pie.update_layout(
                title="Activity Type Distribution",
                height=480,
                paper_bgcolor="#0F172A",
                font=dict(color="#F1F5F9", size=13),
                showlegend=True,
                legend=dict(bgcolor="#1E293B", font=dict(size=12)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Activity by hour heatmap
            act_filt2 = act_filt.copy()
            act_filt2["hour"] = act_filt2["local_time"].dt.hour
            act_filt2["is_active"] = act_filt2["activity"].isin(["walking", "running"]).astype(int)
            hm = act_filt2.groupby(["day", "hour"])["is_active"].mean().reset_index()
            hm_pivot = hm.pivot(index="day", columns="hour", values="is_active").fillna(0)

            fig_hm = go.Figure(go.Heatmap(
                z=hm_pivot.values,
                x=[f"{h:02d}:00" for h in hm_pivot.columns],
                y=[DAY_LABELS.get(d, f"Day {d}") for d in hm_pivot.index],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Active fraction"),
                zmin=0, zmax=1,
            ))
            fig_hm.update_layout(
                title="Active Hours Heatmap (Walking + Running fraction)",
                height=480,
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                font=dict(color="#F1F5F9", size=12),
                xaxis=dict(tickangle=45, tickfont_size=10),
                yaxis=dict(tickfont_size=11),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Weather variables ─────────────────────────────────────────────────────
    elif sensor == "Weather Variables":
        wc = load_weather_clean()
        wc_filt = wc[day_mask_dt(wc, "timestamp_utc", d_start, d_end)]

        var = st.selectbox(
            "Weather variable",
            ["temperature_c", "humidity_pct", "wind_speed_ms",
             "precipitation_mm", "sunshine_duration_s", "cloud_cover_pct",
             "solar_radiation_wm2", "apparent_temperature_c"],
            format_func=lambda v: {
                "temperature_c": "Temperature (°C)",
                "humidity_pct": "Humidity (%)",
                "wind_speed_ms": "Wind Speed (m/s)",
                "precipitation_mm": "Precipitation (mm/h)",
                "sunshine_duration_s": "Sunshine Duration (s/h)",
                "cloud_cover_pct": "Cloud Cover (%)",
                "solar_radiation_wm2": "Solar Radiation (W/m²)",
                "apparent_temperature_c": "Apparent Temperature (°C)",
            }[v],
        )

        # Convert sunshine from seconds to minutes for readability
        plot_data = wc_filt.copy()
        y_col = var
        y_label = {
            "temperature_c": "Temperature (°C)",
            "humidity_pct": "Humidity (%)",
            "wind_speed_ms": "Wind Speed (m/s)",
            "precipitation_mm": "Precipitation (mm/h)",
            "sunshine_duration_s": "Sunshine Duration (min/h)",
            "cloud_cover_pct": "Cloud Cover (%)",
            "solar_radiation_wm2": "Solar Radiation (W/m²)",
            "apparent_temperature_c": "Apparent Temperature (°C)",
        }[var]
        if var == "sunshine_duration_s":
            plot_data = plot_data.copy()
            plot_data["sunshine_min"] = plot_data[var] / 60
            y_col = "sunshine_min"

        fig = go.Figure()
        if var == "precipitation_mm":
            fig.add_trace(go.Bar(
                x=plot_data["timestamp_utc"], y=plot_data[y_col],
                name=y_label, marker_color=CYAN,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=plot_data["timestamp_utc"], y=plot_data[y_col],
                mode="lines", name=y_label,
                line=dict(color=AMBER, width=2),
                fill="tozeroy" if var != "temperature_c" else None,
                fillcolor="rgba(245,158,11,0.1)",
            ))

        fig.update_layout(
            title=f"Open-Meteo: {y_label} — Days {d_start}–{d_end}",
            xaxis_title="Time (UTC)",
            yaxis_title=y_label,
            height=420,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
        )
        if var == "sunshine_duration_s":
            st.caption("ℹ️ Open-Meteo reports sunshine as seconds of sunshine within each hourly bucket (max = 3600 s = 60 min). Displayed here in minutes for readability.")
        st.plotly_chart(fig, use_container_width=True)

    # ── Gyroscope RMS ─────────────────────────────────────────────────────────
    elif sensor == "Gyroscope RMS Energy":
        st.markdown(
            "Gyroscope captures **rotational dynamics** (rad/s). RMS computed the same way as "
            "accelerometer — proven collinear (r = 0.969) so used separately for rotational insight."
        )
        accel_h = load_accel_hourly()
        accel_h = accel_h[day_mask_dt(accel_h, "local_time", d_start, d_end)]

        # We don't have a separate gyro_rms file, so show accel RMS with explanation
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=accel_h["local_time"], y=accel_h["rms_g"],
            mode="lines", name="Accel RMS (g)",
            line=dict(color=PURPLE, width=1.5),
            fill="tozeroy", fillcolor="rgba(168,85,247,0.12)",
        ))
        fig.update_layout(
            title="Accelerometer RMS — Gyroscope is Collinear (r = 0.969, p ≈ 0)",
            xaxis_title="Time (UTC)", yaxis_title="RMS Energy (g)",
            height=380,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="callout">🔗 <b>Collinearity note:</b> Gyroscope RMS and accelerometer RMS '
            'share Pearson r = 0.9693 (p ≈ 0) — they carry virtually identical information. '
            'Gyroscope was used in Script 06 for rotational dynamics by activity type '
            '(boxplot by stationary / walking / running). See DSP Analysis → Step Detection '
            'and Correlations pages for those figures.</div>',
            unsafe_allow_html=True,
        )
        img = Image.open(FIGS / "fig13_gyro_vs_accel_rms.png")
        st.image(img, caption="Fig 13: Gyro vs Accel RMS — dual-axis overlay (r = 0.969)", use_container_width=True)
        img = Image.open(FIGS / "fig14_gyro_by_activity.png")
        st.image(img, caption="Fig 14: Gyro RMS boxplot by activity type", use_container_width=True)

    # ── Pedometer ─────────────────────────────────────────────────────────────
    elif sensor == "Pedometer (Daily Steps)":
        ped = load_pedometer()
        daily = ped.groupby("day")["daily_steps"].first().reset_index()
        wd = load_weather_daily()
        daily = daily.merge(wd[["day", "total_sunshine_s", "dominant_description", "total_precip_mm"]], on="day")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[DAY_LABELS[d] for d in daily["day"]],
            y=daily["daily_steps"],
            marker_color=[
                f"rgba(245,158,11,{min(1.0, row['total_sunshine_s']/43200)})"
                for _, row in daily.iterrows()
            ],
            text=[f"{int(s):,}" for s in daily["daily_steps"]],
            textposition="outside",
            name="Daily Steps",
        ))
        fig.update_layout(
            title="Daily Step Count (colour intensity = sunshine hours)",
            xaxis_title="Day", yaxis_title="Steps",
            height=420,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Steps",   f"{int(daily['daily_steps'].sum()):,}")
        col2.metric("Peak Day",      f"{int(daily['daily_steps'].max()):,}",  "Day 6 · Wed 18 Mar")
        col3.metric("Lowest Day",    f"{int(daily['daily_steps'].min()):,}",  "Day 3 · Sun 15 Mar")
        st.markdown(
            '<div class="callout-green">👣 Pedometer counts are <b>cumulative within each session</b> '
            '— daily_steps = max − min per session, then summed across sessions per day. '
            'Total over 7 days: <b>61,851 steps</b>.</div>',
            unsafe_allow_html=True,
        )

    # ── Location / GPS Speed ──────────────────────────────────────────────────
    elif sensor == "Location (GPS Speed)":
        loc = load_location()
        loc_filt = loc[loc["day"].between(d_start, d_end)].copy()
        loc_filt = loc_filt.dropna(subset=["latitude", "longitude"])
        loc_filt = loc_filt[
            (loc_filt["latitude"].between(51.3, 51.7)) &
            (loc_filt["longitude"].between(-0.5, 0.1))
        ]

        col1, col2 = st.columns([1, 1])
        with col1:
            # Speed over time (where available)
            loc_speed = loc_filt.dropna(subset=["speed"])
            fig_spd = go.Figure()
            fig_spd.add_trace(go.Scatter(
                x=loc_speed["local_time"], y=loc_speed["speed"],
                mode="lines", name="Speed (m/s)",
                line=dict(color=GREEN, width=1),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
            ))
            fig_spd.update_layout(
                title="GPS Speed (m/s)",
                xaxis_title="Time (UTC)", yaxis_title="Speed (m/s)",
                height=340,
                paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                font=dict(color="#F1F5F9"),
                xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_spd, use_container_width=True)

        with col2:
            # GPS map for the selected day range
            fig_map = px.scatter_mapbox(
                loc_filt,
                lat="latitude", lon="longitude",
                color="day",
                color_continuous_scale="plasma",
                opacity=0.6,
                zoom=11,
                center=dict(lat=51.5128, lon=-0.2353),
                mapbox_style="open-street-map",
                labels={"day": "Day"},
            )
            fig_map.add_trace(go.Scattermapbox(
                lat=[51.5128], lon=[-0.2353],
                mode="markers",
                marker=dict(size=12, color=RED),
                name="Home", showlegend=False,
            ))
            fig_map.update_layout(
                height=340,
                paper_bgcolor="#0F172A",
                font=dict(color="#F1F5F9"),
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.caption(f"{len(loc_filt):,} GPS fixes in selected day range. "
                   "Sentinel values (−1.0) for speed/bearing replaced with NaN in preprocessing.")

    # ── Network ───────────────────────────────────────────────────────────────
    elif sensor == "Network (WiFi vs Cellular)":
        net = load_network_daily()

        fig_net = go.Figure()
        fig_net.add_trace(go.Bar(
            x=[DAY_LABELS[d] for d in net["day"]],
            y=net["wifi_samples"],
            name="WiFi samples", marker_color=BLUE,
        ))
        fig_net.add_trace(go.Bar(
            x=[DAY_LABELS[d] for d in net["day"]],
            y=net["cellular_samples"],
            name="Cellular samples", marker_color=AMBER,
        ))
        fig_net.update_layout(
            barmode="stack",
            title="Daily Network Samples — WiFi vs Cellular",
            xaxis_title="Day", yaxis_title="Sample count",
            height=380,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            legend=dict(bgcolor="#1E293B"),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=[DAY_LABELS[d] for d in net["day"]],
            y=net["pct_home"],
            mode="lines+markers+text",
            text=[f"{v:.0f}%" for v in net["pct_home"]],
            textposition="top center",
            name="% at home (WiFi)",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=9),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
        ))
        fig_pct.update_layout(
            title="% Time at Home per Day (WiFi = at home proxy)",
            yaxis_title="% at home", yaxis_range=[0, 105],
            height=300,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_pct, use_container_width=True)
        st.markdown(
            '<div class="callout">📶 <b>at_home flag:</b> WiFi connection used as proxy for being home. '
            '88% of WiFi samples confirmed home IP (192.168.0.x). Sunny days show lower % at home '
            '(ρ = −0.821, p = 0.023) — more outdoor excursions on good weather days.</div>',
            unsafe_allow_html=True,
        )
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DSP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 DSP Analysis":
    st.markdown("<script>document.title = 'MobiSense · DSP Analysis'</script>", unsafe_allow_html=True)
    st.title("DSP Analysis")
    st.markdown(
        "Digital Signal Processing results: **Butterworth LPF** on barometer, "
        "**RMS energy estimation**, **band-pass filtering + Welch PSD** for step detection, "
        "**STFT spectrogram**, and **battery moving average**."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌡️ Barometer LPF",
        "⚡ Accel RMS",
        "👟 Step Detection",
        "🌈 Spectrogram",
        "🔋 Battery MA",
    ])

    # ── Tab 1: Barometer LPF ──────────────────────────────────────────────────
    with tab1:
        img = Image.open(FIGS / "fig01_barometer_filter.png")
        st.image(img, caption="Fig 01: 7-day barometer — raw vs LPF (top); Day 1 zoom (bottom)", use_container_width=True)
        st.markdown("### Butterworth Low-Pass Filter")
        st.markdown("""
**Purpose:** Remove high-frequency measurement noise from barometric pressure readings while preserving the true atmospheric trend.

| Parameter | Value |
|---|---|
| Filter type | Butterworth (maximally flat) |
| Order | 4 |
| Cut-off frequency | 0.01 Hz |
| Sample rate (fₛ) | ~0.94 Hz |
| Nyquist (fₙ) | 0.47 Hz |
| Implementation | `filtfilt` (zero-phase) |

**Result:** Noise RMS removed = **0.026 hPa**. Atmospheric trend rises from 998.5 → 1026.6 hPa (+28.1 hPa) confirming an approaching high-pressure system.

**Justification:** Meteorological pressure changes occur on timescales of hours–days (< 0.0003 Hz). Cutoff at 0.01 Hz is 30× below Nyquist, safely in the stop-band for thermal / vibration noise.
""")
        st.markdown(
            '<div class="callout-green">✅ Zero-phase filtering (filtfilt) avoids group-delay artefacts critical for accurate timing of pressure events.</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 2: Accel RMS ──────────────────────────────────────────────────────
    with tab2:
        img = Image.open(FIGS / "fig02_accel_rms.png")
        st.image(img, caption="Fig 02: 7-day accelerometer RMS energy (1-minute windows)", use_container_width=True)
        st.markdown("### RMS Energy Estimation")
        st.markdown("""
**Purpose:** Collapse 10 Hz triaxial accelerometer into a scalar activity intensity signal for temporal analysis.

$$\\text{RMS}(t) = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N} x_i^2}$$

| Parameter | Value |
|---|---|
| Sample rate | 10 Hz |
| Window size | 60 s = 600 samples |
| Overlap | None (non-overlapping) |
| Output rate | 1 sample/minute |
| Mean RMS | 0.45 g |
| Peak RMS | 5.53 g |

**Why x only?** Dominant locomotion axis; y/z add collinear noise. RMS over magnitude (√(x²+y²+z²)) also computed — results equivalent after normalisation.

**Reading the plot:** Spikes = walking/running bouts. Flat near-zero stretches = stationary/sleeping. Day 6 (Wed 18 Mar) shows the most prominent peaks — corresponding to the highest recorded step count (11,297 steps).
""")

    # ── Tab 3: Step Detection ─────────────────────────────────────────────────
    with tab3:
        img_bp = Image.open(FIGS / "fig03a_accel_bandpass.png")
        st.image(img_bp, caption="Fig 03a: Raw vs band-passed accelerometer (15-min walking window)", use_container_width=True)
        img_psd = Image.open(FIGS / "fig03b_accel_psd.png")
        st.image(img_psd, caption="Fig 03b: Welch PSD — step frequency peak at 1.963 Hz", use_container_width=True)
        st.markdown("### Band-Pass + Welch PSD")
        st.markdown("""
**Purpose:** Isolate human locomotion frequencies and identify dominant step cadence.

**Band-pass filter (0.5–4 Hz)**
- Lower cut-off 0.5 Hz: removes slow body sway and gravity DC component
- Upper cut-off 4 Hz: removes high-frequency sensor noise; step harmonics die off above 3 Hz
- Human walking range: 1.6–2.0 Hz; running: 2.5–3.5 Hz

**Welch Power Spectral Density**
- nperseg = 2048 samples (204.8 s window)
- Hann window (sidelobe suppression)
- 50% overlap for smoother estimate

**Results:**
| Metric | Value |
|---|---|
| Dominant frequency | **1.963 Hz** |
| Cadence | **117.8 steps/min** |
| Classification | Brisk walking |
| 2nd harmonic | 3.93 Hz (visible in PSD) |
""")
        st.markdown(
            '<div class="callout-amber">🔑 Key finding: The iPhone accelerometer alone is sufficient to detect gait cadence at 117.8 steps/min — matching clinical norms for brisk walking (110–130 steps/min). No dedicated pedometer hardware needed.</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 4: Spectrogram ────────────────────────────────────────────────────
    with tab4:
        img = Image.open(FIGS / "fig04_accel_spectrogram.png")
        st.image(img, caption="Fig 04: STFT spectrogram — Day 6 (most active day, 11,297 steps)", use_container_width=True)
        st.markdown("### Short-Time Fourier Transform (STFT)")
        st.markdown("""
**Purpose:** Reveal how frequency content evolves over time — distinguishing walking bouts, stationary periods, and high-intensity activity.

| Parameter | Value |
|---|---|
| Data | Day 6 (Wed 18 Mar) — most active day |
| nperseg | 512 samples (51.2 s per window) |
| noverlap | 384 samples (75% overlap) |
| Window function | Hann |
| Frequency resolution | 0.0195 Hz |
| Time resolution | 12.8 s |
| Total time bins | 6,782 |

**How to read this plot:**
- **Colour (bright yellow → dark purple):** Energy level at that frequency and moment in time
- **Horizontal bright band at ~1.963 Hz:** Sustained walking bout — the dominant step frequency
- **Dark/purple regions:** Stationary or sleeping (little to no movement energy)
- **Vertical bright flashes:** Brief high-intensity events (stair climbing, running)
- **Low-frequency energy (< 0.5 Hz):** Body sway, postural shifts — not locomotion

**Why Day 6?** 11,297 steps (highest of the week) → richest locomotion signal, most informative for spectral analysis.
""")

    # ── Tab 5: Battery MA ─────────────────────────────────────────────────────
    with tab5:
        img = Image.open(FIGS / "fig05_battery_smoothed.png")
        st.image(img, caption="Fig 05: 7-day battery with 60-min moving average and charging periods shaded", use_container_width=True)
        st.markdown("### Moving Average — Battery Level")
        st.markdown("""
**Purpose:** Smooth high-frequency charging fluctuations to reveal the true daily drain rate and charging behaviour over 7 days.

| Parameter | Value |
|---|---|
| Window | 60 minutes = 60 samples |
| Type | Simple moving average (equal weights) |
| Centred | Yes (symmetric, no phase lag) |
| Charging detection | batteryState == 'charging' |

**Key insights from the plot:**
- **Sawtooth pattern:** Gradual drain during active use → sudden vertical rise when charging begins
- **Drain rate:** ~1%/hour during normal use; slightly faster on Day 6 (most outdoor activity)
- **Charging events:** Overnight top-ups visible as sharp rises; brief daytime charges during sedentary periods
- **Continuous recording:** Battery never reached 0% — all 7 days of sensor data are complete with no gaps due to power loss

**Why 60-minute window?** Charging transients last < 5 minutes (need smoothing); daily usage cycles last 12–24 hours (must be preserved). 60 min is the ideal compromise.
""")
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Correlations":
    st.markdown("<script>document.title = 'MobiSense · Correlations'</script>", unsafe_allow_html=True)
    st.title("Cross-Correlation & Statistics")
    st.markdown(
        "Sensor metrics correlated with weather variables using statistically appropriate methods "
        "at two temporal resolutions: **daily (n=7, Spearman)** and **hourly (n=173, OLS regression)**."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Baro Validation",
        "🕸️ Spearman Heatmap",
        "📉 OLS Regression",
        "📐 Key Findings",
    ])

    # ── Tab 1: Baro vs Weather cross-correlation ───────────────────────────────
    with tab1:
        img = Image.open(FIGS / "fig06_baro_vs_weather.png")
        st.image(img, caption="Fig 06: iPhone barometer vs Open-Meteo API pressure (hourly, n=173)", use_container_width=True)
        st.markdown("### Instrument Validation — iPhone Barometer vs Open-Meteo API")
        st.markdown("""
The iPhone barometer is validated against a calibrated independent reference (Open-Meteo ECMWF reanalysis data for West London).

| Statistic | Value | Interpretation |
|---|---|---|
| Method | Pearson correlation | Appropriate: large n=173, both continuous |
| r | **0.9968** | Near-perfect linear agreement |
| p-value | 1.93 × 10⁻¹⁸⁹ | Astronomically significant |
| Mean offset | +1.09 hPa | iPhone reads ~1 hPa higher than API |
| Altitude implied | ~9 m above sea level | Consistent with West London elevation |
| Lag | 0 hours | No time misalignment between datasets |

**What the +1.09 hPa offset means:** Atmospheric pressure decreases ~0.12 hPa per metre of altitude. A +1.09 hPa offset implies the recording device is ~9 m above the reference level used by Open-Meteo — consistent with being inside a building at ground floor in West London. This is a systematic calibration offset, not measurement error.

**Cross-correlation result:** The zero-lag peak in the cross-correlation confirms the two time series are perfectly time-aligned — no clock drift or API timestamp issues between the iPhone sensor and the weather API.
""")

    # ── Tab 2: Spearman heatmap ────────────────────────────────────────────────
    with tab2:
        rho  = load_corr_rho()
        pval = load_corr_pval()

        # Interactive heatmap with significance annotations
        sig_text = []
        for i in range(rho.shape[0]):
            row_text = []
            for j in range(rho.shape[1]):
                r   = rho.iloc[i, j]
                p   = pval.iloc[i, j]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                row_text.append(f"ρ={r:.2f}{sig}<br>p={p:.3f}")
            sig_text.append(row_text)

        fig_hm = go.Figure(go.Heatmap(
            z=rho.values,
            x=rho.columns.tolist(),
            y=rho.index.tolist(),
            text=sig_text,
            texttemplate="%{text}",
            textfont_size=8,
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            colorbar=dict(title="Spearman ρ"),
        ))
        fig_hm.update_layout(
            title="Spearman Rank Correlation Matrix — Daily Sensor Metrics × Weather Variables<br>"
                  "<sup>* p<0.05  ** p<0.01  *** p<0.001</sup>",
            height=560,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9", size=10),
            xaxis=dict(tickangle=35),
            margin=dict(l=140, r=20, t=80, b=100),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Interpretation guide
        st.markdown("#### How to Read This Heatmap")
        gi1, gi2, gi3 = st.columns(3)
        gi1.markdown("""
<div style="background:#1E3A5F;padding:10px;border-radius:6px;font-size:0.85rem">
<b style="color:#3B82F6">🔵 Blue cells (ρ > 0)</b><br>
<b>Positive correlation</b> — as one variable increases, the other tends to increase too.<br>
e.g. steps ↑ when temperature ↑
</div>""", unsafe_allow_html=True)
        gi2.markdown("""
<div style="background:#3B1A1A;padding:10px;border-radius:6px;font-size:0.85rem">
<b style="color:#EF4444">🔴 Red cells (ρ < 0)</b><br>
<b>Negative correlation</b> — as one variable increases, the other tends to decrease.<br>
e.g. steps ↓ when rainfall ↑
</div>""", unsafe_allow_html=True)
        gi3.markdown("""
<div style="background:#1E293B;padding:10px;border-radius:6px;font-size:0.85rem">
<b style="color:#94A3B8">⬜ White/pale cells (ρ ≈ 0)</b><br>
<b>No correlation</b> — the two variables move independently of each other.<br><br>
<b>Strength:</b> |ρ| > 0.7 = strong · 0.4–0.7 = moderate · < 0.4 = weak
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:8px;font-size:0.82rem;color:#94A3B8">
<b>Significance stars:</b> &nbsp;* p&lt;0.05 (95% confidence) &nbsp;·&nbsp; ** p&lt;0.01 (99%) &nbsp;·&nbsp; *** p&lt;0.001 (99.9%) &nbsp;·&nbsp; No star = not statistically significant (could be chance)
</div>
""", unsafe_allow_html=True)

        # Significant pairs table
        st.markdown("#### Statistically Significant Pairs (p < 0.05, n = 7)")
        rows = []
        for i in rho.index:
            for j in rho.columns:
                p = pval.loc[i, j]
                r = rho.loc[i, j]
                if p < 0.05 and i != j:
                    rows.append({
                        "Sensor Metric": i,
                        "Weather Variable": j,
                        "Spearman rho": f"{r:.3f}",
                        "p-value": f"{p:.3f}",
                        "Direction": "positive" if r > 0 else "negative",
                        "Strength": "Strong" if abs(r) >= 0.7 else ("Moderate" if abs(r) >= 0.4 else "Weak"),
                        "Sig.": "***" if p < 0.001 else ("**" if p < 0.01 else "*"),
                    })
        if rows:
            df_sig = pd.DataFrame(rows).drop_duplicates()
            st.dataframe(df_sig, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="callout">📐 <b>Method justification:</b> Spearman rank correlation chosen over Pearson because '
            'n=7 (insufficient power to verify normality assumption) and daily aggregates may not be linearly related. '
            'Spearman is distribution-free and robust to outliers — appropriate for small samples.</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 3: OLS Regression ─────────────────────────────────────────────────
    with tab3:
        img = Image.open(FIGS / "fig09_regression.png")
        st.image(img, caption="Fig 09: OLS multiple regression — accel RMS vs weather predictors (n=173 hourly)", use_container_width=True)
        st.markdown("### OLS Multiple Linear Regression")
        st.markdown("""
**What this tests:** Does weather (temperature, humidity, wind, rain, sunshine) predict how physically active I am hour-by-hour?

**Setup:**
- Dependent variable: Hourly accelerometer RMS energy (g) — proxy for physical activity intensity
- Predictors: 5 weather variables from Open-Meteo API
- Sample size: n = 173 hourly observations (7 days × ~24 hours)

| Statistic | Value | Meaning |
|---|---|---|
| R² | **0.1613** | Weather explains 16% of hourly activity variation |
| Adjusted R² | 0.1413 | Adjusted for number of predictors |
| F-statistic | **8.075** | Overall model is statistically significant |
| p(F) | 5.59 × 10⁻⁶ | Far below 0.001 threshold |
| Only significant predictor | **Temperature** | β = +0.316, p < 0.001 |
| Humidity, wind, rain, sunshine | Not significant | p > 0.05 |

**What does R² = 0.16 mean?** Weather explains 16% of the variation in hourly activity — the remaining 84% comes from non-weather factors like work schedule, sleep, meals, and personal choices. This is actually a meaningful result: even with just 5 atmospheric variables you can significantly predict activity levels.

**Why OLS (not Spearman) here?** With n=173 hourly observations, we have sufficient degrees of freedom for a 5-predictor regression. The Central Limit Theorem ensures normality of residuals at this sample size. OLS is the appropriate method here — Spearman was used only for the n=7 daily analysis where normality cannot be verified.
""")

    # ── Tab 4: Key Findings summary ───────────────────────────────────────────
    with tab4:
        img = Image.open(FIGS / "fig07_daily_steps_weather.png")
        st.image(img, caption="Fig 07: Daily steps vs weather conditions (bars = steps, overlaid weather vars)", use_container_width=True)
        img = Image.open(FIGS / "fig08_daily_all_correlations.png")
        st.image(img, caption="Fig 08: Pairwise scatter grid — all daily sensor metrics vs weather variables", use_container_width=True)

        st.markdown("### Key Statistical Findings")
        st.markdown("""
<div style="background:#1E293B; padding:10px 14px; border-radius:6px; margin-bottom:14px; font-size:0.85rem">
<b>Colour guide:</b> &nbsp;
<span style="color:#22C55E">■ Green border</span> = <b>positive correlation</b> (variables move together — more of one = more of the other) &nbsp;·&nbsp;
<span style="color:#EF4444">■ Red border</span> = <b>negative correlation</b> (variables move opposite — more of one = less of the other) &nbsp;·&nbsp;
All findings below have p &lt; 0.05 (statistically significant, n = 7 days, Spearman)
</div>
""", unsafe_allow_html=True)

        findings = [
            ("🌧️ Steps ↔ Precipitation", "ρ = −0.768", "p = 0.044 *", False,
             "More rainfall → significantly fewer daily steps.",
             "Rainy days 1–3 averaged <b>6,920 steps/day</b>. Dry days 4–7 averaged <b>10,273 steps/day</b> — a 48% increase. "
             "Rain is a strong deterrent to outdoor walking."),
            ("🌡️ Steps ↔ Temperature", "ρ = +0.786", "p = 0.036 *", True,
             "Warmer days → significantly more daily steps.",
             "Each additional 1°C is associated with approximately <b>~400 more daily steps</b>. "
             "Coldest day (Day 2, 5.9°C): 9,713 steps. Warmest day (Day 6, 12.0°C): 11,297 steps."),
            ("🏃 Active fraction ↔ Temperature", "ρ = +0.821", "p = 0.023 *", True,
             "Higher temperature → greater fraction of the day spent walking or running.",
             "This goes beyond just step count — on warmer days the <b>proportion of time</b> actively moving increases, "
             "not just total steps. Suggests temperature lowers barriers to sustained outdoor activity."),
            ("🏠 At-home % ↔ Sunshine", "ρ = −0.821", "p = 0.023 *", False,
             "Sunnier days → LESS time at home (more outdoor excursions).",
             "On the sunniest days (Days 5–7, 8–12h sunshine), I spent <b>41–59% of time at home</b>. "
             "On overcast Day 1 (2.7h sunshine): <b>80% at home</b>. Sunshine is a clear motivator to go outside."),
            ("💨 At-home % ↔ Wind speed", "ρ = +0.857", "p = 0.014 *", True,
             "Windier days → more time at home. Wind deters outdoor movement.",
             "Even on dry days, higher wind speeds correlate with staying in. "
             "This is independent of temperature and rain — wind alone is a significant behavioural deterrent."),
            ("📊 Baro pressure ↔ Weather API", "r = +0.9968", "p ≈ 0", True,
             "iPhone barometer validated as precision scientific instrument.",
             "Near-perfect agreement with Open-Meteo ECMWF reanalysis (r = 0.9968). "
             "Systematic offset of +1.09 hPa explained by ~9m altitude difference. "
             "This validates all barometer-derived findings in this study."),
        ]

        for title, rho_val, p_val, positive, headline, detail in findings:
            border = "#22C55E" if positive else "#EF4444"
            st.markdown(f"""
<div style="background:#1E293B; border-left:4px solid {border};
     padding:12px 16px; border-radius:4px; margin-bottom:12px;">
<b style="font-size:1.0rem">{title}</b><br>
<span style="color:#3B82F6; font-size:1.15rem; font-weight:700">{rho_val}</span>
&nbsp;&nbsp;<span style="color:#94A3B8; font-size:0.9rem">{p_val}</span><br>
<span style="color:#F1F5F9; font-size:0.92rem"><b>{headline}</b></span><br>
<span style="font-size:0.86rem; color:#CBD5E1; line-height:1.5">{detail}</span>
</div>
""", unsafe_allow_html=True)
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DAY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Day Explorer":
    st.markdown("<script>document.title = 'MobiSense · Day Explorer'</script>", unsafe_allow_html=True)
    st.title("Day Explorer")
    st.markdown("Select any recording day for a complete sensor + weather breakdown.")

    selected_day = st.selectbox(
        "Select day",
        options=list(DAY_LABELS.keys()),
        format_func=lambda d: DAY_LABELS[d],
    )

    # Day boundaries
    T0 = pd.Timestamp("2026-03-13", tz="UTC")
    day_start = T0 + pd.Timedelta(days=selected_day - 1)
    day_end   = T0 + pd.Timedelta(days=selected_day)

    def clip_day(df, col):
        return df[(df[col] >= day_start) & (df[col] < day_end)].copy()

    # ── Weather header for this day ───────────────────────────────────────────
    wd = load_weather_daily()
    wrow = wd[wd["day"] == selected_day].iloc[0]
    wc   = load_weather_clean()
    wc_d = clip_day(wc, "timestamp_utc")

    sun_h = wrow["total_sunshine_s"] / 3600
    icon  = "☀️" if sun_h > 5 else ("🌤️" if sun_h > 2 else ("🌧️" if wrow["total_precip_mm"] > 1 else "☁️"))

    st.subheader(f"{icon} {DAY_LABELS[selected_day]} — {wrow['dominant_description']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Temp",   f"{wrow['mean_temp_c']:.1f}°C",    f"max {wrow['max_temp_c']:.1f}°C")
    m2.metric("Rain",        f"{wrow['total_precip_mm']:.1f} mm")
    m3.metric("Sunshine",    f"{sun_h:.1f} h")

    m4, m5, m6 = st.columns(3)
    m4.metric("Wind",        f"{wrow['mean_wind_ms']:.1f} m/s",  f"max {wrow['max_wind_ms']:.1f} m/s")
    m5.metric("Cloud Cover", f"{wrow['mean_cloud_pct']:.0f}%")
    m6.metric("Pressure",    f"{wrow['mean_pressure_hpa']:.1f} hPa")

    st.divider()

    # ── Steps this day ────────────────────────────────────────────────────────
    ped = load_pedometer()
    ped_d = ped[ped["day"] == selected_day]
    day_steps = int(ped_d["daily_steps"].iloc[0]) if len(ped_d) > 0 else 0

    net = load_network_daily()
    net_d = net[net["day"] == selected_day]
    pct_home = net_d["pct_home"].iloc[0] if len(net_d) > 0 else 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Steps", f"{day_steps:,}")
    s2.metric("At Home",     f"{pct_home:.0f}%",  "WiFi fraction")
    s3.metric("Solar",       f"{wrow['mean_solar_wm2']:.0f} W/m²")

    # Step progress bar
    GOAL = STEP_GOAL
    pct_goal = min(day_steps / GOAL, 1.0)
    goal_met = day_steps >= GOAL
    bar_colour = "#22C55E" if goal_met else "#F59E0B"
    st.markdown(
        f"<div style='font-size:0.82rem; color:#94A3B8; margin-top:10px; margin-bottom:2px'>"
        f"Daily step goal progress ({day_steps:,} / {GOAL:,} steps)</div>",
        unsafe_allow_html=True,
    )
    st.progress(pct_goal)
    st.markdown(
        f"<div style='font-size:0.85rem; font-weight:700; color:{bar_colour}; margin-bottom:6px'>"
        f"{'✅ 8,000-step goal achieved!' if goal_met else f'⚠️ {GOAL - day_steps:,} steps short of 8,000-step goal'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Activity timeline ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Activity Timeline</p>', unsafe_allow_html=True)

    act = load_activity()
    act_d = clip_day(act, "local_time")

    if len(act_d) > 0:
        # 5-min bins
        act_d = act_d.sort_values("local_time").copy()
        act_d["bin"] = act_d["local_time"].dt.floor("5min")
        mode_per_bin = act_d.groupby("bin")["activity"].agg(
            lambda x: x.mode()[0] if len(x) > 0 else "stationary"
        ).reset_index()
        mode_per_bin.columns = ["time", "activity"]
        mode_per_bin["color"] = mode_per_bin["activity"].map(
            lambda a: ACTIVITY_COLORS.get(a, SLATE)
        )
        mode_per_bin["numeric"] = mode_per_bin["activity"].map(
            {"stationary": 0, "automotive": 1, "walking": 2, "cycling": 3, "running": 4}
        ).fillna(0)

        fig_act = go.Figure()
        for act_type, col in ACTIVITY_COLORS.items():
            mask = mode_per_bin["activity"] == act_type
            if mask.any():
                fig_act.add_trace(go.Scatter(
                    x=mode_per_bin.loc[mask, "time"],
                    y=mode_per_bin.loc[mask, "numeric"],
                    mode="markers",
                    name=act_type.capitalize(),
                    marker=dict(color=col, size=6, symbol="square"),
                ))

        fig_act.update_layout(
            title="Activity Classification (5-minute bins)",
            xaxis_title="Time (UTC)",
            yaxis=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["Stationary", "Automotive", "Walking", "Cycling", "Running"],
                gridcolor="#1E293B",
            ),
            xaxis=dict(gridcolor="#1E293B"),
            height=300,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            legend=dict(bgcolor="#1E293B"),
        )
        st.plotly_chart(fig_act, use_container_width=True)

    # ── Sensor panel for the day ───────────────────────────────────────────────
    st.markdown('<p class="section-header">Sensor Readings</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    accel_1m = load_accel_1min()
    accel_d  = clip_day(accel_1m, "local_time")

    with col1:
        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(
            x=accel_d["local_time"], y=accel_d["rms_g"],
            mode="lines", name="Accel RMS",
            line=dict(color=BLUE, width=1.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        ))
        fig_ac.update_layout(
            title="Accelerometer RMS Energy",
            xaxis_title="Time (UTC)", yaxis_title="RMS (g)",
            height=280,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_ac, use_container_width=True)

    baro_1m = load_baro_1min()
    baro_d  = clip_day(baro_1m, "local_time")

    with col2:
        fig_br = go.Figure()
        fig_br.add_trace(go.Scatter(
            x=baro_d["local_time"], y=baro_d["pressure_hpa"],
            mode="lines", name="Pressure",
            line=dict(color=CYAN, width=1.5),
        ))
        fig_br.add_trace(go.Scatter(
            x=wc_d["timestamp_utc"], y=wc_d["pressure_hpa"],
            mode="markers", name="Open-Meteo",
            marker=dict(color=AMBER, size=7, symbol="circle"),
        ))
        fig_br.update_layout(
            title="Atmospheric Pressure",
            xaxis_title="Time (UTC)", yaxis_title="Pressure (hPa)",
            height=280,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            legend=dict(bgcolor="#1E293B"),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_br, use_container_width=True)

    # Weather for the day
    col3, col4 = st.columns(2)

    with col3:
        fig_tmp = go.Figure()
        fig_tmp.add_trace(go.Scatter(
            x=wc_d["timestamp_utc"], y=wc_d["temperature_c"],
            mode="lines+markers", name="Temperature",
            line=dict(color=AMBER, width=2),
            marker=dict(size=6),
        ))
        fig_tmp.add_trace(go.Bar(
            x=wc_d["timestamp_utc"], y=wc_d["precipitation_mm"],
            name="Rain (mm)", marker_color="rgba(6,182,212,0.5)",
            yaxis="y2",
        ))
        fig_tmp.update_layout(
            title="Temperature & Precipitation",
            xaxis_title="Time (UTC)",
            yaxis=dict(title="Temp (°C)", gridcolor="#1E293B"),
            yaxis2=dict(title="Rain (mm)", overlaying="y", side="right"),
            height=280,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"),
            legend=dict(bgcolor="#1E293B"),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_tmp, use_container_width=True)

    bat = load_battery_1min()
    bat_d = clip_day(bat, "local_time")

    with col4:
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Scatter(
            x=bat_d["local_time"], y=bat_d["battery_pct"],
            mode="lines", name="Battery %",
            line=dict(color=GREEN, width=1.5),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
        ))
        fig_bat.update_layout(
            title="Battery Level",
            xaxis_title="Time (UTC)", yaxis_title="Battery (%)",
            yaxis_range=[0, 105],
            height=280,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_bat, use_container_width=True)

    # ── GPS trace for the day ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">GPS Trace</p>', unsafe_allow_html=True)

    loc = load_location()
    loc_d = clip_day(loc, "local_time")
    loc_d = loc_d.dropna(subset=["latitude", "longitude"])
    loc_d = loc_d[
        (loc_d["latitude"].between(51.3, 51.7)) &
        (loc_d["longitude"].between(-0.5, 0.1))
    ]

    if len(loc_d) > 0:
        # Colour by time progression within the day
        loc_d = loc_d.sort_values("local_time").reset_index(drop=True)
        loc_d["t_norm"] = (loc_d["local_time"] - loc_d["local_time"].iloc[0]).dt.total_seconds()

        fig_gps = px.scatter_mapbox(
            loc_d,
            lat="latitude", lon="longitude",
            color="t_norm",
            color_continuous_scale="Viridis",
            zoom=12,
            center=dict(lat=loc_d["latitude"].mean(), lon=loc_d["longitude"].mean()),
            mapbox_style="open-street-map",
            opacity=0.7,
            hover_data={"local_time": True, "latitude": ":.5f", "longitude": ":.5f"},
        )
        fig_gps.add_trace(go.Scattermapbox(
            lat=[51.5128], lon=[-0.2353],
            mode="markers+text",
            marker=dict(size=14, color=RED),
            text=["🏠 Home"],
            textposition="top right",
            textfont=dict(color="white"),
            showlegend=False,
        ))
        fig_gps.update_layout(
            height=420,
            paper_bgcolor="#0F172A",
            font=dict(color="#F1F5F9"),
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_gps, use_container_width=True)
        dist_km = len(loc_d)  # rough proxy
        st.caption(f"{len(loc_d)} GPS fixes recorded on {DAY_LABELS[selected_day]}")
    else:
        st.info("No GPS fixes recorded on this day.")

    # ── Insights callout ──────────────────────────────────────────────────────
    st.divider()
    insights = {
        1: {
            "headline": "A rainy, low-pressure indoor day — the worst weather of the week.",
            "detail": "6.4 mm of rain and 1000.3 hPa (lowest of the week) kept activity low at 7,943 steps — "
                      "just 57 steps short of the 8,000-step WHO threshold. Spent 80% of the day at home. "
                      "The rising pressure trend that follows this day signals the approaching high-pressure "
                      "system that transforms the weather over Days 2–7.",
        },
        2: {
            "headline": "Weekend activity as the rain clears.",
            "detail": "Pressure rose to 1005 hPa and rain eased significantly (0.7 mm). 9,713 steps on a Saturday — "
                      "a 22% jump from Day 1 despite still-cloudy skies. Shows that even partial improvement in "
                      "weather drives meaningful increases in outdoor movement.",
        },
        3: {
            "headline": "Sunday rest day — fewest steps of the week despite dry conditions.",
            "detail": "Only 3,105 steps — the weekly low. Dry but overcast (6.0h sunshine). The low step count "
                      "on this day despite decent weather suggests a strong weekly schedule effect (Sunday rest). "
                      "Spent 87% of the day at home.",
        },
        4: {
            "headline": "Back to routine — pressure climbs toward 1016 hPa.",
            "detail": "9,461 steps on a Monday as the working week resumes. Pressure now at 1016 hPa (+16 hPa "
                      "since Day 1) with 5.5h sunshine. The high-pressure system is well established, "
                      "driving improved conditions for the rest of the week.",
        },
        5: {
            "headline": "First proper sunshine day — 8,000-step goal smashed with 10,454 steps.",
            "detail": "11.7 hours of sunshine and 10.7°C produced 10,454 steps — 30% above the 8,000-step "
                      "goal. At-home time dropped to 64% — significantly more outdoor time than rainy Days 1–3. "
                      "Demonstrates the sunshine→outdoor activity correlation (ρ = −0.821).",
        },
        6: {
            "headline": "Peak activity day — step cadence of 117.8 steps/min detected by DSP.",
            "detail": "11,297 steps (weekly peak), 12.4h sunshine, 12.0°C (warmest day). Spent only 59% of time "
                      "at home. The Welch PSD analysis of this day's accelerometer data revealed the dominant "
                      "walking frequency at 1.963 Hz = 117.8 steps/min. This day was chosen for STFT spectrogram "
                      "analysis due to its rich locomotion signal.",
        },
        7: {
            "headline": "High-pressure peak — settled anticyclone confirmed by barometer.",
            "detail": "Pressure reached 1026.6 hPa — the highest of the recording period and +28.1 hPa above Day 1. "
                      "9,878 steps on the final day with 8.8h sunshine. The iPhone barometer tracked this entire "
                      "pressure rise with r = 0.9968 agreement against the Open-Meteo API, validating it as a "
                      "precision scientific instrument.",
        },
    }
    info = insights.get(selected_day, {})
    if info:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1E3A5F 0%, #1E293B 100%);
     border: 1px solid #3B82F6; border-left: 5px solid #3B82F6;
     padding: 16px 20px; border-radius: 8px;">
  <div style="font-size:0.78rem; color:#3B82F6; font-weight:700; letter-spacing:0.08em;
              text-transform:uppercase; margin-bottom:6px">💡 Day {selected_day} Key Insight</div>
  <div style="font-size:1.05rem; font-weight:700; color:#F1F5F9; margin-bottom:8px">
    {info['headline']}
  </div>
  <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.6">
    {info['detail']}
  </div>
</div>
""", unsafe_allow_html=True)

    show_footer()
