"""
03_clean.py
MobiSense — Step 3: Clean and preprocess all combined sensor files.

Operations per sensor:
  Accelerometer / Gyroscope : clip x/y/z to ±4σ, add day_of_week
  Barometer                 : clip pressure outliers, add day_of_week
  Pedometer                 : compute net daily_steps per session, add day_of_week
  Activity                  : map 'unknown' → 'stationary', add day_of_week
  Location                  : replace sentinel -1.0 values with NaN, add day_of_week
  Battery                   : convert 0–1 level → 0–100 %, add day_of_week
  Network                   : fill blank ssid/bssid, derive at_home flag, add day_of_week
  Weather                   : add WMO description column, compute daily aggregates

Outputs:
  /processed/*_clean.csv   — cleaned versions of every sensor file
  /processed/weather_daily.csv — one row per day with aggregated weather
  /processed/dataset_summary.csv — high-level dataset statistics table
  /figures/                — empty folder ready for Script 04 plots

Run: python3 03_clean.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR  = Path("/Users/vedant/Downloads/IOT PROJECT/processed")
FIG_DIR   = Path("/Users/vedant/Downloads/IOT PROJECT/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── WMO weather code descriptions ─────────────────────────────────────────────
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(name, parse_time_col='local_time'):
    path = PROC_DIR / f"{name}_combined.csv"
    df = pd.read_csv(path, low_memory=False)
    df[parse_time_col] = pd.to_datetime(df[parse_time_col], format='ISO8601', utc=True)
    return df

def save(df, name):
    path = PROC_DIR / f"{name}_clean.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(df):,} rows)")
    return path

def add_day_of_week(df, time_col='local_time'):
    df['day_of_week'] = df[time_col].dt.day_name()
    return df

def clip_sigma(df, cols, n_sigma=4):
    """Clip columns to mean ± n_sigma standard deviations."""
    for col in cols:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - n_sigma * sigma, mu + n_sigma * sigma)
    return df

# ── Per-sensor cleaning ───────────────────────────────────────────────────────

def clean_imu(name):
    """Accelerometer and Gyroscope: clip outliers, add day_of_week."""
    print(f"\n[ {name} ]")
    # Read in chunks to handle 1.2 GB files without exhausting RAM
    chunks = []
    for chunk in pd.read_csv(PROC_DIR / f"{name}_combined.csv",
                              chunksize=500_000, low_memory=False):
        chunk['local_time'] = pd.to_datetime(chunk['local_time'], format='ISO8601', utc=True)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    before = len(df)
    df = clip_sigma(df, ['x', 'y', 'z'], n_sigma=4)
    df = add_day_of_week(df)
    print(f"  Rows      : {before:,}  (clipped ±4σ on x/y/z — no rows removed)")
    print(f"  x range   : {df['x'].min():.3f} → {df['x'].max():.3f}")
    print(f"  y range   : {df['y'].min():.3f} → {df['y'].max():.3f}")
    print(f"  z range   : {df['z'].min():.3f} → {df['z'].max():.3f}")
    save(df, name)
    return df


def clean_barometer():
    print("\n[ Barometer ]")
    df = load('barometer')

    # Pressure sanity check: sea-level range 950–1050 hPa; remove hardware glitches
    before = len(df)
    df = df[(df['pressure'] > 950) & (df['pressure'] < 1050)].copy()
    removed = before - len(df)

    df = add_day_of_week(df)
    print(f"  Rows      : {len(df):,}  (removed {removed} outliers outside 950–1050 hPa)")
    print(f"  Pressure  : {df['pressure'].min():.2f} → {df['pressure'].max():.2f} hPa")
    save(df, 'barometer')
    return df


def clean_pedometer():
    print("\n[ Pedometer ]")
    df = load('pedometer')

    # 'steps' is cumulative within each session — compute net steps per session
    # then sum per day
    df = df.sort_values(['session', 'local_time']).reset_index(drop=True)

    session_steps = (
        df.groupby(['day', 'session'])['steps']
        .agg(session_steps=lambda s: s.max() - s.min())
        .reset_index()
    )
    daily_steps = (
        session_steps.groupby('day')['session_steps']
        .sum()
        .reset_index()
        .rename(columns={'session_steps': 'daily_steps'})
    )

    # Merge daily_steps back onto full pedometer df
    df = df.merge(daily_steps, on='day', how='left')
    df = add_day_of_week(df)

    print(f"  Rows      : {len(df):,}")
    print(f"  Daily steps summary:")
    for _, row in daily_steps.iterrows():
        print(f"    Day {int(row['day'])}: {int(row['daily_steps']):,} steps")
    save(df, 'pedometer')
    return df, daily_steps


def clean_activity():
    print("\n[ Activity ]")
    df = load('activity')

    before_counts = df['activity'].value_counts().to_dict()
    # 'unknown' at session boundaries = sensor not yet settled → treat as stationary
    df['activity'] = df['activity'].replace('unknown', 'stationary')
    after_counts = df['activity'].value_counts().to_dict()

    df = add_day_of_week(df)
    print(f"  Rows      : {len(df):,}")
    print(f"  Before remap : {before_counts}")
    print(f"  After remap  : {after_counts}")
    save(df, 'activity')
    return df


def clean_location():
    print("\n[ Location ]")
    df = load('location')

    # Sentinel value -1.0 used by Sensor Logger for unavailable readings
    sentinel_cols = ['speed', 'bearing', 'speedAccuracy', 'bearingAccuracy']
    for col in sentinel_cols:
        if col in df.columns:
            n_sentinel = (df[col] == -1.0).sum()
            df[col] = df[col].replace(-1.0, np.nan)
            print(f"  {col:<22}: {n_sentinel:,} sentinels → NaN")

    df = add_day_of_week(df)
    print(f"  Rows      : {len(df):,}")
    print(f"  Lat range : {df['latitude'].min():.4f} → {df['latitude'].max():.4f}")
    print(f"  Lon range : {df['longitude'].min():.4f} → {df['longitude'].max():.4f}")
    save(df, 'location')
    return df


def clean_battery():
    print("\n[ Battery ]")
    df = load('battery')

    # batteryLevel is 0.0–1.0 → convert to 0–100 %
    df['battery_pct'] = (df['batteryLevel'] * 100).round(1)
    df = add_day_of_week(df)

    print(f"  Rows      : {len(df):,}")
    print(f"  Level range: {df['battery_pct'].min():.1f}% → {df['battery_pct'].max():.1f}%")
    charging_hrs = (df['batteryState'] == 'charging').sum()
    print(f"  Charging samples: {charging_hrs:,}")
    save(df, 'battery')
    return df


def clean_network():
    print("\n[ Network ]")
    df = load('network')

    # Fill blank string / NaN ssid and bssid
    df['ssid']  = df['ssid'].fillna('').astype(str).str.strip()
    df['bssid'] = df['bssid'].fillna('').astype(str).str.strip()
    if 'ipAddress' in df.columns:
        df['ipAddress'] = df['ipAddress'].fillna('').astype(str).str.strip()

    # at_home: device is on WiFi (home network proxy)
    # When away, phone falls back to cellular; WiFi = at home or known location
    df['at_home'] = df['type'] == 'wifi'

    wifi_rows     = (df['type'] == 'wifi').sum()
    cellular_rows = (df['type'] == 'cellular').sum()
    df = add_day_of_week(df)

    print(f"  Rows      : {len(df):,}")
    print(f"  WiFi      : {wifi_rows:,}  |  Cellular: {cellular_rows:,}")
    print(f"  at_home   : {df['at_home'].sum():,} samples on WiFi")
    save(df, 'network')
    return df


def clean_weather():
    print("\n[ Weather ]")
    df = pd.read_csv(PROC_DIR / "weather_combined.csv")
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    # WMO descriptions
    df['weather_description'] = df['weather_code'].map(WMO_CODES).fillna("Unknown")

    # Day column (1-indexed from 2026-03-13)
    recording_start = pd.Timestamp("2026-03-13", tz="UTC")
    df['day'] = ((df['timestamp_utc'] - recording_start).dt.days + 1).clip(lower=1)
    df['day_of_week'] = df['timestamp_utc'].dt.day_name()

    print(f"  Rows      : {len(df):,}")
    print(f"  Codes seen: {sorted(df['weather_code'].unique().tolist())}")

    save_path = PROC_DIR / "weather_clean.csv"
    df.to_csv(save_path, index=False)
    print(f"  Saved weather_clean.csv")

    # Daily aggregates
    daily = df.groupby('day').agg(
        mean_pressure_hpa    = ('pressure_hpa',      'mean'),
        mean_temp_c          = ('temperature_c',     'mean'),
        max_temp_c           = ('temperature_c',     'max'),
        min_temp_c           = ('temperature_c',     'min'),
        mean_humidity_pct    = ('humidity_pct',      'mean'),
        total_precip_mm      = ('precipitation_mm',  'sum'),
        mean_wind_ms         = ('wind_speed_ms',     'mean'),
        max_wind_ms          = ('wind_speed_ms',     'max'),
        total_sunshine_s     = ('sunshine_duration_s','sum'),
        mean_solar_wm2       = ('solar_radiation_wm2','mean'),
        mean_cloud_pct       = ('cloud_cover_pct',   'mean'),
        mean_apparent_c      = ('apparent_temperature_c','mean'),
        dominant_description = ('weather_description', lambda x: x.mode()[0]),
    ).reset_index()

    daily_path = PROC_DIR / "weather_daily.csv"
    daily.to_csv(daily_path, index=False)
    print(f"  Saved weather_daily.csv")
    print(f"\n  Daily weather summary:")
    for _, row in daily.iterrows():
        print(f"    Day {int(row['day'])}: {row['mean_pressure_hpa']:.1f} hPa  "
              f"{row['mean_temp_c']:.1f}°C  {row['total_precip_mm']:.1f}mm rain  "
              f"{row['dominant_description']}")
    return df, daily


# ── Dataset summary table ─────────────────────────────────────────────────────
def make_summary(daily_steps):
    print("\n[ Dataset Summary Table ]")

    summary_rows = [
        # Sensor, Source, Sample Rate, Duration, Rows, Notes
        ("Accelerometer", "iPhone 16 Pro Max (MEMS)", "10 Hz", "7.21 days", "6,192,485", "x/y/z in g; ±4σ clipped"),
        ("Gyroscope",     "iPhone 16 Pro Max (MEMS)", "10 Hz", "7.21 days", "6,192,485", "x/y/z in rad/s; ±4σ clipped"),
        ("Barometer",     "iPhone 16 Pro Max (MEMS)", "~1 Hz",  "7.21 days", "579,124",   "Pressure in hPa"),
        ("Pedometer",     "iPhone 16 Pro Max (CoreMotion)", "Event", "7.21 days", "17,819", "Cumulative steps; daily_steps derived"),
        ("Activity",      "iPhone 16 Pro Max (CoreMotion)", "Event", "7.21 days", "65,415",  "stationary/walking/running/cycling"),
        ("Location",      "iPhone 16 Pro Max (GPS/WiFi)", "30 s", "7.21 days", "20,983",  "Lat/lon/speed; −1 sentinels removed"),
        ("Battery",       "iPhone 16 Pro Max",         "~1 Hz",  "7.21 days", "613,889",  "Level 0–1 → %; charging state"),
        ("Network",       "iPhone 16 Pro Max",         "~0.5 Hz","7.21 days", "307,540",  "WiFi/Cellular; at_home flag derived"),
        ("Weather",       "Open-Meteo Archive API",    "Hourly", "7.21 days", "192",       "10 atmospheric variables for London"),
    ]

    df_summary = pd.DataFrame(summary_rows,
        columns=['Sensor', 'Source', 'Sample Rate', 'Duration', 'Rows', 'Notes'])

    # Add daily step counts as a supplementary table
    if daily_steps is not None:
        print("\n  Daily step counts:")
        total = 0
        for _, row in daily_steps.iterrows():
            print(f"    Day {int(row['day'])}: {int(row['daily_steps']):,} steps")
            total += int(row['daily_steps'])
        print(f"    Total : {total:,} steps over 7 days")

    path = PROC_DIR / "dataset_summary.csv"
    df_summary.to_csv(path, index=False)
    print(f"\n  Saved dataset_summary.csv")

    print("\n  Dataset summary:")
    print(df_summary.to_string(index=False))
    return df_summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  MobiSense — Data Cleaning & Preprocessing")
    print("=" * 65)

    clean_imu('accelerometer')
    clean_imu('gyroscope')
    clean_barometer()
    _, daily_steps = clean_pedometer()
    clean_activity()
    clean_location()
    clean_battery()
    clean_network()
    clean_weather()
    make_summary(daily_steps)

    print("\n" + "=" * 65)
    print("  Cleaning complete.")
    print(f"  Figures folder: {FIG_DIR}")
    print(f"  All _clean.csv files saved to: {PROC_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
