"""
02_fetch_weather.py
MobiSense — Step 2: Fetch historical weather data from Open-Meteo API.

Fetches hourly atmospheric data for the exact recording period and location,
aligned with the sensor dataset for cross-correlation analysis.

Run: python3 02_fetch_weather.py
"""

import requests
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
# Coordinates from Location sensor data (home location, London)
LATITUDE  = 51.5128
LONGITUDE = -0.2353

# Exact recording period
START_DATE = "2026-03-13"
END_DATE   = "2026-03-20"

OUTPUT_DIR = Path("/Users/vedant/Downloads/IOT PROJECT/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Variables to fetch ───────────────────────────────────────────────────────
HOURLY_VARS = [
    "pressure_msl",           # Atmospheric pressure at sea level (hPa) — cross-correlates with barometer
    "temperature_2m",         # Air temperature at 2m (°C)
    "relative_humidity_2m",   # Relative humidity (%)
    "wind_speed_10m",         # Wind speed at 10m (km/h)
    "precipitation",          # Precipitation (mm)
    "weather_code",           # WMO weather code (clear/cloudy/rain/etc)
    "cloud_cover",            # Total cloud cover (%)
    "sunshine_duration",      # Seconds of actual sunshine per hour (0–3600)
    "shortwave_radiation",    # Solar radiation received (W/m²) — proxy for brightness/sunniness
    "apparent_temperature",   # Feels-like temperature (°C) — better behavioural predictor than actual temp
]

def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "hourly":          ",".join(HOURLY_VARS),
        "timezone":        "UTC",
        "wind_speed_unit": "ms",   # m/s to match sensor units
    }

    print(f"Fetching weather data from Open-Meteo...")
    print(f"  Location : {LATITUDE}°N, {LONGITUDE}°E (London)")
    print(f"  Period   : {START_DATE} → {END_DATE}")
    print(f"  Variables: {', '.join(HOURLY_VARS)}\n")

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Build DataFrame
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Rename columns for clarity
    df = df.rename(columns={
        "time":                  "timestamp_utc",
        "pressure_msl":          "pressure_hpa",
        "temperature_2m":        "temperature_c",
        "relative_humidity_2m":  "humidity_pct",
        "wind_speed_10m":        "wind_speed_ms",
        "precipitation":         "precipitation_mm",
        "weather_code":          "weather_code",
        "cloud_cover":           "cloud_cover_pct",
        "sunshine_duration":     "sunshine_duration_s",
        "shortwave_radiation":   "solar_radiation_wm2",
        "apparent_temperature":  "apparent_temperature_c",
    })

    return df

def main():
    print("=" * 60)
    print("  MobiSense — Weather Data Fetch")
    print("=" * 60 + "\n")

    df = fetch_weather()

    print(f"Retrieved {len(df)} hourly records\n")
    print(f"  Period   : {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    print(f"  Pressure : {df['pressure_hpa'].min():.1f} to {df['pressure_hpa'].max():.1f} hPa")
    print(f"  Temp     : {df['temperature_c'].min():.1f} to {df['temperature_c'].max():.1f} °C")
    print(f"  Humidity : {df['humidity_pct'].min():.0f} to {df['humidity_pct'].max():.0f} %")
    print(f"  Wind     : {df['wind_speed_ms'].min():.1f} to {df['wind_speed_ms'].max():.1f} m/s")
    print(f"  Rain days: {(df['precipitation_mm'] > 0).sum()} hours with precipitation")
    print(f"  Sunshine : {df['sunshine_duration_s'].min():.0f} to {df['sunshine_duration_s'].max():.0f} s/hr")
    print(f"  Solar rad: {df['solar_radiation_wm2'].min():.1f} to {df['solar_radiation_wm2'].max():.1f} W/m²")
    print(f"  Feels like: {df['apparent_temperature_c'].min():.1f} to {df['apparent_temperature_c'].max():.1f} °C")
    print(f"  Nulls    : {df.isnull().sum().to_dict()}")

    # Save
    output_path = OUTPUT_DIR / "weather_combined.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Preview
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("  Weather data fetch complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
