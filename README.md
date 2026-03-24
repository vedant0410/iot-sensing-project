# MobiSense: Smartphone-Based IoT Sensing Platform

**Personal mobility and environmental monitoring using iPhone sensors and Open-Meteo weather data.**

| | |
|---|---|
| **Module** | ELEC70126 Internet of Things and Applications |
| **Institution** | Imperial College London, MSc AI Applications & Innovation |
| **Student** | Vedant Parasrampuria (CID: 06053623) |
| **Recording period** | 7 days, 12–18 March 2025 |
| **Live dashboard** | [vedant-iot-dashboard.streamlit.app](https://vedant-iot-dashboard.streamlit.app) |

---

## Project Overview

This project collects, processes, and analyses 7 days of smartphone sensor data from an iPhone 16 Pro Max using the **Sensor Logger** app, combined with historical weather data from the **Open-Meteo API**. The goal is to identify relationships between environmental conditions (pressure, temperature, sunshine) and personal mobility patterns (steps, activity, location).

**Key findings:**
- Atmospheric pressure rose 28.1 hPa over 7 days (998 → 1027 hPa), tracked by the iPhone barometer with r = 0.9968 agreement against Open-Meteo
- Temperature is the only significant weather predictor of hourly activity (β = +0.316, p < 0.001; R² = 0.16)
- Sunshine strongly predicts outdoor time (Spearman ρ = −0.821, p = 0.023)
- Walking cadence of 117.8 steps/min detected from accelerometer PSD peak at 1.963 Hz

---

## Repository Structure

```
iot-sensing-project/
├── analysis/
│   ├── 01_concatenate.py       # Merge daily Sensor Logger CSVs per sensor
│   ├── 02_fetch_weather.py     # Fetch Open-Meteo historical weather API
│   ├── 03_clean.py             # Clean, resample, and align all sensors
│   ├── 04_dsp.py               # DSP: PSD, STFT, bandpass, step detection
│   ├── 05_crosscorrelation.py  # Spearman correlations + OLS regression
│   └── 06_insights.py          # Summary figures and daily overview plots
├── dashboard/
│   ├── app.py                  # Streamlit dashboard (Script 07)
│   └── .streamlit/config.toml  # Dark theme configuration
├── figures/                    # 15 output figures (PNG) from scripts 04–06
├── processed/                  # Small processed CSVs committed to repo
│   ├── accel_rms_1min.csv      # Accelerometer RMS energy (1-min resampled)
│   ├── baro_1min.csv           # Barometer pressure (1-min resampled)
│   ├── battery_1min.csv        # Battery level (1-min resampled)
│   ├── activity_clean.csv      # Activity classification (cleaned)
│   ├── location_clean.csv      # GPS location (cleaned)
│   ├── pedometer_clean.csv     # Step counts (cleaned)
│   ├── weather_clean.csv       # Hourly weather data
│   ├── weather_daily.csv       # Daily weather summary
│   ├── correlation_matrix_rho.csv    # Spearman correlation matrix
│   └── correlation_matrix_pvalues.csv
├── requirements.txt
└── README.md
```

> **Note:** Large raw and processed CSVs (accelerometer, gyroscope, barometer, battery — totalling ~5GB) are excluded from this repo due to GitHub file size limits. They are available via the Google Drive link below.

---

## Data Sources

| Source | Details |
|---|---|
| **iPhone 16 Pro Max** | Accelerometer, Gyroscope, Barometer, Pedometer, Activity, Location, Battery, Network — collected via [Sensor Logger](https://www.thelasso.app) |
| **Open-Meteo API** | Historical hourly weather: temperature, humidity, wind speed, precipitation, sunshine duration, solar radiation, cloud cover |

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/vedant0410/iot-sensing-project.git
cd iot-sensing-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add large data files
Download the large processed CSVs from Google Drive and place them in `processed/`:
- `accelerometer_clean.csv`
- `gyroscope_clean.csv`
- `barometer_clean.csv`
- `battery_clean.csv`
- `network_clean.csv`

### 4. Run analysis scripts (in order)
```bash
python analysis/01_concatenate.py
python analysis/02_fetch_weather.py
python analysis/03_clean.py
python analysis/04_dsp.py
python analysis/05_crosscorrelation.py
python analysis/06_insights.py
```

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

Or visit the live version: [vedant-iot-dashboard.streamlit.app](https://vedant-iot-dashboard.streamlit.app)

---

## Pipeline Overview

```
iPhone Sensors (Sensor Logger)
        │
        ▼
  Raw CSV exports (daily, per sensor)
        │
        ▼
  01_concatenate.py  ──►  combined CSVs
        │
        ▼
  02_fetch_weather.py ──► weather_clean.csv
        │
        ▼
  03_clean.py  ──►  *_clean.csv + resampled CSVs
        │
        ▼
  04_dsp.py  ──►  Figs 01–05 (PSD, STFT, bandpass, steps)
        │
        ▼
  05_crosscorrelation.py  ──►  Figs 06–10 (correlations, regression)
        │
        ▼
  06_insights.py  ──►  Figs 11–15 (daily overview, activity)
        │
        ▼
  dashboard/app.py  ──►  Interactive Streamlit dashboard
```

---

## Links

| Resource | Link |
|---|---|
| Live Dashboard | [vedant-iot-dashboard.streamlit.app](https://vedant-iot-dashboard.streamlit.app) |
| GitHub Repo | [github.com/vedant0410/iot-sensing-project](https://github.com/vedant0410/iot-sensing-project) |
| Raw + Large Data | [Google Drive — Raw Sensor Logger files](https://drive.google.com/drive/folders/1BZaKtBlSxeQbl19O8JZVS0U18_TbiLw3?usp=sharing) |
