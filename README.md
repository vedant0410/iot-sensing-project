# iot-sensing-project
Smartphone-based IoT sensing platform for personal mobility and environmental monitoring

# MobiSense: Smartphone-Based IoT Sensing Platform

  Personal mobility and environmental monitoring using iPhone sensors
  and Open-Meteo weather data.

  **Module:** ELEC70126 Internet of Things and Applications
  **Institution:** Imperial College London

  ## Data Sources
  - Source 1: iPhone 16 Pro Max (Accelerometer, Gyroscope, Barometer,
    Pedometer, Activity, Location, Battery) via Sensor Logger
  - Source 2: Open-Meteo Historical Weather API
    (pressure, temperature, humidity, wind)

  ## Structure
  - `data/raw/` — daily sensor CSV exports
  - `data/weather/` — Open-Meteo weather data
  - `data/processed/` — cleaned and merged datasets
  - `analysis/` — Python analysis scripts (DSP, cross-correlation, insights)
  - `dashboard/` — Streamlit web dashboard
