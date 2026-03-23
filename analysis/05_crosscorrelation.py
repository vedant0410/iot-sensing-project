"""
05_crosscorrelation.py
MobiSense — Step 5: Cross-Correlation & Statistical Analysis

Analyses performed:
  A. Data preparation
     — Derive daily sensor metrics from all 8 sensors:
       daily_steps (pedometer), pct_active (activity),
       distance_km (location, haversine), battery_drain_pct (battery),
       pct_home (network)
     — Align hourly barometer + accelerometer RMS with hourly weather

  B. Analysis 1 — Barometer vs Open-Meteo pressure (hourly, n≈173)
     — Pearson r + p-value (linear sensor validation)
     — Cross-correlation lag plot (detrended signals)
     — Scatter plot with regression line

  C. Analysis 2 — Daily sensor metrics vs daily weather (n=7)
     — Spearman rank correlation + p-value for every pair
     — Bar+line figure: daily steps vs sunshine duration + precipitation
     — Justified choice: n=7 violates normality assumption for Pearson;
       multivariate regression would overfit (more predictors than df)

  D. Analysis 3 — Hourly RMS vs weather (n≈173, multivariate regression)
     — OLS multiple linear regression: RMS ~ temp + sunshine + precip + wind
     — Reports R², F-statistic, individual coefficient p-values
     — Justified choice: n=173 provides sufficient df; multivariate
       regression disentangles collinear weather predictors

  E. Analysis 4 — Full pairwise correlation heatmap (daily)
     — All sensor-derived daily metrics vs all daily weather variables
     — Annotated with Spearman r and significance stars

Figures saved to /figures/:
  fig06_baro_vs_weather.png        — scatter + cross-correlation lag plot
  fig07_daily_steps_weather.png    — steps vs sunshine and precipitation
  fig08_daily_all_correlations.png — all daily sensor metrics vs weather
  fig09_regression.png             — multivariate regression (hourly RMS)
  fig10_correlation_heatmap.png    — full pairwise heatmap with significance

Run: python3 05_crosscorrelation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import signal, stats
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR = Path("/Users/vedant/Downloads/IOT PROJECT/processed")
FIG_DIR  = Path("/Users/vedant/Downloads/IOT PROJECT/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       150,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'lines.linewidth':   1.4,
    'legend.framealpha': 0.85,
    'legend.fontsize':   9,
})

C_BLUE   = '#2563EB'
C_RED    = '#DC2626'
C_GREEN  = '#16A34A'
C_ORANGE = '#D97706'
C_PURPLE = '#7C3AED'
C_GREY   = '#6B7280'

ALPHA    = 0.05     # significance threshold


# ══════════════════════════════════════════════════════════════════════════════
# A.  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two lat/lon points."""
    R    = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi        = np.radians(lat2 - lat1)
    dlambda     = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def build_daily_sensor_metrics():
    """
    Derive one row per day (days 1–7) from each sensor.

    Metrics:
      daily_steps      — from pedometer_clean (daily_steps column)
      pct_active       — % of activity samples that are walking or running
      distance_km      — total GPS distance per day (haversine, gaps excluded)
      battery_drain    — total battery drop during unplugged periods per day (%)
      pct_home         — % of network samples where at_home == True
    """
    print("  Building daily sensor metrics...")

    # ── daily_steps ────────────────────────────────────────────────────────────
    ped = pd.read_csv(PROC_DIR / "pedometer_clean.csv")
    daily_steps = (ped.groupby('day')['daily_steps']
                      .first()
                      .reset_index()
                      .rename(columns={'daily_steps': 'daily_steps'}))

    # ── pct_active (walking + running fraction per day) ────────────────────────
    act = pd.read_csv(PROC_DIR / "activity_clean.csv",
                      usecols=['day', 'activity'])
    act['is_active'] = act['activity'].isin(['walking', 'running'])
    pct_active = (act.groupby('day')['is_active']
                     .mean()
                     .mul(100)
                     .reset_index()
                     .rename(columns={'is_active': 'pct_active'}))

    # ── distance_km (haversine between consecutive GPS points, within sessions) ─
    loc = pd.read_csv(PROC_DIR / "location_clean.csv",
                      usecols=['local_time', 'latitude', 'longitude',
                               'day', 'session'])
    loc['local_time'] = pd.to_datetime(loc['local_time'],
                                        format='ISO8601', utc=True)
    loc = loc.sort_values('local_time').reset_index(drop=True)

    # Compute time gap and distance between consecutive points
    loc['time_gap_s'] = loc['local_time'].diff().dt.total_seconds()
    loc['dist_km'] = haversine_km(
        loc['latitude'].shift(), loc['longitude'].shift(),
        loc['latitude'],         loc['longitude']
    )
    # Exclude session boundaries (time gap > 5 min) or implausible jumps (> 2 km in 30 s)
    valid = (loc['time_gap_s'] < 300) & (loc['dist_km'] < 2.0)
    distance_km = (loc[valid]
                     .groupby('day')['dist_km']
                     .sum()
                     .reset_index()
                     .rename(columns={'dist_km': 'distance_km'}))

    # ── battery_drain (total drop during unplugged periods per day) ────────────
    bat = pd.read_csv(PROC_DIR / "battery_clean.csv",
                      usecols=['local_time', 'day', 'battery_pct', 'batteryState'])
    bat['local_time'] = pd.to_datetime(bat['local_time'],
                                        format='ISO8601', utc=True)
    bat = bat.sort_values('local_time').reset_index(drop=True)

    # Only count drops during unplugged/full (not charging rises)
    bat['delta'] = bat['battery_pct'].diff()
    bat_drain    = bat[bat['batteryState'].isin(['unplugged', 'full'])]
    battery_drain = (bat_drain[bat_drain['delta'] < 0]
                       .groupby('day')['delta']
                       .sum()
                       .abs()
                       .reset_index()
                       .rename(columns={'delta': 'battery_drain_pct'}))

    # ── pct_home (% of network samples on WiFi per day) ───────────────────────
    net = pd.read_csv(PROC_DIR / "network_clean.csv",
                      usecols=['day', 'at_home'])
    pct_home = (net.groupby('day')['at_home']
                    .mean()
                    .mul(100)
                    .reset_index()
                    .rename(columns={'at_home': 'pct_home'}))

    # ── Merge all into one daily DataFrame ────────────────────────────────────
    daily = daily_steps.copy()
    for df in [pct_active, distance_km, battery_drain, pct_home]:
        daily = daily.merge(df, on='day', how='left')

    # Keep only days 1–7 (full recording days; day 8 is partial)
    daily = daily[daily['day'] <= 7].reset_index(drop=True)

    # Merge with daily weather (days 1–7)
    weather_d = pd.read_csv(PROC_DIR / "weather_daily.csv")
    weather_d = weather_d[weather_d['day'] <= 7].reset_index(drop=True)
    daily = daily.merge(weather_d, on='day', how='left')

    print(f"  Daily metrics shape: {daily.shape}  (7 days × {daily.shape[1]} columns)")
    print(f"\n  Daily metrics overview:")
    cols = ['day', 'daily_steps', 'pct_active', 'distance_km',
            'battery_drain_pct', 'pct_home']
    print(daily[cols].to_string(index=False))

    return daily


def build_hourly_data():
    """
    Align barometer, accelerometer RMS and weather data on the same
    hourly UTC timestamps (inner join — only hours present in all three).
    """
    print("\n  Building hourly aligned dataset...")

    baro = pd.read_csv(PROC_DIR / "baro_hourly.csv")
    baro.columns = ['timestamp_utc', 'baro_pressure_hpa']
    baro['timestamp_utc'] = pd.to_datetime(baro['timestamp_utc'],
                                            format='ISO8601', utc=True)
    baro = baro.set_index('timestamp_utc')

    rms = pd.read_csv(PROC_DIR / "accel_rms_hourly.csv")
    rms.columns = ['timestamp_utc', 'rms_g']
    rms['timestamp_utc'] = pd.to_datetime(rms['timestamp_utc'],
                                           format='ISO8601', utc=True)
    rms = rms.set_index('timestamp_utc')

    wth = pd.read_csv(PROC_DIR / "weather_clean.csv")
    wth['timestamp_utc'] = pd.to_datetime(wth['timestamp_utc'],
                                           format='ISO8601', utc=True)
    wth = wth.set_index('timestamp_utc')

    # Inner join: only hours where all three sources have data
    hourly = baro.join(rms, how='inner').join(wth, how='inner')
    hourly = hourly.dropna(subset=['baro_pressure_hpa', 'rms_g',
                                    'pressure_hpa', 'temperature_c'])

    print(f"  Hourly dataset: {len(hourly)} rows  "
          f"({hourly.index.min().date()} → {hourly.index.max().date()})")
    return hourly


def _sig_stars(p):
    """Return significance stars string for a p-value."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# B.  ANALYSIS 1 — Barometer vs Open-Meteo Pressure  (hourly)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_baro_vs_weather(hourly):
    """
    Validate the iPhone barometer against the Open-Meteo reference pressure.

    Two complementary views:
      1. Pearson correlation + scatter — measures overall linear agreement
      2. Cross-correlation lag plot    — identifies time offset (lead/lag)
         between the two signals after removing the shared long-term trend
         (detrended via scipy.signal.detrend).

    The phone barometer measures station pressure at phone altitude.
    Open-Meteo reports Mean Sea Level (MSL) pressure.  A systematic
    positive offset in the weather data (~1.2 hPa per 10 m altitude) is
    expected and is accounted for in the scatter plot.
    """
    print("\n[ Analysis 1 ] Barometer vs Open-Meteo Pressure")

    x = hourly['pressure_hpa'].values       # Open-Meteo MSL pressure
    y = hourly['baro_pressure_hpa'].values  # iPhone barometer (station)

    r, p = stats.pearsonr(x, y)
    print(f"  Pearson r = {r:.4f}   p = {p:.2e}  {_sig_stars(p)}")
    mean_offset = np.mean(x - y)
    print(f"  Mean MSL − Station offset: {mean_offset:.2f} hPa  "
          f"(≈ {mean_offset/0.12:.0f} m altitude equivalent)")

    # ── Cross-correlation lag (detrended) ──────────────────────────────────────
    x_dt = signal.detrend(x)
    y_dt = signal.detrend(y)
    xcorr = signal.correlate(y_dt, x_dt, mode='full')
    xcorr /= np.sqrt(np.sum(y_dt**2) * np.sum(x_dt**2))
    lags  = signal.correlation_lags(len(y_dt), len(x_dt), mode='full')

    # Restrict to ±24 hour lags
    mask     = np.abs(lags) <= 24
    peak_lag = lags[mask][np.argmax(xcorr[mask])]
    peak_r   = xcorr[mask].max()
    print(f"  Cross-corr peak lag : {peak_lag:+d} h  "
          f"(r = {peak_r:.4f})")

    # ── Figure 6: two-panel — scatter left, lag plot right ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter + regression line
    ax = axes[0]
    ax.scatter(x, y, s=6, alpha=0.4, color=C_BLUE, label='Hourly data points')
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color=C_RED, lw=2, label=f'Linear fit  (r = {r:.3f}{_sig_stars(p)})')
    ax.set_xlabel('Open-Meteo MSL Pressure (hPa)')
    ax.set_ylabel('iPhone Barometer Station Pressure (hPa)')
    ax.set_title('Barometer Sensor Validation\n'
                 'iPhone vs Open-Meteo Reference Pressure')
    ax.legend()
    ax.text(0.05, 0.95,
            f'r = {r:.3f}  p = {p:.1e}\nOffset = {mean_offset:.1f} hPa (altitude)',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

    # Right: cross-correlation lag plot
    ax2 = axes[1]
    ax2.plot(lags[mask], xcorr[mask], color=C_BLUE, lw=1.5)
    ax2.axvline(peak_lag, color=C_RED, lw=2, linestyle='--',
                label=f'Peak lag: {peak_lag:+d} h  (r = {peak_r:.3f})')
    ax2.axhline(0, color=C_GREY, lw=0.8, linestyle=':')
    ax2.axvline(0, color=C_GREY, lw=0.8, linestyle=':')
    ax2.fill_between(lags[mask], xcorr[mask], 0,
                     where=xcorr[mask] > 0, alpha=0.15, color=C_BLUE)
    ax2.set_xlabel('Lag (hours)  [+ve = barometer lags weather]')
    ax2.set_ylabel('Normalised Cross-Correlation')
    ax2.set_title('Cross-Correlation Lag Analysis\n'
                  '(Detrended signals — shared trend removed)')
    ax2.legend()
    ax2.set_xlim(-24, 24)

    plt.tight_layout()
    _save(fig, "fig06_baro_vs_weather.png")

    return r, p, peak_lag


# ══════════════════════════════════════════════════════════════════════════════
# C.  ANALYSIS 2 — Daily Sensor Metrics vs Daily Weather  (n=7, Spearman)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_daily_correlations(daily):
    """
    Spearman rank correlation between each daily sensor metric and each
    daily weather variable.

    Method choice justification:
      n = 7 observations. Pearson correlation requires the data to be drawn
      from a bivariate normal distribution — with n < 30, this assumption
      cannot be tested or relied upon.  Spearman's ρ is non-parametric and
      rank-based, making it robust to the small sample and to any non-linear
      monotonic relationships.  Multivariate regression is not appropriate
      here (7 observations, ~5 predictors → overfit, p << n_features).

    Statistical significance threshold: p < 0.05 (two-tailed).
    Note: with n=7, |ρ| > 0.75 is required to achieve p < 0.05.
    """
    print("\n[ Analysis 2 ] Daily Sensor Metrics vs Daily Weather  (Spearman, n=7)")
    print(f"  Significance threshold: α = {ALPHA}  "
          f"(note: |ρ| > 0.75 required for significance at n=7)")

    sensor_cols  = ['daily_steps', 'pct_active', 'distance_km',
                     'battery_drain_pct', 'pct_home']
    weather_cols = ['total_sunshine_s', 'total_precip_mm', 'mean_temp_c',
                     'mean_apparent_c', 'mean_cloud_pct', 'mean_wind_ms',
                     'mean_pressure_hpa']

    sensor_labels  = {
        'daily_steps':       'Daily Steps',
        'pct_active':        '% Time Active',
        'distance_km':       'Distance (km)',
        'battery_drain_pct': 'Battery Drain (%)',
        'pct_home':          '% Time at Home',
    }
    weather_labels = {
        'total_sunshine_s':  'Total Sunshine (s)',
        'total_precip_mm':   'Precipitation (mm)',
        'mean_temp_c':       'Mean Temp (°C)',
        'mean_apparent_c':   'Feels-Like Temp (°C)',
        'mean_cloud_pct':    'Cloud Cover (%)',
        'mean_wind_ms':      'Wind Speed (m/s)',
        'mean_pressure_hpa': 'Mean Pressure (hPa)',
    }

    results = []
    print(f"\n  {'Sensor Metric':<22} {'Weather Variable':<25} {'ρ':>7} {'p':>9} {'Sig':>4}")
    print("  " + "-" * 70)

    for sc in sensor_cols:
        for wc in weather_cols:
            rho, p = stats.spearmanr(daily[sc].values, daily[wc].values)
            results.append({'sensor': sc, 'weather': wc,
                             'rho': rho, 'p': p, 'sig': p < ALPHA})
            flag = _sig_stars(p)
            print(f"  {sensor_labels[sc]:<22} {weather_labels[wc]:<25} "
                  f"{rho:>+7.3f} {p:>9.3f} {flag:>4}")

    res_df = pd.DataFrame(results)
    sig    = res_df[res_df['sig']]
    print(f"\n  Significant pairs (p < {ALPHA}): {len(sig)}")
    for _, row in sig.iterrows():
        print(f"    {sensor_labels[row['sensor']]} ↔ "
              f"{weather_labels[row['weather']]}: "
              f"ρ = {row['rho']:+.3f}  p = {row['p']:.3f}")

    # ── Figure 7: Daily steps vs sunshine + precipitation ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    days = daily['day'].values

    # Left: steps vs sunshine
    ax = axes[0]
    color_bars = [C_ORANGE if s > 30000 else C_GREY for s in daily['total_sunshine_s']]
    bars = ax.bar(days, daily['total_sunshine_s'] / 3600,
                  color=color_bars, alpha=0.7, label='Sunshine (hours)')
    ax2_left = ax.twinx()
    ax2_left.spines['top'].set_visible(False)
    ax2_left.plot(days, daily['daily_steps'], 'o-',
                  color=C_BLUE, lw=2, ms=7, label='Daily Steps')
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Sunshine (hours)', color=C_ORANGE)
    ax2_left.set_ylabel('Daily Steps', color=C_BLUE)
    ax.set_title('Daily Steps vs Sunshine Duration\n'
                 f'Spearman ρ = {res_df[(res_df.sensor=="daily_steps") & (res_df.weather=="total_sunshine_s")].iloc[0]["rho"]:+.3f}  '
                 f'p = {res_df[(res_df.sensor=="daily_steps") & (res_df.weather=="total_sunshine_s")].iloc[0]["p"]:.3f}')
    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {d}' for d in days])

    # Merge legends
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2_left.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='upper left')

    # Right: steps vs precipitation
    ax = axes[1]
    color_bars2 = [C_BLUE if p > 0 else C_GREY for p in daily['total_precip_mm']]
    ax.bar(days, daily['total_precip_mm'], color=color_bars2, alpha=0.7,
           label='Precipitation (mm)')
    ax2_right = ax.twinx()
    ax2_right.spines['top'].set_visible(False)
    ax2_right.plot(days, daily['daily_steps'], 'o-',
                   color=C_RED, lw=2, ms=7, label='Daily Steps')
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Precipitation (mm)', color=C_BLUE)
    ax2_right.set_ylabel('Daily Steps', color=C_RED)
    ax.set_title('Daily Steps vs Precipitation\n'
                 f'Spearman ρ = {res_df[(res_df.sensor=="daily_steps") & (res_df.weather=="total_precip_mm")].iloc[0]["rho"]:+.3f}  '
                 f'p = {res_df[(res_df.sensor=="daily_steps") & (res_df.weather=="total_precip_mm")].iloc[0]["p"]:.3f}')
    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {d}' for d in days])

    lines3, lab3 = ax.get_legend_handles_labels()
    lines4, lab4 = ax2_right.get_legend_handles_labels()
    ax.legend(lines3 + lines4, lab3 + lab4, loc='upper left')

    plt.tight_layout()
    _save(fig, "fig07_daily_steps_weather.png")

    # ── Figure 8: All sensor metrics vs key weather — multi-panel ─────────────
    key_weather = ['total_sunshine_s', 'total_precip_mm', 'mean_temp_c']
    fig, axes = plt.subplots(len(sensor_cols), len(key_weather),
                              figsize=(14, 14), sharex=False)
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_RED]

    for i, sc in enumerate(sensor_cols):
        for j, wc in enumerate(key_weather):
            ax = axes[i][j]
            row_r = res_df[(res_df.sensor == sc) & (res_df.weather == wc)].iloc[0]
            rho, p = row_r['rho'], row_r['p']

            ax.scatter(daily[wc], daily[sc], s=60, color=colors[i],
                       alpha=0.85, zorder=3)

            # Annotate day numbers
            for _, d_row in daily.iterrows():
                ax.annotate(str(int(d_row['day'])),
                            (d_row[wc], d_row[sc]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=8, color=C_GREY)

            # Trend line if significant
            if row_r['sig']:
                m, b = np.polyfit(daily[wc], daily[sc], 1)
                x_l  = np.linspace(daily[wc].min(), daily[wc].max(), 100)
                ax.plot(x_l, m * x_l + b, '--', color=colors[i], lw=1.4,
                        alpha=0.7)

            sig_str = f'  {_sig_stars(p)}' if p < ALPHA else ''
            ax.set_title(f'ρ = {rho:+.2f}  p = {p:.2f}{sig_str}', fontsize=9)
            if i == len(sensor_cols) - 1:
                ax.set_xlabel(weather_labels[wc], fontsize=9)
            if j == 0:
                ax.set_ylabel(sensor_labels[sc], fontsize=9)

    fig.suptitle('Daily Sensor Metrics vs Weather Variables\n'
                 '(Spearman ρ, n=7 days — day numbers annotated; '
                 'dashed line = significant trend)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, "fig08_daily_all_correlations.png")

    return res_df


# ══════════════════════════════════════════════════════════════════════════════
# D.  ANALYSIS 3 — Hourly RMS vs Weather  (OLS Multiple Regression, n≈173)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_regression(hourly):
    """
    Ordinary Least Squares (OLS) multiple linear regression:
      RMS_activity ~ β₀ + β₁·temperature + β₂·sunshine + β₃·precipitation
                         + β₄·wind_speed + ε

    Method choice justification:
      n = 173 hourly observations.  With ~4 predictors, we have
      n/k ≈ 43 observations per predictor, well above the rule-of-thumb
      minimum of 10.  Multivariate regression is preferred over pairwise
      correlations because sunshine, cloud cover, and temperature are
      mutually correlated (multicollinear). OLS estimates the independent
      contribution of each predictor controlling for the others.

    Collinear predictors excluded from the model:
      cloud_cover_pct   — negatively correlated with sunshine_duration_s
      solar_radiation   — nearly identical information to sunshine_duration_s
      apparent_temp     — correlated with temperature_c (r > 0.95 typically)

    Statistical significance: p < 0.05 for individual coefficients (t-test);
    F-statistic tests whether the model as a whole is significant.
    """
    print("\n[ Analysis 3 ] Hourly RMS vs Weather — OLS Multiple Regression")

    # Predictors: chosen to minimise collinearity
    predictors = {
        'temperature_c':     'Temperature (°C)',
        'sunshine_duration_s': 'Sunshine Duration (s/hr)',
        'precipitation_mm':  'Precipitation (mm)',
        'wind_speed_ms':     'Wind Speed (m/s)',
    }

    X_raw = hourly[list(predictors.keys())].copy()
    y     = hourly['rms_g'].values

    # Standardise predictors for interpretable coefficient magnitudes
    X_std  = (X_raw - X_raw.mean()) / X_raw.std()
    X_sm   = sm.add_constant(X_std)   # add intercept column

    model  = sm.OLS(y, X_sm).fit()

    print(f"  n                 : {len(y)}")
    print(f"  R²                : {model.rsquared:.4f}")
    print(f"  Adjusted R²       : {model.rsquared_adj:.4f}")
    print(f"  F-statistic       : {model.fvalue:.3f}  "
          f"(p = {model.f_pvalue:.2e}  {_sig_stars(model.f_pvalue)})")
    print(f"\n  Coefficients (standardised predictors):")
    print(f"  {'Predictor':<28} {'Coef':>8} {'p':>9} {'Sig':>4}")
    print("  " + "-" * 52)
    for i, (col, label) in enumerate(predictors.items()):
        coef = model.params.iloc[i + 1]
        p    = model.pvalues.iloc[i + 1]
        print(f"  {label:<28} {coef:>+8.4f} {p:>9.3f} {_sig_stars(p):>4}")

    # ── Figure 9: Actual vs Predicted + residuals ──────────────────────────────
    y_pred = model.fittedvalues
    resids = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: actual vs predicted scatter
    ax = axes[0]
    ax.scatter(y, y_pred, s=5, alpha=0.35, color=C_BLUE)
    lim = [min(y.min(), y_pred.min()) - 0.05,
           max(y.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, '--', color=C_RED, lw=1.5, label='Perfect fit (y=x)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual RMS Acceleration (g)')
    ax.set_ylabel('Predicted RMS Acceleration (g)')
    ax.set_title(f'OLS Regression: Actual vs Predicted\n'
                 f'R² = {model.rsquared:.3f}  |  '
                 f'F({int(model.df_model)},{int(model.df_resid)}) = '
                 f'{model.fvalue:.1f}  p = {model.f_pvalue:.2e}')
    ax.legend()
    ax.text(0.05, 0.95,
            '\n'.join([f'{l}: β={model.params.iloc[i+1]:+.3f} '
                       f'({_sig_stars(model.pvalues.iloc[i+1])})'
                       for i, l in enumerate(
                           ['Temp', 'Sunshine', 'Precip', 'Wind'])]),
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    # Right: residual plot
    ax2 = axes[1]
    ax2.scatter(y_pred, resids, s=5, alpha=0.35, color=C_ORANGE)
    ax2.axhline(0, color=C_RED, lw=1.5, linestyle='--')
    ax2.set_xlabel('Predicted RMS (g)')
    ax2.set_ylabel('Residuals (g)')
    ax2.set_title('Residual Plot\n(random scatter around 0 = good model fit)')

    plt.tight_layout()
    _save(fig, "fig09_regression.png")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# E.  ANALYSIS 4 — Full Pairwise Correlation Heatmap  (daily)
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(daily):
    """
    Full pairwise Spearman correlation matrix between all sensor-derived
    daily metrics and all daily weather variables.  Cells are annotated
    with ρ and significance stars.  Colour encodes direction and strength.
    """
    print("\n[ Analysis 4 ] Pairwise Correlation Heatmap (daily, Spearman)")

    sensor_cols = ['daily_steps', 'pct_active', 'distance_km',
                    'battery_drain_pct', 'pct_home']
    weather_cols = ['total_sunshine_s', 'total_precip_mm', 'mean_temp_c',
                     'mean_apparent_c', 'mean_cloud_pct', 'mean_wind_ms',
                     'mean_pressure_hpa']

    sensor_labels = {
        'daily_steps':       'Daily Steps',
        'pct_active':        '% Time Active',
        'distance_km':       'Distance (km)',
        'battery_drain_pct': 'Battery Drain (%)',
        'pct_home':          '% Time at Home',
    }
    weather_labels = {
        'total_sunshine_s':  'Sunshine (s)',
        'total_precip_mm':   'Precipitation (mm)',
        'mean_temp_c':       'Temperature (°C)',
        'mean_apparent_c':   'Feels-Like (°C)',
        'mean_cloud_pct':    'Cloud Cover (%)',
        'mean_wind_ms':      'Wind Speed (m/s)',
        'mean_pressure_hpa': 'Pressure (hPa)',
    }

    rho_matrix = np.zeros((len(sensor_cols), len(weather_cols)))
    p_matrix   = np.ones ((len(sensor_cols), len(weather_cols)))

    for i, sc in enumerate(sensor_cols):
        for j, wc in enumerate(weather_cols):
            rho, p = stats.spearmanr(daily[sc].values, daily[wc].values)
            rho_matrix[i, j] = rho
            p_matrix[i, j]   = p

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(weather_cols)))
    ax.set_yticks(range(len(sensor_cols)))
    ax.set_xticklabels([weather_labels[c] for c in weather_cols],
                        rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels([sensor_labels[c] for c in sensor_cols], fontsize=9)

    # Annotate each cell with ρ and stars
    for i in range(len(sensor_cols)):
        for j in range(len(weather_cols)):
            rho = rho_matrix[i, j]
            p   = p_matrix[i, j]
            stars = _sig_stars(p)
            cell_text = f'{rho:+.2f}\n{stars}'
            color = 'white' if abs(rho) > 0.55 else 'black'
            ax.text(j, i, cell_text, ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=10)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax.set_title('Pairwise Spearman Correlation — Daily Sensor Metrics vs Weather\n'
                 '(n = 7 days | *** p<0.001  ** p<0.01  * p<0.05  ns = not significant)',
                 fontweight='bold', pad=14)

    plt.tight_layout()
    _save(fig, "fig10_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  MobiSense — Cross-Correlation & Statistical Analysis (Script 05)")
    print("=" * 65)

    # A. Prepare data
    print("\n[ A ] Data Preparation")
    daily  = build_daily_sensor_metrics()
    hourly = build_hourly_data()

    # B. Analysis 1: barometer vs weather pressure
    r_baro, p_baro, lag = analyse_baro_vs_weather(hourly)

    # C. Analysis 2: daily Spearman correlations
    daily_results = analyse_daily_correlations(daily)

    # D. Analysis 3: hourly OLS regression
    model = analyse_regression(hourly)

    # E. Analysis 4: full heatmap
    plot_correlation_heatmap(daily)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  Barometer vs Open-Meteo:")
    print(f"    Pearson r = {r_baro:.4f}  p = {p_baro:.2e}  {_sig_stars(p_baro)}")
    print(f"    Cross-corr peak lag = {lag:+d} h")
    print(f"\n  Daily correlations (Spearman, n=7):")
    sig = daily_results[daily_results['sig']]
    if len(sig) > 0:
        for _, row in sig.iterrows():
            print(f"    {row['sensor']:<22} ↔ {row['weather']:<25} "
                  f"ρ={row['rho']:+.3f}  p={row['p']:.3f}  {_sig_stars(row['p'])}")
    else:
        print("    No pairs reached p < 0.05 at n=7  "
              "(expected — low power with small n)")
    print(f"\n  Hourly OLS regression (n={int(model.nobs)}):")
    print(f"    R² = {model.rsquared:.3f}  |  "
          f"F = {model.fvalue:.1f}  p = {model.f_pvalue:.2e}  "
          f"{_sig_stars(model.f_pvalue)}")
    print(f"\n  Figures saved:")
    for f in sorted(FIG_DIR.glob("fig0[6-9]*.png")) + \
             sorted(FIG_DIR.glob("fig1[0]*.png")):
        print(f"    {f.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
