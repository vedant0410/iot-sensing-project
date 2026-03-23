"""
06_insights.py
MobiSense — Step 6: Insights & Visualisations

Produces polished figures for the written report, covering:
  Fig 11 — 7-day sensor overview: activity intensity, pressure, temperature,
            precipitation, and battery across the full recording period
  Fig 12 — Daily step count summary: annotated bar chart with weather context
  Fig 13 — Gyroscope vs Accelerometer RMS: proves collinearity, justifies
            exclusion of gyroscope from the regression in Script 05
  Fig 14 — Gyroscope RMS by activity type: shows distinct rotational dynamics
            per activity class (stationary / walking / automotive / running)
  Fig 15 — Activity time-of-day heatmap: when during the day each activity
            type occurs, aggregated across all 7 recording days

All figures saved to /figures/ at 150 DPI.

Run: python3 06_insights.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats
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
    'grid.alpha':        0.22,
    'grid.linewidth':    0.6,
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
C_TEAL   = '#0891B2'

# Activity colours — consistent across all figures
ACT_COLORS = {
    'stationary': '#94A3B8',
    'walking':    '#16A34A',
    'running':    '#DC2626',
    'automotive': '#D97706',
}

# Day date labels (recording period: Mar 13–19)
DAY_DATES = {
    1: 'Mar 13', 2: 'Mar 14', 3: 'Mar 15', 4: 'Mar 16',
    5: 'Mar 17', 6: 'Mar 18', 7: 'Mar 19',
}


def _rms(x):
    """Root Mean Square of array x."""
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else np.nan


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — 7-Day Sensor Overview (4-panel time series)
# ══════════════════════════════════════════════════════════════════════════════

def plot_7day_overview():
    """
    Four-panel time-series figure covering the full 7.21-day recording period:
      Panel 1: Accelerometer RMS energy (1-min windows) — activity intensity
      Panel 2: iPhone barometer vs Open-Meteo pressure — sensor validation
      Panel 3: Hourly temperature (°C) + precipitation (mm) — weather context
      Panel 4: Battery level (%) with 30-min moving average

    Open-Meteo pressure is shifted by −1.09 hPa (altitude correction for ~9 m
    above sea level) so both pressure signals plot on the same scale.
    """
    print("\n[ Fig 11 ] 7-Day Sensor Overview")

    # ── Load data ──────────────────────────────────────────────────────────────
    rms_df = pd.read_csv(PROC_DIR / "accel_rms_1min.csv")
    rms_df['local_time'] = pd.to_datetime(rms_df['local_time'],
                                           format='ISO8601', utc=True)
    rms_df = rms_df.set_index('local_time').sort_index()

    baro_df = pd.read_csv(PROC_DIR / "baro_1min.csv")
    baro_df['local_time'] = pd.to_datetime(baro_df['local_time'],
                                            format='ISO8601', utc=True)
    baro_df = baro_df.set_index('local_time').sort_index()

    wth = pd.read_csv(PROC_DIR / "weather_clean.csv")
    wth['timestamp_utc'] = pd.to_datetime(wth['timestamp_utc'],
                                           format='ISO8601', utc=True)
    wth = wth.set_index('timestamp_utc').sort_index()

    bat = pd.read_csv(PROC_DIR / "battery_clean.csv",
                      usecols=['local_time', 'battery_pct', 'batteryState'])
    bat['local_time'] = pd.to_datetime(bat['local_time'],
                                        format='ISO8601', utc=True)
    bat = bat.set_index('local_time').sort_index()
    bat_1min   = bat['battery_pct'].resample('1min').mean().dropna()
    bat_smooth = bat_1min.rolling(30, center=True, min_periods=5).mean()
    charging   = (bat['batteryState'] == 'charging').resample('1min').max()

    # ── Build figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True,
                              gridspec_kw={'height_ratios': [1.5, 1.2, 1.2, 1.0],
                                           'hspace': 0.08})

    x_min = pd.Timestamp('2026-03-13 09:00', tz='UTC')
    x_max = pd.Timestamp('2026-03-20 15:00', tz='UTC')
    day_starts = pd.date_range('2026-03-13', '2026-03-20', freq='D', tz='UTC')

    def _day_lines(ax):
        for d in day_starts:
            ax.axvline(d, color=C_GREY, lw=0.6, linestyle=':', alpha=0.55)

    # ── Panel 1: Accelerometer RMS ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(rms_df.index, rms_df['rms_g'],
                     alpha=0.35, color=C_GREEN)
    ax1.plot(rms_df.index, rms_df['rms_g'],
             color=C_GREEN, lw=0.5, alpha=0.8)
    ax1.set_ylabel('RMS Accel (g)')
    ax1.set_title('Panel 1 — Physical Activity Intensity (Accelerometer RMS, 1-min windows)')
    _day_lines(ax1)

    # Annotate day numbers mid-day
    for day_n, label in DAY_DATES.items():
        mid = pd.Timestamp(f'2026-03-{12+day_n} 13:00', tz='UTC')
        if mid < x_max:
            ax1.text(mid, ax1.get_ylim()[1] * 0.88 if ax1.get_ylim()[1] > 0 else 0.5,
                     f'Day {day_n}', ha='center', fontsize=8.5,
                     color=C_GREY, style='italic')

    # ── Panel 2: Pressure ─────────────────────────────────────────────────────
    # Open-Meteo MSL pressure corrected to station level (−1.09 hPa for ~9 m)
    ALTITUDE_OFFSET = 1.09  # hPa
    ax2 = axes[1]
    ax2.plot(baro_df.index, baro_df['pressure_hpa'],
             color=C_BLUE, lw=0.7, alpha=0.75,
             label='iPhone Barometer (filtered, ~0.94 Hz)')
    ax2.plot(wth.index, wth['pressure_hpa'] - ALTITUDE_OFFSET,
             color=C_RED, lw=2.0, linestyle='--', alpha=0.85,
             label=f'Open-Meteo (MSL − {ALTITUDE_OFFSET} hPa altitude correction)')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('Panel 2 — Atmospheric Pressure: iPhone Barometer vs Open-Meteo Reference  '
                  '(r = 0.9968, zero lag)')
    ax2.legend(loc='upper left')
    _day_lines(ax2)

    # ── Panel 3: Temperature + Precipitation ──────────────────────────────────
    ax3 = axes[2]
    ax3.plot(wth.index, wth['temperature_c'],
             color=C_ORANGE, lw=1.8, label='Temperature (°C)')
    ax3.set_ylabel('Temperature (°C)', color=C_ORANGE)
    ax3.tick_params(axis='y', colors=C_ORANGE)

    ax3r = ax3.twinx()
    ax3r.bar(wth.index, wth['precipitation_mm'],
             width=0.038, color=C_TEAL, alpha=0.55,
             label='Precipitation (mm)')
    ax3r.set_ylabel('Precipitation (mm)', color=C_TEAL)
    ax3r.tick_params(axis='y', colors=C_TEAL)
    ax3r.spines['top'].set_visible(False)
    ax3r.set_ylim(0, wth['precipitation_mm'].max() * 5)  # compress rain bars

    handles1, lab1 = ax3.get_legend_handles_labels()
    handles2, lab2 = ax3r.get_legend_handles_labels()
    ax3.legend(handles1 + handles2, lab1 + lab2, loc='upper left')
    ax3.set_title('Panel 3 — Weather: Temperature and Precipitation (Open-Meteo, hourly)')
    _day_lines(ax3)

    # ── Panel 4: Battery ───────────────────────────────────────────────────────
    ax4 = axes[3]

    # Shade charging periods
    charging_aligned = charging.reindex(bat_1min.index, fill_value=False)
    edges = charging_aligned.astype(int).diff().fillna(0)
    starts_c = bat_1min.index[edges == 1]
    ends_c   = bat_1min.index[edges == -1]
    if charging_aligned.iloc[0]:
        starts_c = starts_c.insert(0, bat_1min.index[0])
    if charging_aligned.iloc[-1]:
        ends_c = ends_c.append(pd.DatetimeIndex([bat_1min.index[-1]]))

    first = True
    for s, e in zip(starts_c, ends_c):
        ax4.axvspan(s, e, alpha=0.2, color=C_GREEN,
                    label='Charging' if first else '_nolegend_')
        first = False

    ax4.plot(bat_1min.index, bat_1min.values,
             color=C_PURPLE, lw=0.3, alpha=0.25)
    ax4.plot(bat_smooth.index, bat_smooth.values,
             color=C_PURPLE, lw=2.0, label='Battery % (30-min MA)')
    ax4.set_ylabel('Battery (%)')
    ax4.set_xlabel('Date (UTC)')
    ax4.set_title('Panel 4 — Battery Level (green = charging periods)')
    ax4.set_ylim(0, 110)
    ax4.legend(loc='lower right')
    _day_lines(ax4)

    # ── X-axis ticks (shared) ──────────────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].set_xlim(x_min, x_max)

    fig.suptitle('MobiSense: 7-Day Sensor Overview\n'
                 'iPhone 16 Pro Max + Open-Meteo API  |  London, 13–20 March 2026',
                 fontsize=13, fontweight='bold', y=1.005)

    plt.tight_layout()
    _save(fig, 'fig11_7day_overview.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Daily Steps Summary (annotated, report-ready)
# ══════════════════════════════════════════════════════════════════════════════

def plot_daily_steps_summary():
    """
    Clean annotated bar chart showing daily step counts across 7 days,
    with bars coloured by sunshine duration and each bar annotated with:
      - step count (top)
      - sunshine hours and precipitation (below bar)
      - weather description (inside bar)
    A temperature overlay line is included on a secondary y-axis.
    """
    print("\n[ Fig 12 ] Daily Steps Summary")

    ped = pd.read_csv(PROC_DIR / "pedometer_clean.csv")
    daily_steps = ped.groupby('day')['daily_steps'].first()

    wd = pd.read_csv(PROC_DIR / "weather_daily.csv")
    wd = wd[wd['day'] <= 7].set_index('day')

    days        = list(range(1, 8))
    steps       = [int(daily_steps[d]) for d in days]
    sunshine_h  = [wd.loc[d, 'total_sunshine_s'] / 3600 for d in days]
    precip      = [wd.loc[d, 'total_precip_mm'] for d in days]
    temp        = [wd.loc[d, 'mean_temp_c'] for d in days]
    descriptions = [wd.loc[d, 'dominant_description'] for d in days]

    # Colour bars by sunshine (grey=no sun → gold=full sun)
    sun_norm = np.array(sunshine_h)
    sun_norm = (sun_norm - sun_norm.min()) / (sun_norm.max() - sun_norm.min())
    bar_colors = [plt.cm.YlOrBr(0.2 + 0.65 * v) for v in sun_norm]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(days))
    bars = ax.bar(x, steps, color=bar_colors, edgecolor='white',
                  linewidth=1.2, width=0.65, zorder=3)

    # Temperature line overlay
    ax2 = ax.twinx()
    ax2.plot(x, temp, 'o-', color=C_RED, lw=2.0, ms=7,
             label='Mean Temp (°C)', zorder=5)
    ax2.set_ylabel('Mean Temperature (°C)', color=C_RED)
    ax2.tick_params(axis='y', colors=C_RED)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, max(temp) + 5)

    # Annotate each bar
    for i, (bar, s, p, desc, sun) in enumerate(
            zip(bars, steps, precip, descriptions, sunshine_h)):
        h = bar.get_height()
        # Step count on top
        ax.text(bar.get_x() + bar.get_width() / 2, h + 150,
                f'{s:,}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#1E293B')
        # Weather description inside bar (upper third)
        if h > 2000:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 0.78,
                    desc, ha='center', va='center',
                    fontsize=7.5, color='#1E293B', style='italic')
        # Sunshine and precipitation below x-axis (as annotations)
        rain_str = f'☔ {p:.1f}mm' if p > 0 else '✓ No rain'
        ax.text(bar.get_x() + bar.get_width() / 2, -550,
                f'☀ {sun:.1f}h\n{rain_str}',
                ha='center', va='top', fontsize=7.5, color=C_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Day {d}\n{DAY_DATES[d]}' for d in days])
    ax.set_xlabel('Recording Day', labelpad=40)
    ax.set_ylabel('Daily Step Count')
    ax.set_title('Daily Step Count Over 7 Days — Annotated with Weather Context\n'
                 '(bar colour = sunshine level: darker gold = more sunshine)',
                 pad=12)
    ax.set_ylim(-1200, max(steps) + 1500)
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.axhline(np.mean(steps), color=C_GREY, lw=1.2, linestyle='--', alpha=0.7,
               label=f'7-day mean: {int(np.mean(steps)):,} steps')
    ax.legend(loc='upper left')

    # Colourbar for sunshine
    sm = plt.cm.ScalarMappable(cmap='YlOrBr',
                                norm=plt.Normalize(min(sunshine_h), max(sunshine_h)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                         shrink=0.6, pad=0.12, fraction=0.03)
    cbar.set_label('Sunshine (hours)', fontsize=9)

    plt.tight_layout()
    _save(fig, 'fig12_daily_steps_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 + 14 — Gyroscope Analysis (chunked processing)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gyro_rms():
    """
    Load gyroscope_clean.csv in 500,000-row chunks (1.2 GB).
    For each chunk:
      - Compute vector magnitude: |ω| = √(x² + y² + z²)  [rad/s]
    Return a sorted time-indexed Series of magnitudes (for resampling).
    Also assign activity labels via backward merge_asof with activity events.
    """
    print("\n  Computing gyroscope RMS (chunked)...")

    # Load activity events once for labelling
    act = pd.read_csv(PROC_DIR / "activity_clean.csv",
                      usecols=['local_time', 'activity'])
    act['local_time'] = pd.to_datetime(act['local_time'],
                                        format='ISO8601', utc=True)
    act = act.sort_values('local_time').reset_index(drop=True)

    chunks_mag = []
    n_loaded   = 0

    for chunk in pd.read_csv(
            PROC_DIR / "gyroscope_clean.csv",
            usecols=['local_time', 'x', 'y', 'z'],
            chunksize=500_000, low_memory=False):
        chunk['local_time'] = pd.to_datetime(chunk['local_time'],
                                              format='ISO8601', utc=True)
        chunk['magnitude'] = np.sqrt(chunk['x']**2 + chunk['y']**2 + chunk['z']**2)
        chunks_mag.append(chunk[['local_time', 'magnitude']])
        n_loaded += len(chunk)
        print(f"    Loaded {n_loaded:,} rows...", end='\r')

    print()
    df_gyro = pd.concat(chunks_mag, ignore_index=True).sort_values('local_time')

    # ── 1-minute RMS for correlation with accelerometer (Fig 13) ──────────────
    df_gyro_idx = df_gyro.set_index('local_time')
    gyro_rms_1min = (df_gyro_idx['magnitude']
                     .resample('1min')
                     .apply(_rms)
                     .dropna())

    # ── 10-second RMS with activity labels for boxplot (Fig 14) ───────────────
    gyro_rms_10s = (df_gyro_idx['magnitude']
                    .resample('10s')
                    .apply(_rms)
                    .dropna()
                    .reset_index())
    gyro_rms_10s.columns = ['local_time', 'rms_rads']

    # Assign activity label to each 10-sec bin via backward merge
    # (each bin gets the most recent activity event before it)
    gyro_rms_10s = pd.merge_asof(
        gyro_rms_10s.sort_values('local_time'),
        act[['local_time', 'activity']].sort_values('local_time'),
        on='local_time',
        direction='backward'
    ).dropna(subset=['activity'])

    print(f"  Gyro 1-min bins : {len(gyro_rms_1min)}")
    print(f"  Gyro 10-sec bins: {len(gyro_rms_10s)}")
    print("  Activity counts in 10-sec bins:")
    print(gyro_rms_10s['activity'].value_counts().to_string())

    return gyro_rms_1min, gyro_rms_10s


def plot_gyro_vs_accel(gyro_rms_1min):
    """
    Fig 13: Overlay gyroscope RMS (rad/s) and accelerometer RMS (g) on dual axes
    across all 7 days at 1-minute resolution. Computes Pearson correlation
    to quantify collinearity — this figure justifies excluding gyroscope from
    the multivariate regression (Script 05) on the grounds that the two signals
    carry redundant activity information.
    """
    print("\n[ Fig 13 ] Gyroscope vs Accelerometer RMS")

    accel_rms = pd.read_csv(PROC_DIR / "accel_rms_1min.csv")
    accel_rms['local_time'] = pd.to_datetime(accel_rms['local_time'],
                                              format='ISO8601', utc=True)
    accel_rms = accel_rms.set_index('local_time').sort_index()

    # Align on common timestamps (inner join)
    combined = accel_rms.join(gyro_rms_1min.rename('gyro_rms'),
                               how='inner').dropna()

    r, p = stats.pearsonr(combined['rms_g'], combined['gyro_rms'])
    print(f"  Pearson r (accel vs gyro RMS, 1-min) = {r:.4f}  p = {p:.2e}")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Accelerometer on left axis
    ax1.plot(combined.index, combined['rms_g'],
             color=C_GREEN, lw=0.6, alpha=0.75,
             label=f'Accelerometer RMS (g)')
    ax1.set_ylabel('Accelerometer RMS (g)', color=C_GREEN)
    ax1.tick_params(axis='y', colors=C_GREEN)

    # Gyroscope on right axis
    ax2 = ax1.twinx()
    ax2.plot(combined.index, combined['gyro_rms'],
             color=C_PURPLE, lw=0.6, alpha=0.65,
             label=f'Gyroscope RMS (rad/s)')
    ax2.set_ylabel('Gyroscope RMS (rad/s)', color=C_PURPLE)
    ax2.tick_params(axis='y', colors=C_PURPLE)
    ax2.spines['top'].set_visible(False)

    # Day boundary lines
    for d in pd.date_range('2026-03-13', '2026-03-20', freq='D', tz='UTC'):
        ax1.axvline(d, color=C_GREY, lw=0.6, linestyle=':', alpha=0.5)

    ax1.set_xlabel('Date (UTC)')
    ax1.set_title(
        f'Gyroscope RMS vs Accelerometer RMS — 7-Day, 1-Minute Windows\n'
        f'Pearson r = {r:.3f}  (p = {p:.1e}) — high correlation confirms '
        f'redundancy; gyroscope excluded from regression to avoid multicollinearity',
        pad=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.set_xlim(combined.index.min(), combined.index.max())

    # Shared legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    plt.tight_layout()
    _save(fig, 'fig13_gyro_vs_accel_rms.png')

    return r


def plot_gyro_by_activity(gyro_rms_10s):
    """
    Fig 14: Boxplot of gyroscope RMS (rad/s) grouped by iOS CoreMotion activity
    class (stationary, walking, automotive, running). This shows that the
    gyroscope captures distinct rotational dynamics for each activity — a
    qualitative insight the accelerometer magnitude cannot provide alone.
    The figure completes the gyroscope's scientific role in the project.
    """
    print("\n[ Fig 14 ] Gyroscope RMS by Activity Type")

    # Order by expected rotation level
    act_order = ['stationary', 'walking', 'automotive', 'running']
    act_order = [a for a in act_order
                 if a in gyro_rms_10s['activity'].unique()]

    groups = [gyro_rms_10s[gyro_rms_10s['activity'] == a]['rms_rads'].values
              for a in act_order]
    ns     = [len(g) for g in groups]

    print("  Samples per activity (10-sec bins):")
    for a, n in zip(act_order, ns):
        print(f"    {a:<14}: {n:>5} bins")

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        groups,
        labels=[f'{a.capitalize()}\n(n={n})' for a, n in zip(act_order, ns)],
        patch_artist=True,
        notch=False,
        medianprops=dict(color='white', linewidth=2.0),
        flierprops=dict(marker='o', markersize=2, alpha=0.25,
                        markeredgewidth=0.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, act in zip(bp['boxes'], act_order):
        patch.set_facecolor(ACT_COLORS.get(act, C_GREY))
        patch.set_alpha(0.75)

    # Annotate median values
    for i, group in enumerate(groups):
        median = np.median(group)
        ax.text(i + 1, median + 0.01, f'{median:.3f}',
                ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color='#1E293B')

    ax.set_xlabel('Activity Type (iOS CoreMotion Classification)')
    ax.set_ylabel('Gyroscope Magnitude RMS (rad/s)')
    ax.set_title(
        'Gyroscope RMS by Activity Type — 10-Second Windows\n'
        'Distinct rotational dynamics confirm gyroscope provides '
        'complementary (not redundant) information per activity class',
        pad=10)

    # Add note about running sample size
    if 'running' in act_order:
        run_idx = act_order.index('running')
        ax.text(run_idx + 1, ax.get_ylim()[1] * 0.92,
                '⚠ small n', ha='center', fontsize=8, color=C_RED, style='italic')

    plt.tight_layout()
    _save(fig, 'fig14_gyro_by_activity.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 15 — Activity Time-of-Day Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_activity_heatmap():
    """
    Shows what fraction of each hour of the day (0–23 UTC = 0–23 local,
    since UK did not shift to BST until 29 March 2026) was spent in each
    activity type, aggregated across all 7 recording days.

    Method: activity events are forward-filled to a 1-minute grid
    (each minute assigned the most recent activity label). Minutes are
    then grouped by (hour, activity) and normalised to percentages.

    This reveals the circadian structure of behaviour: when activity
    peaks occur, when the user is typically stationary, and whether
    automotive/walking bouts cluster at particular times of day.
    """
    print("\n[ Fig 15 ] Activity Time-of-Day Heatmap")

    act = pd.read_csv(PROC_DIR / "activity_clean.csv",
                      usecols=['local_time', 'activity'])
    act['local_time'] = pd.to_datetime(act['local_time'],
                                        format='ISO8601', utc=True)
    act = act.sort_values('local_time').reset_index(drop=True)

    # Create 1-minute grid covering the recording period
    t_min  = act['local_time'].min().floor('min')
    t_max  = act['local_time'].max().ceil('min')
    grid   = pd.DataFrame({'local_time': pd.date_range(t_min, t_max,
                                                         freq='1min', tz='UTC')})

    # Assign each minute the most recent activity label
    grid = pd.merge_asof(grid, act[['local_time', 'activity']],
                          on='local_time', direction='backward').dropna()

    grid['hour'] = grid['local_time'].dt.hour

    # Count minutes per (hour, activity), normalise per hour → %
    pivot = (grid.groupby(['hour', 'activity'])
                 .size()
                 .unstack(fill_value=0))

    # Reorder columns by activity order
    col_order = [c for c in ['stationary', 'walking', 'automotive', 'running']
                 if c in pivot.columns]
    pivot = pivot[col_order]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100)

    print(f"  1-min grid size: {len(grid):,} minutes")
    print(f"  Hours covered  : {grid['hour'].nunique()}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [2, 1.2]})

    # Left: heatmap (hours × activity type)
    ax = axes[0]
    im = ax.imshow(pivot_pct.T.values, aspect='auto', cmap='YlGn',
                   vmin=0, vmax=100, origin='upper')

    ax.set_xticks(range(len(pivot_pct.index)))
    ax.set_xticklabels([f'{h:02d}:00' for h in pivot_pct.index],
                        rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(col_order)))
    ax.set_yticklabels([c.capitalize() for c in col_order], fontsize=10)

    # Annotate cells with % value
    for j, act_type in enumerate(col_order):
        for i, hour in enumerate(pivot_pct.index):
            val = pivot_pct.loc[hour, act_type]
            if val > 2:
                color = 'white' if val > 60 else '#1E293B'
                ax.text(i, j, f'{val:.0f}%', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('% of minutes in activity', fontsize=9)
    ax.set_xlabel('Hour of Day (UTC = London local time during recording)')
    ax.set_title('Activity Fraction by Hour of Day\n'
                 '(aggregated across all 7 recording days)', pad=10)

    # Right: overall activity distribution (pie chart)
    ax2 = axes[1]
    total_mins = pivot.sum()
    total_mins = total_mins[total_mins > 0]
    colors_pie  = [ACT_COLORS.get(a, C_GREY) for a in total_mins.index]
    wedges, texts, autotexts = ax2.pie(
        total_mins.values,
        labels=[f'{a.capitalize()}\n{total_mins[a]:,.0f} min'
                for a in total_mins.index],
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax2.set_title('Overall Activity Distribution\n(7-day total minutes)', pad=10)

    plt.suptitle('Activity Classification — Time-of-Day Pattern & Overall Distribution\n'
                 '(iOS CoreMotion via Sensor Logger  |  1-Minute Resolution)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, 'fig15_activity_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  MobiSense — Insights & Visualisations (Script 06)")
    print("=" * 65)

    plot_7day_overview()
    plot_daily_steps_summary()

    # Gyroscope analyses (both share the same chunked computation)
    gyro_rms_1min, gyro_rms_10s = compute_gyro_rms()
    r_gyro = plot_gyro_vs_accel(gyro_rms_1min)
    plot_gyro_by_activity(gyro_rms_10s)

    plot_activity_heatmap()

    print("\n" + "=" * 65)
    print("  Insights complete.")
    print(f"  Figures saved to: {FIG_DIR}")
    print()
    for f in sorted(FIG_DIR.glob("fig1[1-5]*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name}  ({size_kb} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
