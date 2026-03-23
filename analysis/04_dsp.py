"""
04_dsp.py
MobiSense — Step 4: Digital Signal Processing Analysis

DSP Techniques:
  1. Butterworth Low-Pass Filter    — Barometer: remove sensor noise, extract atmospheric trend
  2. Butterworth Band-Pass Filter   — Accelerometer: isolate human-movement frequency band (0.5–4 Hz)
  3. Welch Power Spectral Density   — Accelerometer: identify dominant step frequency via FFT
  4. Short-Time Fourier Transform   — Accelerometer: visualise how frequency content evolves over time
  5. RMS Energy (sliding window)    — Accelerometer: quantify activity intensity over 7 days

Supporting analysis (not DSP, but feeds downstream scripts):
  6. Moving Average                 — Battery: reveal daily charge/discharge cycle pattern
  7. Resampling                     — Barometer + Accel RMS → 1-min and hourly, for Script 05

Figures saved to /figures/:
  fig01_barometer_filter.png       — 7-day overview + zoom showing LPF noise removal
  fig02_accel_rms.png              — 7-day RMS energy time series (1-min windows)
  fig03a_accel_bandpass.png        — Raw vs band-passed accelerometer magnitude (15-min window)
  fig03b_accel_psd.png             — Welch PSD with annotated step frequency peak
  fig04_accel_spectrogram.png      — STFT spectrogram over Day 6 (most active day)
  fig05_battery_smoothed.png       — Battery % with 60-min moving average + charging periods

Data outputs for Script 05 (cross-correlation):
  processed/baro_1min.csv          — barometer filtered, resampled to 1-minute means
  processed/baro_hourly.csv        — barometer filtered, resampled to hourly means
  processed/accel_rms_1min.csv     — accelerometer RMS, 1-minute bins
  processed/accel_rms_hourly.csv   — accelerometer RMS, hourly bins

Run: python3 04_dsp.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — safe for saving without a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from scipy import signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR = Path("/Users/vedant/Downloads/IOT PROJECT/processed")
FIG_DIR  = Path("/Users/vedant/Downloads/IOT PROJECT/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       150,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.6,
    'lines.linewidth':   1.2,
    'legend.framealpha': 0.85,
    'legend.fontsize':   10,
})

# Colour palette (accessible, consistent across all figures)
C_BLUE   = '#2563EB'
C_RED    = '#DC2626'
C_GREEN  = '#16A34A'
C_ORANGE = '#D97706'
C_GREY   = '#6B7280'


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BAROMETER — Butterworth Low-Pass Filter
# ══════════════════════════════════════════════════════════════════════════════
def analyse_barometer():
    """
    Apply a 4th-order Butterworth low-pass filter to the barometer pressure
    signal to remove high-frequency sensor noise while preserving the slow
    atmospheric pressure trend.

    Nyquist-Shannon justification:
      The barometer samples at ~0.94 Hz → Nyquist = 0.47 Hz.
      Atmospheric pressure varies on timescales of minutes to hours.
      A cutoff of 0.01 Hz (period = 100 s ≈ 1.7 min) suppresses second-to-second
      jitter (breathing artefacts, phone vibration) while retaining all
      meteorologically meaningful variation.
    """
    print("\n[ 1 ] Barometer — Butterworth Low-Pass Filter")

    df = pd.read_csv(PROC_DIR / "barometer_clean.csv", low_memory=False)
    df['local_time'] = pd.to_datetime(df['local_time'], format='ISO8601', utc=True)
    df = df.sort_values('local_time').reset_index(drop=True)

    # Compute actual sample rate from the median inter-sample interval
    dt_s  = df['local_time'].diff().dt.total_seconds().median()
    fs    = 1.0 / dt_s          # ~0.94 Hz
    Nyq   = fs / 2.0            # ~0.47 Hz

    print(f"  Sample rate  : {fs:.4f} Hz  (median Δt = {dt_s:.4f} s)")
    print(f"  Nyquist freq : {Nyq:.4f} Hz")

    # ── Design 4th-order Butterworth LPF ──────────────────────────────────────
    cutoff_hz = 0.01            # 1 cycle per 100 seconds
    Wn        = cutoff_hz / Nyq
    b, a      = signal.butter(4, Wn, btype='low', analog=False)

    print(f"  LPF cutoff   : {cutoff_hz} Hz  (period = {1/cutoff_hz:.0f} s = {1/cutoff_hz/60:.1f} min)")
    print(f"  Filter order : 4  (effective 8th-order zero-phase via filtfilt)")

    # Apply zero-phase forward-backward filter (no phase distortion)
    pressure_raw      = df['pressure'].values
    pressure_filtered = signal.filtfilt(b, a, pressure_raw)
    df['pressure_filtered'] = pressure_filtered

    noise_rms = np.sqrt(np.mean((pressure_raw - pressure_filtered) ** 2))
    print(f"  Raw range    : {pressure_raw.min():.2f} – {pressure_raw.max():.2f} hPa")
    print(f"  Filtered range: {pressure_filtered.min():.2f} – {pressure_filtered.max():.2f} hPa")
    print(f"  Noise RMS    : {noise_rms:.4f} hPa  (removed by filter)")

    # ── Figure 1: Two-panel — 7-day overview + Day 1 zoom ─────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Top panel: full 7-day time series
    ax = axes[0]
    ax.plot(df['local_time'], pressure_raw,
            color=C_BLUE, lw=0.3, alpha=0.45, label='Raw (~0.94 Hz)')
    ax.plot(df['local_time'], pressure_filtered,
            color=C_RED, lw=1.6, label='Filtered (Butterworth LPF, 0.01 Hz)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Barometer: Butterworth Low-Pass Filter — 7-Day Overview\n'
                 'Rising high-pressure system visible in filtered trend (998 → 1027 hPa)')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlim(df['local_time'].min(), df['local_time'].max())

    # Bottom panel: Day 1 zoom — noise removal effect clearly visible
    day1_date = df['local_time'].dt.date.min()
    df_d1     = df[df['local_time'].dt.date == day1_date].copy()

    ax2 = axes[1]
    ax2.plot(df_d1['local_time'], df_d1['pressure'],
             color=C_BLUE, lw=0.5, alpha=0.55, label='Raw (every ~1 s)')
    ax2.plot(df_d1['local_time'], df_d1['pressure_filtered'],
             color=C_RED, lw=2.0, label='Filtered (LPF 0.01 Hz)')
    ax2.set_xlabel('Time (UTC)')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title(f'Zoom: Day 1 ({day1_date}) — High-Frequency Noise Removal Clearly Visible')
    ax2.legend(loc='upper left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.set_xlim(df_d1['local_time'].min(), df_d1['local_time'].max())

    plt.tight_layout(h_pad=3.0)
    _save(fig, "fig01_barometer_filter.png")

    # ── Resample filtered signal for Script 05 ────────────────────────────────
    df_idx      = df.set_index('local_time').sort_index()
    baro_1min   = df_idx['pressure_filtered'].resample('1min').mean().dropna()
    baro_hourly = df_idx['pressure_filtered'].resample('1h').mean().dropna()

    baro_1min.name   = 'pressure_hpa'
    baro_hourly.name = 'pressure_hpa'

    baro_1min.to_csv(PROC_DIR  / "baro_1min.csv",   header=True)
    baro_hourly.to_csv(PROC_DIR / "baro_hourly.csv", header=True)

    print(f"  Saved: baro_1min.csv    ({len(baro_1min)} rows)")
    print(f"  Saved: baro_hourly.csv  ({len(baro_hourly)} rows)")

    return fs


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ACCELEROMETER — RMS Energy (chunked, all 7 days)
# ══════════════════════════════════════════════════════════════════════════════
def compute_accel_rms():
    """
    Compute the Root Mean Square (RMS) of the 3-axis accelerometer magnitude
    in rolling 1-minute windows across all 7 days.

    Magnitude:  |a| = √(x² + y² + z²)  — combines all three axes into a single
    activity-intensity signal, removing dependence on phone orientation.

    RMS in a window W:  √( (1/N) Σ aᵢ² )  — gives the effective signal power,
    with higher values indicating more vigorous movement.

    Processed in 500,000-row chunks to handle the 1.2 GB file without
    exhausting system RAM.
    """
    print("\n[ 2 ] Accelerometer — RMS Energy (1-minute windows, all 7 days)")

    ACCEL_FS = 10.0     # Hz — from Metadata.csv (100 ms sample interval)

    chunks = []
    n_loaded = 0
    for chunk in pd.read_csv(
            PROC_DIR / "accelerometer_clean.csv",
            usecols=['local_time', 'x', 'y', 'z'],
            chunksize=500_000, low_memory=False):
        chunk['local_time'] = pd.to_datetime(chunk['local_time'],
                                              format='ISO8601', utc=True)
        chunk['magnitude'] = np.sqrt(chunk['x']**2 + chunk['y']**2 + chunk['z']**2)
        chunks.append(chunk[['local_time', 'magnitude']])
        n_loaded += len(chunk)
        print(f"    Loaded {n_loaded:,} rows...", end='\r')

    print()
    df_mag = pd.concat(chunks, ignore_index=True)
    df_mag = df_mag.set_index('local_time').sort_index()

    # 1-minute RMS (for figures and detailed analysis)
    rms_1min = (df_mag['magnitude']
                .resample('1min')
                .apply(lambda x: float(np.sqrt(np.mean(x.values**2)))
                       if len(x) > 0 else np.nan)
                .dropna())

    # 1-hour RMS (for cross-correlation with hourly weather in Script 05)
    rms_hourly = (df_mag['magnitude']
                  .resample('1h')
                  .apply(lambda x: float(np.sqrt(np.mean(x.values**2)))
                         if len(x) > 0 else np.nan)
                  .dropna())

    print(f"  Total samples     : {len(df_mag):,}")
    print(f"  1-min RMS bins    : {len(rms_1min)}")
    print(f"  1-hour RMS bins   : {len(rms_hourly)}")
    print(f"  Mean RMS (1-min)  : {rms_1min.mean():.4f} g")
    print(f"  Peak RMS (1-min)  : {rms_1min.max():.4f} g  ← most active minute")

    # Save for Script 05
    rms_1min.name   = 'rms_g'
    rms_hourly.name = 'rms_g'
    rms_1min.to_csv(PROC_DIR   / "accel_rms_1min.csv",   header=True)
    rms_hourly.to_csv(PROC_DIR / "accel_rms_hourly.csv", header=True)
    print(f"  Saved: accel_rms_1min.csv, accel_rms_hourly.csv")

    # ── Figure 2: 7-day RMS energy time series ────────────────────────────────
    # Also compute per-day mean for annotation
    rms_daily = rms_1min.resample('1D').mean()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(rms_1min.index, rms_1min.values,
                    alpha=0.35, color=C_GREEN)
    ax.plot(rms_1min.index, rms_1min.values,
            color=C_GREEN, lw=0.5, alpha=0.7, label='RMS (1-min window)')

    # Overlay daily mean as step function for comparison
    ax.step(rms_daily.index, rms_daily.values, where='post',
            color=C_RED, lw=1.8, linestyle='--', label='Daily mean RMS', alpha=0.85)

    # Annotate day numbers
    for i, (ts, val) in enumerate(zip(rms_daily.index, rms_daily.values)):
        ax.text(ts + pd.Timedelta(hours=12), val + 0.005,
                f'Day {i+1}', fontsize=8, color=C_RED, ha='center')

    ax.set_xlabel('Date (UTC)')
    ax.set_ylabel('RMS Acceleration (g)')
    ax.set_title('Accelerometer RMS Energy — 1-Minute Windows Over 7 Days\n'
                 '(higher = more vigorous movement; dashed = daily mean)')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlim(rms_1min.index.min(), rms_1min.index.max())

    plt.tight_layout()
    _save(fig, "fig02_accel_rms.png")

    return rms_1min, rms_hourly


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ACCELEROMETER — Butterworth Band-Pass Filter + Welch PSD
# ══════════════════════════════════════════════════════════════════════════════
def analyse_step_frequency():
    """
    (a) Band-Pass Filter: isolate the human locomotion frequency band (0.5–4 Hz)
        from the raw accelerometer signal.  This removes:
          - DC offset and slow body sway (< 0.5 Hz)
          - High-frequency vibration and quantisation noise (> 4 Hz)

    (b) Welch Power Spectral Density: apply FFT-based spectral estimation to the
        band-passed signal to find the dominant step frequency.
        Welch's method averages multiple overlapping FFTs (Hann-windowed segments
        of 2048 samples) to reduce spectral variance while maintaining
        frequency resolution Δf = fs / nperseg = 10 / 2048 ≈ 0.005 Hz.

    Analysis is performed on Day 6 (2026-03-18), the most active day with
    11,297 steps, to maximise the signal-to-noise ratio of the walking peaks.

    Nyquist-Shannon justification:
      The accelerometer samples at 10 Hz → Nyquist = 5 Hz.
      Human walking cadence (1.5–2.5 Hz) and running (up to ~3.5 Hz) are both
      well below Nyquist, confirming 10 Hz is a sufficient and appropriate
      sampling rate for step detection.
    """
    print("\n[ 3 ] Accelerometer — Band-Pass Filter + Welch PSD (Step Frequency)")

    ACCEL_FS = 10.0
    NYQUIST  = ACCEL_FS / 2.0

    # Load Day 6 in chunks to avoid pulling all 6M rows into memory
    print("  Loading Day 6 (most active: 11,297 steps)...")
    chunks = []
    for chunk in pd.read_csv(
            PROC_DIR / "accelerometer_clean.csv",
            usecols=['local_time', 'x', 'y', 'z', 'day'],
            chunksize=500_000, low_memory=False):
        sub = chunk[chunk['day'] == 6]
        if len(sub) > 0:
            chunks.append(sub)

    df_day = pd.concat(chunks, ignore_index=True)
    df_day['local_time'] = pd.to_datetime(df_day['local_time'],
                                           format='ISO8601', utc=True)
    df_day = df_day.sort_values('local_time').reset_index(drop=True)
    df_day['magnitude'] = np.sqrt(df_day['x']**2 + df_day['y']**2 + df_day['z']**2)
    magnitude = df_day['magnitude'].values

    print(f"  Day 6 samples     : {len(df_day):,}")
    print(f"  Sampling rate     : {ACCEL_FS} Hz | Nyquist : {NYQUIST} Hz")

    # ── (a) Design Butterworth Band-Pass Filter ────────────────────────────────
    # Passband: 0.5 Hz – 4.0 Hz  (human locomotion frequency band)
    # Walking cadence: 1.5–2.5 Hz;  Running cadence: up to ~3.5 Hz
    lowcut  = 0.5
    highcut = 4.0
    Wn_band = [lowcut / NYQUIST, highcut / NYQUIST]
    b_bp, a_bp = signal.butter(4, Wn_band, btype='band', analog=False)

    magnitude_bp = signal.filtfilt(b_bp, a_bp, magnitude)

    print(f"  Band-pass filter  : {lowcut}–{highcut} Hz  (human locomotion band)")
    print(f"  Filter order      : 4  (zero-phase via filtfilt)")

    # ── Figure 3a: Raw vs Band-Passed — 15-minute window ─────────────────────
    # Choose a 15-minute window from the middle of the day (most likely active)
    win_samples = int(15 * 60 * ACCEL_FS)  # 9,000 samples
    mid         = len(magnitude) // 2
    s_idx       = max(0, mid - win_samples // 2)
    e_idx       = s_idx + win_samples

    t_win  = df_day['local_time'].iloc[s_idx:e_idx]
    raw_win = magnitude[s_idx:e_idx]
    bp_win  = magnitude_bp[s_idx:e_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t_win, raw_win, color=C_BLUE, lw=0.5, alpha=0.7,
                 label='Raw magnitude  |a| = √(x²+y²+z²)')
    axes[0].set_ylabel('|a| (g)')
    axes[0].set_title('Accelerometer: Raw vs Butterworth Band-Pass Filtered Magnitude\n'
                      '15-Minute Active Window — Day 6 (Most Active Day)')
    axes[0].legend(loc='upper right')

    axes[1].plot(t_win, bp_win, color=C_RED, lw=0.6, alpha=0.85,
                 label='Band-passed (0.5–4 Hz)  — DC and vibration removed')
    axes[1].set_xlabel('Time (UTC)')
    axes[1].set_ylabel('|a| band-passed (g)')
    axes[1].legend(loc='upper right')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=3))

    plt.tight_layout(h_pad=2.5)
    _save(fig, "fig03a_accel_bandpass.png")

    # ── (b) Welch Power Spectral Density ──────────────────────────────────────
    # nperseg = 2048 samples → Δf = 10/2048 ≈ 0.0049 Hz (fine resolution)
    # noverlap = 1024 (50% overlap) → reduces variance without losing resolution
    # Hann window → reduces spectral leakage from non-periodic signal boundaries
    freqs, psd = signal.welch(
        magnitude_bp, fs=ACCEL_FS,
        nperseg=2048, noverlap=1024,
        window='hann', scaling='density'
    )

    # Find dominant peak within the locomotion band
    loco_mask  = (freqs >= 0.5) & (freqs <= 4.0)
    peak_freq  = freqs[loco_mask][np.argmax(psd[loco_mask])]
    peak_psd   = psd[loco_mask].max()
    step_cadence = peak_freq * 60  # steps per minute

    print(f"  Dominant step freq : {peak_freq:.3f} Hz  ({step_cadence:.1f} steps/min)")
    print(f"  Peak PSD           : {peak_psd:.4e} g²/Hz")

    # ── Figure 3b: PSD plot ────────────────────────────────────────────────────
    plot_mask = freqs <= 5.2   # show up to just above Nyquist
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.semilogy(freqs[plot_mask], psd[plot_mask],
                color=C_BLUE, lw=1.5, label='Welch PSD (band-passed magnitude)')

    # Highlight locomotion band
    ax.axvspan(lowcut, highcut, alpha=0.08, color=C_GREEN,
               label=f'Locomotion band ({lowcut}–{highcut} Hz)')

    # Mark step frequency peak
    ax.axvline(peak_freq, color=C_RED, lw=2.0, linestyle='--',
               label=f'Step frequency: {peak_freq:.2f} Hz  ({step_cadence:.0f} steps/min)')
    ax.plot(peak_freq, peak_psd, 'o', color=C_RED, ms=8, zorder=5)

    # Annotate Nyquist limit
    ax.axvline(NYQUIST, color=C_GREY, lw=1.0, linestyle=':', alpha=0.7)
    ax.text(NYQUIST + 0.05, psd[plot_mask].max() * 0.6,
            f'Nyquist\n{NYQUIST:.0f} Hz', fontsize=9, color=C_GREY, va='top')

    # Walking cadence reference lines
    for hz, label in [(1.5, '1.5 Hz\n(slow walk)'), (2.0, '2.0 Hz\n(brisk walk)'),
                       (2.5, '2.5 Hz\n(fast walk)')]:
        ax.axvline(hz, color=C_ORANGE, lw=0.8, linestyle=':', alpha=0.5)
        ax.text(hz + 0.03, psd[plot_mask].min() * 3, label,
                fontsize=7.5, color=C_ORANGE, va='bottom')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (g²/Hz) — log scale')
    ax.set_title(f'Welch PSD — Accelerometer Magnitude (Band-Passed)\n'
                 f'Day 6 | Dominant Step Frequency: {peak_freq:.2f} Hz  '
                 f'= {step_cadence:.0f} steps/min\n'
                 f'(Nyquist = {NYQUIST} Hz confirms 10 Hz sampling is sufficient for step detection)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 5.2)

    plt.tight_layout()
    _save(fig, "fig03b_accel_psd.png")

    return df_day, magnitude_bp, ACCEL_FS, peak_freq


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ACCELEROMETER — STFT / Spectrogram
# ══════════════════════════════════════════════════════════════════════════════
def analyse_spectrogram(df_day, magnitude_bp, ACCEL_FS):
    """
    Short-Time Fourier Transform (STFT): divide the signal into overlapping
    windows, compute FFT on each, and stack the results into a time-frequency
    map (spectrogram).

    Unlike a single FFT (which gives one spectrum averaged over the entire day),
    the STFT reveals *when* different frequencies are active — walking bouts
    appear as bright horizontal bands at 1.5–2.5 Hz, while stationary periods
    are dark across all frequencies.

    Parameters:
      nperseg  = 512 samples → time resolution = 512/10 = 51.2 s per window
      noverlap = 384 samples → hop = 128 → time step = 12.8 s between bins
      window   = Hann → suppresses spectral leakage between frequency bins
      Frequency resolution = ACCEL_FS / nperseg = 10/512 ≈ 0.020 Hz

    Applied to the band-passed Day 6 signal (868,592 samples).
    """
    print("\n[ 4 ] Accelerometer — STFT Spectrogram (Day 6)")

    nperseg  = 512
    noverlap = 384
    hop      = nperseg - noverlap   # 128 samples = 12.8 s per time bin

    f_stft, t_sec, Sxx = signal.spectrogram(
        magnitude_bp, fs=ACCEL_FS,
        nperseg=nperseg, noverlap=noverlap,
        window='hann', scaling='density'
    )

    # Convert relative time in seconds → absolute UTC timestamps
    t_start = df_day['local_time'].iloc[0]
    t_abs   = pd.DatetimeIndex([t_start + pd.Timedelta(seconds=float(s))
                                 for s in t_sec])

    # Focus plot on 0–5 Hz (above Nyquist = 5 Hz contains no information)
    f_mask  = f_stft <= 5.0
    f_plot  = f_stft[f_mask]
    S_plot  = Sxx[f_mask, :]

    # Avoid log(0) by clamping minimum
    S_min = max(S_plot[S_plot > 0].min(), 1e-9)
    S_max = S_plot.max()

    print(f"  STFT shape       : {S_plot.shape}  (freq bins × time bins)")
    print(f"  Time resolution  : {hop / ACCEL_FS:.1f} s per bin  "
          f"({len(t_sec)} bins over Day 6)")
    print(f"  Freq resolution  : {ACCEL_FS / nperseg:.4f} Hz per bin")
    print(f"  PSD range        : {S_min:.2e} – {S_max:.2e} g²/Hz")

    # ── Figure 4: Spectrogram ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    pcm = ax.pcolormesh(
        t_abs, f_plot, S_plot,
        norm=LogNorm(vmin=S_min, vmax=S_max),
        cmap='inferno', shading='auto'
    )

    # Walking cadence band annotation
    ax.axhline(1.5, color='cyan', lw=1.2, linestyle='--', alpha=0.9)
    ax.axhline(2.5, color='cyan', lw=1.2, linestyle='--', alpha=0.9,
               label='Walking cadence band (1.5–2.5 Hz)')
    ax.fill_betweenx([1.5, 2.5],
                     t_abs[0], t_abs[-1],
                     color='cyan', alpha=0.06)
    ax.text(t_abs[-1], 2.0, '  Walking\n  band', fontsize=8,
            color='cyan', va='center')

    # Nyquist line
    ax.axhline(5.0, color=C_GREY, lw=0.8, linestyle=':', alpha=0.5)
    ax.text(t_abs[0], 4.85, 'Nyquist limit (5 Hz)',
            fontsize=8, color=C_GREY, va='top')

    cbar = plt.colorbar(pcm, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label('PSD (g²/Hz, log scale)', fontsize=10)

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('STFT Spectrogram — Accelerometer Magnitude (Band-Passed 0.5–4 Hz)\n'
                 'Day 6 (Most Active: 11,297 steps) | Bright bands = sustained periodic movement')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_ylim(0, 5.2)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    _save(fig, "fig04_accel_spectrogram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BATTERY — Moving Average  (visualisation, not DSP)
# ══════════════════════════════════════════════════════════════════════════════
def analyse_battery():
    """
    Apply a 60-minute rolling mean to the battery percentage signal to smooth
    second-to-second fluctuations and clearly reveal the daily charge/discharge
    cycle.  Charging periods are highlighted as green shading.

    This is a visualisation technique (not a DSP analysis technique), included
    because it produces an interpretable behavioural signal: drain rate during
    active daytime hours vs overnight charging rhythm.
    """
    print("\n[ 5 ] Battery — Moving Average (Daily Charge/Discharge Pattern)")

    df = pd.read_csv(PROC_DIR / "battery_clean.csv",
                     usecols=['local_time', 'battery_pct', 'batteryState'],
                     low_memory=False)
    df['local_time'] = pd.to_datetime(df['local_time'], format='ISO8601', utc=True)
    df = df.set_index('local_time').sort_index()

    # Resample to 1-minute means (reduces 613k → ~10k points, plottable)
    batt_1min    = df['battery_pct'].resample('1min').mean().dropna()
    charging_1min = (df['batteryState'] == 'charging').resample('1min').max()

    # 60-minute centred rolling mean
    batt_smooth = batt_1min.rolling(window=60, center=True, min_periods=10).mean()

    print(f"  1-min samples : {len(batt_1min)}")
    print(f"  Min battery   : {batt_1min.min():.1f}%")
    print(f"  Max battery   : {batt_1min.max():.1f}%")
    print(f"  Charging mins : {charging_1min.sum()} min  "
          f"({charging_1min.sum()/len(charging_1min)*100:.1f}% of recording)")

    # Find contiguous charging blocks for shading
    is_charging = charging_1min.reindex(batt_1min.index, fill_value=False)
    edges       = is_charging.astype(int).diff().fillna(0)
    starts      = batt_1min.index[edges == 1]
    ends        = batt_1min.index[edges == -1]

    # Handle edge case: recording starts/ends mid-charge
    if is_charging.iloc[0]:
        starts = starts.insert(0, batt_1min.index[0])
    if is_charging.iloc[-1]:
        ends   = ends.append(pd.DatetimeIndex([batt_1min.index[-1]]))

    # ── Figure 5: Battery over 7 days ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Raw 1-minute data (faint)
    ax.plot(batt_1min.index, batt_1min.values,
            color=C_BLUE, lw=0.35, alpha=0.30, label='Raw (1-min mean)')

    # 60-min moving average (solid)
    ax.plot(batt_smooth.index, batt_smooth.values,
            color=C_BLUE, lw=2.2, label='60-min moving average')

    # Charging shading
    first_charge = True
    for start, end in zip(starts, ends):
        ax.axvspan(start, end, alpha=0.22, color=C_GREEN,
                   label='Charging' if first_charge else '_nolegend_')
        first_charge = False

    # Annotate daily low points
    for day_i in range(1, 9):
        mask = batt_1min.index.date == (pd.Timestamp('2026-03-13') +
                                         pd.Timedelta(days=day_i-1)).date()
        if mask.any():
            day_min_val = batt_1min[mask].min()
            day_min_idx = batt_1min[mask].idxmin()
            if day_min_val < 85:   # only annotate significant dips
                ax.annotate(f'{day_min_val:.0f}%',
                            xy=(day_min_idx, day_min_val),
                            xytext=(0, -16), textcoords='offset points',
                            ha='center', fontsize=8.5, color=C_BLUE,
                            arrowprops=dict(arrowstyle='->', color=C_BLUE,
                                            lw=0.8))

    ax.set_xlabel('Date (UTC)')
    ax.set_ylabel('Battery Level (%)')
    ax.set_title('Battery Level Over 7 Days — Raw vs 60-Minute Moving Average\n'
                 '(green = charging periods; daily low points annotated)')
    ax.set_ylim(0, 108)
    ax.legend(loc='lower right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlim(batt_1min.index.min(), batt_1min.index.max())

    plt.tight_layout()
    _save(fig, "fig05_battery_smoothed.png")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: save figure and print confirmation
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, filename):
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  MobiSense — DSP Analysis (Script 04)")
    print("=" * 65)

    # 1. Barometer: LPF + resample
    analyse_barometer()

    # 2. Accelerometer RMS energy (all 7 days, chunked)
    compute_accel_rms()

    # 3. Band-pass + Welch PSD (Day 6)
    df_day, magnitude_bp, accel_fs, peak_freq = analyse_step_frequency()

    # 4. STFT spectrogram (Day 6, uses band-passed signal from step 3)
    analyse_spectrogram(df_day, magnitude_bp, accel_fs)

    # 5. Battery moving average
    analyse_battery()

    print("\n" + "=" * 65)
    print("  DSP Analysis complete.")
    print(f"\n  Figures ({FIG_DIR.name}/):")
    for f in sorted(FIG_DIR.glob("fig0*.png")):
        print(f"    {f.name}")
    print(f"\n  Data for Script 05 ({PROC_DIR.name}/):")
    for f in ['baro_1min.csv', 'baro_hourly.csv',
               'accel_rms_1min.csv', 'accel_rms_hourly.csv']:
        path = PROC_DIR / f
        print(f"    {f}  ({path.stat().st_size // 1024} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
