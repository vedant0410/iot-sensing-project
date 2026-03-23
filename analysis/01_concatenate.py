"""
01_concatenate.py
MobiSense — Step 1: Concatenate all daily sensor recordings into one file per sensor.

Handles:
- Day 1 two-session split (main 24hr + remaining 3 hours)
- Days 2-7 single sessions each
- Sorts everything by timestamp
- Removes duplicate rows
- Reports gaps between sessions
- Saves one combined CSV per sensor to /processed/

Run: python3 01_concatenate.py
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/vedant/Downloads/IOT PROJECT/final data")
OUTPUT_DIR = Path("/Users/vedant/Downloads/IOT PROJECT/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sensors to process (skip Uncalibrated, Network, Annotation, Metadata) ───
SENSORS = ['Accelerometer', 'Gyroscope', 'Barometer', 'Pedometer', 'Activity', 'Location', 'Battery', 'Network']

# ── Discover all sessions across all day folders ─────────────────────────────
def get_sessions():
    sessions = []
    day_folders = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.lower().startswith('day')],
        key=lambda d: int(d.name.split()[-1])
    )
    for day_folder in day_folders:
        day_num = int(day_folder.name.split()[-1])
        session_dirs = sorted([d for d in day_folder.iterdir() if d.is_dir()])
        for session_dir in session_dirs:
            sessions.append({
                'day':          day_num,
                'session_name': session_dir.name,
                'path':         session_dir
            })
    return sessions

# ── Load one sensor CSV from one session ─────────────────────────────────────
def load_sensor(session_path, sensor_name, day_num, session_name):
    filepath = session_path / f"{sensor_name}.csv"
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty or len(df.columns) == 0:
            return None
    except Exception:
        return None
    df['local_time'] = pd.to_datetime(df['local_time'], utc=True)
    df['day']        = day_num
    df['session']    = session_name
    return df

# ── Concatenate all sessions for one sensor ───────────────────────────────────
def concatenate_sensor(sensor_name, sessions):
    dfs = []
    for s in sessions:
        df = load_sensor(s['path'], sensor_name, s['day'], s['session_name'])
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('local_time').reset_index(drop=True)
    combined = combined.drop_duplicates(subset=['time']).reset_index(drop=True)
    return combined

# ── Report gaps between sessions ──────────────────────────────────────────────
def report_gaps(df, sensor_name):
    time_diffs = df['local_time'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta('30min')]
    if len(large_gaps) > 0:
        print(f"  Gaps found ({len(large_gaps)}):")
        for idx, gap in large_gaps.items():
            print(f"    {gap} at {df.loc[idx, 'local_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        print(f"  No large gaps detected")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  MobiSense — Data Concatenation")
    print("=" * 65)

    sessions = get_sessions()
    print(f"\nFound {len(sessions)} recording sessions:\n")
    for s in sessions:
        print(f"  Day {s['day']} | {s['session_name']}")

    print(f"\nOutput: {OUTPUT_DIR}\n")
    print("=" * 65)

    summary = []

    for sensor in SENSORS:
        print(f"\n[ {sensor} ]")
        combined = concatenate_sensor(sensor, sessions)

        if combined is None:
            print(f"  No data found — skipping")
            continue

        report_gaps(combined, sensor)

        duration  = combined['local_time'].max() - combined['local_time'].min()
        days_span = duration.days + duration.seconds / 86400

        print(f"  Rows      : {len(combined):,}")
        print(f"  Start     : {combined['local_time'].min().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  End       : {combined['local_time'].max().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Duration  : {days_span:.2f} days")

        output_path = OUTPUT_DIR / f"{sensor.lower()}_combined.csv"
        combined.to_csv(output_path, index=False)
        print(f"  Saved     : {output_path.name}")

        summary.append({
            'Sensor':   sensor,
            'Rows':     f"{len(combined):,}",
            'Duration': f"{days_span:.2f} days",
            'File':     output_path.name
        })

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    for s in summary:
        print(f"  {s['Sensor']:<15} {s['Rows']:<12} {s['Duration']:<15} → {s['File']}")
    print("\nConcatenation complete. All files saved to /processed/")
    print("=" * 65)

if __name__ == "__main__":
    main()
