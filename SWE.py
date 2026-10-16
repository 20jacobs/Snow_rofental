
# ============================================================
# Snow Water Equivalent (SWE) analysis for two weather stations
#
# - Reads all available data from both stations
# - Uses paths relative to this script for portability
# - Parses timestamps with or without fractional seconds
# - Converts SWE measurements to numeric values
# - Excludes negative SWE measurements
# - Handles duplicate timestamps transparently
# - Calculates descriptive statistics for each station
# - Identifies missing measurements and large timestamp gaps
# - Interrupts plotted lines across gaps exceeding 20 minutes
# - Plots SWE time series for Bella Vista and Proviantdepot
# - Saves the figure as a PDF in the repository directory
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ============================================================
# 1. SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files and output figure are stored in this directory
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR

TIME_COL = "Date and time"

# Expected measurement interval: 10 minutes.
# Interrupt plotted lines when the gap exceeds 20 minutes.
MAX_GAP = pd.Timedelta(minutes=20)

STATIONS = {
    "Bella Vista": {
        "file": "1_BellaVista_10min_2015-2025.csv",
        "column": "SSG-2 snow water equivalent II (mm)",
        "color": "steelblue",
    },
    "Proviantdepot": {
        "file": "3_Proviantdepot_10min_2019-2025.csv",
        "column": "SSG-2 snow water equivalent (mm)",
        "color": "forestgreen",
    },
}


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

data = {}
statistics = {}

for station, settings in STATIONS.items():

    file_path = DATA_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Ensure that the CSV file is located in: {DATA_DIR}"
        )

    # Automatically detect the CSV delimiter
    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
    )

    # Remove leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()

    required = [
        TIME_COL,
        settings["column"],
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{station}: Missing columns {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required].copy()

    # --------------------------------------------------------
    # 2.1 Parse timestamps
    # --------------------------------------------------------

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    invalid_timestamps = int(
        df[TIME_COL].isna().sum()
    )

    # Remove rows without valid timestamps
    df = df.dropna(subset=[TIME_COL]).copy()

    # --------------------------------------------------------
    # 2.2 Convert SWE measurements to numeric values
    # --------------------------------------------------------

    swe_col = settings["column"]

    df[swe_col] = pd.to_numeric(
        df[swe_col],
        errors="coerce",
    )

    # Count negative measurements before excluding them
    negative_count = int(
        (df[swe_col] < 0).sum()
    )

    # Treat negative SWE values as invalid measurements
    # Retain their timestamps so missing values remain visible
    df.loc[
        df[swe_col] < 0,
        swe_col,
    ] = np.nan

    # --------------------------------------------------------
    # 2.3 Handle duplicate records and timestamps
    # --------------------------------------------------------

    # Remove completely identical rows
    exact_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # Identify timestamps that still have multiple records
    duplicate_mask = df[TIME_COL].duplicated(keep=False)

    duplicate_timestamp_rows = int(duplicate_mask.sum())

    duplicate_timestamp_count = int(
        df.loc[duplicate_mask, TIME_COL].nunique()
    )

    if duplicate_timestamp_count > 0:

        print(
            f"Warning: {station} has "
            f"{duplicate_timestamp_count} timestamps with "
            "multiple records after exact duplicates were removed. "
            "Valid SWE values at each duplicated timestamp "
            "will be averaged."
        )

        # Aggregate duplicate timestamps to a single observation.
        # Only valid SWE values contribute to the mean.
        df = (
            df.groupby(TIME_COL, as_index=False)
            .agg({swe_col: "mean"})
        )

    # Sort chronologically and use timestamps as index
    df = (
        df.sort_values(TIME_COL)
        .set_index(TIME_COL)
    )

    swe = df[swe_col].copy()

    # --------------------------------------------------------
    # 2.4 Identify gaps in the time series
    # --------------------------------------------------------

    time_gap = df.index.to_series().diff()

    gap_mask = time_gap > MAX_GAP

    # A gap is defined as the interval between two consecutive
    # timestamps exceeding the permitted maximum.
    large_gap_count = int(gap_mask.sum())

    # Insert NaN at the first observation after a large gap.
    # This prevents Matplotlib from connecting observations
    # across long periods without measurements.
    df["SWE_plot"] = swe.copy()

    df.loc[
        gap_mask.to_numpy(),
        "SWE_plot",
    ] = np.nan

    data[station] = df

    # --------------------------------------------------------
    # 2.5 Calculate descriptive statistics
    # --------------------------------------------------------

    valid_swe = swe.dropna()

    statistics[station] = {
        "max": valid_swe.max(),
        "mean": valid_swe.mean(),
        "valid": int(valid_swe.count()),
        "missing": int(swe.isna().sum()),
        "negative_removed": negative_count,
        "invalid_timestamps": invalid_timestamps,
        "exact_duplicates": exact_duplicates,
        "duplicate_timestamps": duplicate_timestamp_count,
        "large_gaps": large_gap_count,
    }

    # --------------------------------------------------------
    # 2.6 Print data-quality report
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print(station)
    print("=" * 55)

    print("Input file:", file_path.name)
    print("Period:", df.index.min(), "to", df.index.max())
    print("Records after timestamp cleaning:", len(df))
    print("Invalid timestamps removed:", invalid_timestamps)
    print("Exact duplicate rows removed:", exact_duplicates)
    print(
        "Timestamps with multiple records:",
        duplicate_timestamp_count,
    )
    print("Negative SWE measurements excluded:", negative_count)
    print("Valid SWE measurements:", int(valid_swe.count()))
    print("Missing or invalid SWE measurements:", int(swe.isna().sum()))
    print("Gaps exceeding 20 minutes:", large_gap_count)

    if valid_swe.notna().any():
        print(f"Maximum SWE: {valid_swe.max():.1f} mm")
        print(f"Mean SWE: {valid_swe.mean():.1f} mm")
    else:
        print("No valid SWE measurements available.")


# ============================================================
# 3. CREATE PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))

for station, df in data.items():

    settings = STATIONS[station]

    ax.plot(
        df.index,
        df["SWE_plot"],
        color=settings["color"],
        linewidth=0.7,
        label=station,
    )

ax.set_title(
    "Snow Water Equivalent (SWE) at Two Weather Stations",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Snow Water Equivalent (mm)", fontsize=12)

ax.legend(
    loc="upper left",
    title="Weather station",
    frameon=True,
)

ax.grid(True, alpha=0.3)


# ============================================================
# 4. DESCRIPTIVE STATISTICS BOX
# ============================================================

stats_lines = [
    "DESCRIPTIVE STATISTICS",
    "",
]

for station in STATIONS:

    values = statistics[station]

    mean_text = (
        f"{values['mean']:.1f} mm"
        if pd.notna(values["mean"])
        else "N/A"
    )

    max_text = (
        f"{values['max']:.1f} mm"
        if pd.notna(values["max"])
        else "N/A"
    )

    stats_lines.extend([
        station,
        f"Valid measurements: {values['valid']:,}",
        f"Mean:    {mean_text}",
        f"Maximum: {max_text}",
        f"Missing/invalid: {values['missing']:,}",
        f"Negative excluded: {values['negative_removed']:,}",
        f"Invalid timestamps: {values['invalid_timestamps']:,}",
        f"Duplicate timestamps: {values['duplicate_timestamps']:,}",
        f"Gaps > 20 min: {values['large_gaps']:,}",
        "",
    ])

stats_text = "\n".join(stats_lines).strip()

ax.text(
    0.985,
    0.97,
    stats_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    family="monospace",
    linespacing=1.35,
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="grey",
        linewidth=0.8,
        alpha=0.95,
    ),
    zorder=6,
)


# ============================================================
# 5. X-AXIS: JUNE AND DECEMBER OF EVERY YEAR
# ============================================================

ax.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=[6, 12])
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=0,
    ha="center",
)


# ============================================================
# 6. SAVE AND DISPLAY
# ============================================================

fig.tight_layout()

output_file = (
    OUTPUT_DIR / "swe_two_stations_adjusted_timestamps.pdf"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nAnalysis complete.")
print("Plot saved to:", output_file)
print("Output directory:", OUTPUT_DIR)
