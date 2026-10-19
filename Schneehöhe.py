
# ============================================================
# SH-Analysis for all three stations
#
# - Performs a reproducible analysis of the available dataset
# - Reads CSV files from the repository directory
# - Creates an output folder for generated files
# - Saves figures and result tables in the output folder
# - Parses timestamps and validates measurement columns
# - Handles duplicate timestamps transparently
# - Excludes negative measurements
# - Calculates descriptive statistics for each station
# - Handles data gaps by interrupting plotted lines
# - Supports timestamps with and without milliseconds
# - Generates a combined time-series plot of snow depth
# - Saves the plot as a PDF file in the output folder
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ==================================================
# 1. SETTINGS
# ==================================================

sns.set_theme(style="whitegrid")

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files remain in the repository directory
DATA_DIR = BASE_DIR / "input"

# Create an output folder for all generated files
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"

# Expected measurement interval: 10 minutes.
# Lines are interrupted if the gap exceeds 20 minutes.
MAX_GAP = pd.Timedelta(minutes=20)

stations = {
      "Bella Vista (protected)": {
            "file": "1_BellaVista_10min_2015-2025.csv",
            "column": "SSG-2 snow water equivalent II (mm)",
            "color": "#0072B2",
        },
      
        "Bella Vista (exposed)": {
            "file": "1_BellaVista_10min_2015-2025.csv",
            "column": "Snow pillow snow water equivalent (mm)",
            "color": "#D55E00",
        },
        
    "Latschbloder": {
        "file": "2_Latschbloder_10min_2013-2025.csv",
        "column": "USH8 snow depth (mm)",
        "color": "darkorange",
    },
    "Proviantdepot": {
        "file": "3_Proviantdepot_10min_2019-2025.csv",
        "column": "USH9 snow depth (mm)",
        "color": "forestgreen",
    },
}


# ==================================================
# 2. LOAD AND CLEAN DATA
# ==================================================

data = {}
statistics = {}

for station, settings in stations.items():

    file_path = DATA_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    # Check whether required columns exist
    required = [TIME_COL, settings["column"]]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{station}: missing columns {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    snow_col = settings["column"]

    # --------------------------------------------------
    # 2.1 Convert timestamps consistently
    # --------------------------------------------------

    # Handles timestamps with and without milliseconds,
    # e.g. 2018-03-07 14:40:00
    # and 2018-03-07 14:40:00.000
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    # Convert snow-depth measurements to numeric values
    df[snow_col] = pd.to_numeric(
        df[snow_col],
        errors="coerce",
    )

    # Count invalid timestamps before removing them
    invalid_timestamps = int(df[TIME_COL].isna().sum())

    # Remove rows without valid timestamps
    df = df.dropna(subset=[TIME_COL])

    # Sort chronologically and set timestamps as index
    df = (
        df.sort_values(TIME_COL)
          .set_index(TIME_COL)
    )

    # --------------------------------------------------
    # 2.2 Remove negative snow-depth values
    # --------------------------------------------------

    negative_count = int((df[snow_col] < 0).sum())

    # Treat negative measurements as invalid.
    # Keep timestamps so gaps remain visible in the series.
    df.loc[df[snow_col] < 0, snow_col] = float("nan")

    # Store cleaned data
    snow = df[snow_col]
    data[station] = df

    # --------------------------------------------------
    # 2.3 Calculate descriptive statistics
    # --------------------------------------------------

    statistics[station] = {
        "max": snow.max(),
        "mean": snow.mean(),
        "valid": int(snow.notna().sum()),
        "missing": int(snow.isna().sum()),
        "negative_removed": negative_count,
        "invalid_timestamps": invalid_timestamps,
    }

    # --------------------------------------------------
    # 2.4 Print data-quality report
    # --------------------------------------------------

    print(f"\n{'=' * 50}")
    print(station)
    print(f"{'=' * 50}")

    print(
        "Period:",
        df.index.min(),
        "to",
        df.index.max(),
    )

    print("Total records:", len(df))
    print("Invalid timestamps removed:", invalid_timestamps)
    print("Negative snow-depth values excluded:", negative_count)
    print("Valid snow-depth values:", snow.notna().sum())
    print("Missing or invalid snow-depth values:", snow.isna().sum())

    if snow.notna().any():
        print(f"Maximum snow depth: {snow.max():.1f} mm")
        print(f"Mean snow depth: {snow.mean():.1f} mm")
    else:
        print("No valid snow-depth values available.")


# ==================================================
# 3. CREATE PLOT
# ==================================================

fig, ax = plt.subplots(figsize=(16, 8))

for station, df in data.items():

    settings = stations[station]
    snow_col = settings["column"]

    # Copy the cleaned snow-depth series
    snow = df[snow_col].copy()

    # Calculate time differences between consecutive records
    time_gap = df.index.to_series().diff()

    # Insert NaN at the later timestamp when the time gap
    # exceeds the maximum permitted interval.
    # This prevents lines from bridging long data gaps.
    snow.loc[time_gap > MAX_GAP] = float("nan")

    # Plot the time series
    ax.plot(
        df.index,
        snow,
        color=settings["color"],
        linewidth=0.7,
        label=station,
    )

# Titles and axis labels
ax.set_title(
    "Snow Depth at Three Weather Stations",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Snow depth (mm)", fontsize=12)

# Y-axis: 0 to 3000 mm
ax.set_ylim(0, 3000)

# X-axis: show June and December each year
ax.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=[6, 12])
)

# Display months and years
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%m\n%Y")
)

ax.tick_params(axis="x", labelsize=9)

ax.legend(
    loc="upper left",
    title="Weather station",
    frameon=True,
)

ax.grid(True, alpha=0.3)


# ==================================================
# 4. STATISTICS BOX
# ==================================================

stats_text = "DESCRIPTIVE STATISTICS\n\n"

for station in stations:

    values = statistics[station]

    max_text = (
        f"{values['max']:.1f} mm"
        if pd.notna(values["max"])
        else "No valid data"
    )

    mean_text = (
        f"{values['mean']:.1f} mm"
        if pd.notna(values["mean"])
        else "No valid data"
    )

    stats_text += (
        f"{station}\n"
        f"  Max: {max_text}\n"
        f"  Mean: {mean_text}\n"
        f"  Valid values: {values['valid']}\n"
        f"  Negative values excluded: "
        f"{values['negative_removed']}\n\n"
    )

# Place statistics in the upper-right corner
ax.text(
    0.985,
    0.97,
    stats_text.strip(),
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
        alpha=0.95,
    ),
)


# ==================================================
# 5. SAVE AND DISPLAY
# ==================================================

fig.tight_layout()

# Save the PDF in the output folder
output_file = OUTPUT_DIR / "snow_depth_three_stations_cleaned.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nPlot saved to:", output_file)
