
# ============================================================
# SH-Analysis: Bella Vista (exposed), 2015-2016
#
# - Analyses snow depth for the years 2015 and 2016 only
# - Reads the CSV file from the input directory
# - Creates an output folder for generated files
# - Saves the figure as a high-resolution PNG
# - Parses timestamps and validates measurement columns
# - Handles duplicate timestamps transparently
# - Excludes negative measurements
# - Calculates descriptive statistics for 2015-2016 only
# - Handles data gaps by interrupting plotted lines
# - Supports timestamps with and without milliseconds
# - Plots the snow-depth time series
# - Displays a compact horizontal statistics box
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

# Input and output directories
DATA_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"

# Analysis period: 2015 and 2016
START_DATE = pd.Timestamp("2015-01-01")
END_DATE = pd.Timestamp("2017-01-01")  # Exclusive

# Expected measurement interval: 10 minutes.
# Lines are interrupted if the gap exceeds 20 minutes.
MAX_GAP = pd.Timedelta(minutes=20)

stations = {
    "Bella Vista (exposed)": {
        "file": "1_BellaVista_10min_2015-2025.csv",
        "column": "USH8 snow depth (mm)",
        "color": "#D55E00",
    },
}


# ==================================================
# 2. LOAD, CLEAN AND FILTER DATA
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

    # Check required columns
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
    # 2.1 Convert timestamps and measurements
    # --------------------------------------------------

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    df[snow_col] = pd.to_numeric(
        df[snow_col],
        errors="coerce",
    )

    # Count invalid timestamps before removing them
    invalid_timestamps = int(df[TIME_COL].isna().sum())

    # Remove rows without valid timestamps
    df = df.dropna(subset=[TIME_COL])

    # --------------------------------------------------
    # 2.2 Filter to 2015-2016
    # --------------------------------------------------

    df = df.loc[
        (df[TIME_COL] >= START_DATE)
        & (df[TIME_COL] < END_DATE)
    ].copy()

    # Sort chronologically
    df = df.sort_values(TIME_COL)

    # Handle duplicate timestamps transparently:
    # retain the first record for each timestamp
    duplicate_count = int(
        df[TIME_COL].duplicated(keep="first").sum()
    )

    df = df.drop_duplicates(
        subset=TIME_COL,
        keep="first",
    )

    # Set timestamps as index
    df = df.set_index(TIME_COL)

    # --------------------------------------------------
    # 2.3 Remove negative snow-depth values
    # --------------------------------------------------

    negative_count = int((df[snow_col] < 0).sum())

    # Replace negative measurements with NaN
    # so that they are excluded from statistics and plotting
    df.loc[df[snow_col] < 0, snow_col] = float("nan")

    snow = df[snow_col]

    # Store filtered and cleaned data
    data[station] = df

    # --------------------------------------------------
    # 2.4 Calculate statistics for 2015-2016 only
    # --------------------------------------------------

    statistics[station] = {
        "max": snow.max(),
        "mean": snow.mean(),
        "valid": int(snow.notna().sum()),
        "missing": int(snow.isna().sum()),
        "negative_removed": negative_count,
        "invalid_timestamps": invalid_timestamps,
        "duplicates_removed": duplicate_count,
    }

    # --------------------------------------------------
    # 2.5 Print data-quality report
    # --------------------------------------------------

    print(f"\n{'=' * 55}")
    print(station)
    print("Analysis period: 2015-01-01 to 2016-12-31")
    print(f"{'=' * 55}")

    if not df.empty:
        print(
            "Available period:",
            df.index.min(),
            "to",
            df.index.max(),
        )
    else:
        print("No records found in the selected period.")

    print("Records after filtering:", len(df))
    print("Invalid timestamps removed:", invalid_timestamps)
    print("Duplicate timestamps removed:", duplicate_count)
    print("Negative snow-depth values excluded:", negative_count)
    print("Valid snow-depth values:", snow.notna().sum())
    print("Missing or invalid snow-depth values:", snow.isna().sum())

    if snow.notna().any():
        print(f"Maximum snow depth: {snow.max():.1f} mm")
        print(f"Mean snow depth: {snow.mean():.1f} mm")
    else:
        print("No valid snow-depth values available.")


# ==================================================
# 3. CREATE TIME-SERIES PLOT
# ==================================================

fig, ax = plt.subplots(figsize=(16, 8))

for station, df in data.items():

    settings = stations[station]
    snow_col = settings["column"]

    snow = df[snow_col].copy()

    # Calculate time differences between consecutive records
    time_gap = df.index.to_series().diff()

    # Interrupt lines after gaps exceeding MAX_GAP
    snow.loc[time_gap > MAX_GAP] = float("nan")

    # Plot the time series
    ax.plot(
        df.index,
        snow,
        color=settings["color"],
        linewidth=0.7,
        label=station,
    )

# --------------------------------------------------
# 3.1 Titles and axis labels
# --------------------------------------------------

ax.set_title(
    "Snow Depth at Bella Vista (exposed), 2015-2016",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Snow depth (mm)", fontsize=12)

# Keep the original y-axis range
ax.set_ylim(0, 3000)

# Restrict the x-axis to the selected period
ax.set_xlim(START_DATE, END_DATE)

# Display dates every three months
ax.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=[1, 4, 7, 10])
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.tick_params(axis="x", labelsize=9)

ax.legend(
    loc="upper left",
    title="Weather station",
    frameon=True,
)

ax.grid(True, alpha=0.3)


# ==================================================
# 4. COMPACT HORIZONTAL STATISTICS BOX
# ==================================================

values = statistics["Bella Vista (exposed)"]

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

# Arrange the main statistics horizontally
stats_text = (
    f"Max: {max_text}   |   "
    f"Mean: {mean_text}   |   "
    f"Valid: {values['valid']:,}   |   "
    f"Excluded negative: {values['negative_removed']:,}"
)

# Place the box in the upper-right corner
ax.text(
    0.985,
    0.97,
    stats_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    family="monospace",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="grey",
        alpha=0.9,
    ),
)


# ==================================================
# 5. SAVE AND DISPLAY
# ==================================================

fig.tight_layout()

output_file = (
    OUTPUT_DIR
    / "snow_depth_BellaVista_2015-2016.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nPlot saved to:", output_file)
