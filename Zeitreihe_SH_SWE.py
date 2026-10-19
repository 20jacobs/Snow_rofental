
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"
START_DATE = pd.Timestamp("2017-09-01")
MAX_GAP = pd.Timedelta(minutes=20)

# Convert millimetres to centimetres
MM_TO_CM = 0.1

# Snow depth stations
SNOW_STATIONS = {
    "Bella Vista (protected)": {
        "file": "1_BellaVista_10min_2015-2025.csv",
        "column": "USH9 snow depth II (mm)",
        "color": "#0072B2",
    },
    "Bella Vista (exposed)": {
        "file": "1_BellaVista_10min_2015-2025.csv",
        "column": "USH8 snow depth (mm)",
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

# Snow Water Equivalent stations
SWE_STATIONS = {
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
    "Proviantdepot": {
        "file": "3_Proviantdepot_10min_2019-2025.csv",
        "column": "SSG-2 snow water equivalent (mm)",
        "color": "forestgreen",
    },
}


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

def load_station_data(station, settings):
    """Load, clean and prepare one measurement time series."""

    file_path = DATA_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    # Automatically detect CSV delimiter
    df = pd.read_csv(file_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    required = [TIME_COL, settings["column"]]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{station}: Missing columns {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required].copy()
    measurement_col = settings["column"]

    # Parse timestamps
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    invalid_timestamps = int(df[TIME_COL].isna().sum())
    df = df.dropna(subset=[TIME_COL]).copy()

    # Convert measurements to numeric values
    df[measurement_col] = pd.to_numeric(
        df[measurement_col],
        errors="coerce",
    )

    # Exclude negative measurements
    negative_count = int((df[measurement_col] < 0).sum())
    df.loc[df[measurement_col] < 0, measurement_col] = np.nan

    # Remove exact duplicates
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Aggregate remaining duplicate timestamps
    duplicate_mask = df[TIME_COL].duplicated(keep=False)
    duplicate_timestamp_count = int(
        df.loc[duplicate_mask, TIME_COL].nunique()
    )

    df = (
        df.groupby(TIME_COL, as_index=False)
        .agg({measurement_col: "mean"})
        .sort_values(TIME_COL)
        .set_index(TIME_COL)
    )

    # Convert millimetres to centimetres
    df["value_cm"] = df[measurement_col] * MM_TO_CM

    # Keep the complete record history for data-quality reporting,
    # but plot only the requested period.
    df = df.loc[df.index >= START_DATE].copy()

    # Break lines across long gaps
    time_gap = df.index.to_series().diff()
    gap_mask = time_gap > MAX_GAP

    df["plot_cm"] = df["value_cm"].copy()
    df.loc[gap_mask.to_numpy(), "plot_cm"] = np.nan

    valid = df["value_cm"].dropna()

    statistics = {
        "max_cm": valid.max(),
        "mean_cm": valid.mean(),
        "valid": int(valid.count()),
        "missing": int(df["value_cm"].isna().sum()),
        "negative_removed": negative_count,
        "invalid_timestamps": invalid_timestamps,
        "exact_duplicates": exact_duplicates,
        "duplicate_timestamps": duplicate_timestamp_count,
        "large_gaps": int(gap_mask.sum()),
    }

    print(f"\n{'=' * 55}")
    print(station)
    print(f"{'=' * 55}")
    print("File:", file_path.name)
    print("Plot period:", START_DATE, "to", df.index.max())
    print("Invalid timestamps:", invalid_timestamps)
    print("Exact duplicates removed:", exact_duplicates)
    print("Duplicate timestamps aggregated:", duplicate_timestamp_count)
    print("Negative measurements excluded:", negative_count)
    print("Valid measurements since September 2017:", statistics["valid"])
    print("Missing/invalid measurements:", statistics["missing"])
    print("Gaps exceeding 20 minutes:", statistics["large_gaps"])

    if valid.notna().any():
        print(f"Maximum: {valid.max():.2f} cm")
        print(f"Mean: {valid.mean():.2f} cm")

    return df, statistics


# Load snow depth
snow_data = {}
snow_stats = {}

for station, settings in SNOW_STATIONS.items():
    snow_data[station], snow_stats[station] = load_station_data(
        station, settings
    )

# Load SWE
swe_data = {}
swe_stats = {}

for station, settings in SWE_STATIONS.items():
    swe_data[station], swe_stats[station] = load_station_data(
        station, settings
    )


# ============================================================
# 3. CREATE COMBINED FIGURE
# ============================================================

fig, (ax_snow, ax_swe) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(18, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]},
)

# ------------------------------------------------------------
# 3.1 Snow depth
# ------------------------------------------------------------

for station, df in snow_data.items():
    settings = SNOW_STATIONS[station]

    ax_snow.plot(
        df.index,
        df["plot_cm"],
        color=settings["color"],
        linewidth=0.7,
        label=station,
    )

ax_snow.set_title(
    "Snow Depth",
    fontsize=15,
    fontweight="bold",
    pad=10,
)
ax_snow.set_ylabel("Snow depth (cm)", fontsize=12)
ax_snow.set_ylim(0, 250)
ax_snow.legend(
    loc="upper left",
    title="Weather station",
    frameon=True,
    fontsize=9,
)
ax_snow.grid(True, alpha=0.3)


# ============================================================
# DESCRIPTIVE STATISTICS: SNOW DEPTH
# ============================================================

snow_stats_text = "DESCRIPTIVE STATISTICS\n\n"

for station in SNOW_STATIONS:
    values = snow_stats[station]

    max_text = (
        f"{values['max_cm']:.1f} cm"
        if pd.notna(values["max_cm"])
        else "N/A"
    )

    mean_text = (
        f"{values['mean_cm']:.1f} cm"
        if pd.notna(values["mean_cm"])
        else "N/A"
    )

    snow_stats_text += (
        f"{station}\n"
        f"  Mean: {mean_text}\n"
        f"  Maximum: {max_text}\n"
        f"  Valid: {values['valid']:,}\n"
        f"  Missing: {values['missing']:,}\n\n"
    )

ax_snow.text(
    0.985, 0.97,
    snow_stats_text.strip(),
    transform=ax_snow.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    family="monospace",
    linespacing=1.25,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="grey",
        linewidth=0.8,
        alpha=0.92,
    ),
    zorder=6,
)



# ------------------------------------------------------------
# 3.2 Snow Water Equivalent
# ------------------------------------------------------------

for station, df in swe_data.items():
    settings = SWE_STATIONS[station]

    ax_swe.plot(
        df.index,
        df["plot_cm"],
        color=settings["color"],
        linewidth=0.7,
        label=station,
    )

ax_swe.set_title(
    "Snow Water Equivalent (SWE)",
    fontsize=15,
    fontweight="bold",
    pad=10,
)
ax_swe.set_ylabel("SWE (cm)", fontsize=12)
ax_swe.set_ylim(0, 150)
ax_swe.set_xlabel("Date", fontsize=12)
ax_swe.legend(
    loc="upper left",
    title="Weather station",
    frameon=True,
    fontsize=9,
)
ax_swe.grid(True, alpha=0.3)


# ============================================================
# DESCRIPTIVE STATISTICS: SWE
# ============================================================

swe_stats_text = "DESCRIPTIVE STATISTICS\n\n"

for station in SWE_STATIONS:
    values = swe_stats[station]

    max_text = (
        f"{values['max_cm']:.1f} cm"
        if pd.notna(values["max_cm"])
        else "N/A"
    )

    mean_text = (
        f"{values['mean_cm']:.1f} cm"
        if pd.notna(values["mean_cm"])
        else "N/A"
    )

    swe_stats_text += (
        f"{station}\n"
        f"  Mean: {mean_text}\n"
        f"  Maximum: {max_text}\n"
        f"  Valid: {values['valid']:,}\n"
        f"  Missing: {values['missing']:,}\n\n"
    )

ax_swe.text(
    0.985, 0.97,
    swe_stats_text.strip(),
    transform=ax_swe.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    family="monospace",
    linespacing=1.25,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="grey",
        linewidth=0.8,
        alpha=0.92,
    ),
    zorder=6,
)


# ============================================================
# 4. COMMON X-AXIS
# ============================================================

ax_swe.set_xlim(left=START_DATE)

ax_swe.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=[12])
)
ax_swe.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.setp(
    ax_swe.get_xticklabels(),
    rotation=0,
    ha="center",
)


# ============================================================
# 5. SAVE COMBINED PNG
# ============================================================

fig.suptitle(
    "Snow Depth and Snow Water Equivalent "
    "at the Weather Stations",
    fontsize=17,
    fontweight="bold",
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

output_file = OUTPUT_DIR / "snow_depth_and_SWE_combined.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("\nAnalysis complete.")
print("Combined plot saved to:", output_file)
