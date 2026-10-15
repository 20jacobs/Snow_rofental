
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ==================================================
# 1. SETTINGS
# ==================================================

sns.set_theme(style="whitegrid")

BASE_DIR = Path(
    "/Users/franzimaster/Desktop/Master_Geographie/3/Hydro/Datenanalyse"
)

TIME_COL = "Date and time"

# Expected measurement interval is 10 minutes.
# Interrupt plotted lines when the gap exceeds 20 minutes.
MAX_GAP = pd.Timedelta(minutes=20)

stations = {
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


# ==================================================
# 2. LOAD AND CLEAN DATA
# ==================================================

data = {}
statistics = {}

for station, settings in stations.items():

    file_path = BASE_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Automatically detect the CSV delimiter.
    df = pd.read_csv(file_path, sep=None, engine="python")

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    required = [TIME_COL, settings["column"]]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{station}: missing columns {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # --------------------------------------------------
    # Parse timestamps in different formats.
    # Examples:
    # 2018-03-07 14:40:00
    # 2018-03-07 14:40:00.000
    # --------------------------------------------------

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    # Convert SWE measurements to numeric.
    df[settings["column"]] = pd.to_numeric(
        df[settings["column"]],
        errors="coerce",
    )

    # Remove negative SWE values.
    # NaN values are retained to preserve gaps in the plot.
    df.loc[
        df[settings["column"]] < 0,
        settings["column"],
    ] = float("nan")

    # Remove invalid timestamps and sort chronologically.
    df = (
        df.dropna(subset=[TIME_COL])
          .sort_values(TIME_COL)
          .set_index(TIME_COL)
    )

    # Report duplicate timestamps.
    duplicate_count = df.index.duplicated().sum()

    if duplicate_count > 0:
        print(
            f"\nWarning: {station} has "
            f"{duplicate_count} duplicate timestamps."
        )

    swe = df[settings["column"]]

    # --------------------------------------------------
    # Break lines across large timestamp gaps.
    # A missing SWE measurement also remains a break.
    # --------------------------------------------------

    time_gap = df.index.to_series().diff()

    gap_mask = time_gap > MAX_GAP

    # Insert NaN at the first observation after a large gap.
    # This prevents Matplotlib from connecting the two periods.
    df["SWE_plot"] = swe.copy()
    df.loc[gap_mask.to_numpy(), "SWE_plot"] = float("nan")

    data[station] = df

    # Statistics use valid, non-negative SWE values only.
    valid_swe = swe.dropna()

    statistics[station] = {
        "max": valid_swe.max(),
        "mean": valid_swe.mean(),
        "valid": valid_swe.count(),
        "missing": swe.isna().sum(),
        "negative_removed": 0,  # Counted separately below if needed.
        "large_gaps": int(gap_mask.sum()),
    }

    print(f"\n{station}")
    print("Period:", df.index.min(), "to", df.index.max())
    print("Records:", len(df))
    print("Valid SWE values:", valid_swe.count())
    print("Missing/invalid SWE values:", swe.isna().sum())
    print("Gaps exceeding 20 minutes:", int(gap_mask.sum()))
    print("Maximum SWE:", valid_swe.max())
    print("Mean SWE:", valid_swe.mean())


# ==================================================
# 3. CREATE PLOT
# ==================================================

fig, ax = plt.subplots(figsize=(16, 8))

for station, df in data.items():

    settings = stations[station]

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


# ==================================================
# 4. DESCRIPTIVE STATISTICS BOX - TOP RIGHT
# ==================================================

stats_lines = [
    "DESCRIPTIVE STATISTICS",
    "",
]

for station, settings in stations.items():

    values = statistics[station]

    stats_lines.extend([
        station,
        f"Valid measurements: {values['valid']:,}",
        (
            f"Mean:    {values['mean']:.1f} mm"
            if pd.notna(values["mean"])
            else "Mean:    N/A"
        ),
        (
            f"Maximum: {values['max']:.1f} mm"
            if pd.notna(values["max"])
            else "Maximum: N/A"
        ),
        f"Missing/invalid: {values['missing']:,}",
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


# ==================================================
# 5. X-AXIS: JUNE AND DECEMBER OF EVERY YEAR
# ==================================================

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


# ==================================================
# 6. SAVE AND DISPLAY
# ==================================================

fig.tight_layout()

output_file = BASE_DIR / "swe_two_stations_adjusted_timestamps.pdf"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nPlot saved to:", output_file)
