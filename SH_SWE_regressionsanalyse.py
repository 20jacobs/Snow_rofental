
# ============================================================
# SH–SWE analysis for the entire available dataset
#
# - Reads all available data from both stations
# - Parses timestamps with or without fractional seconds
# - Removes negative SH and SWE measurements
# - Calculates daily means
# - Defines snow years from July to June
# - Identifies the seasonal SWE maximum
# - Classifies accumulation and melting relative to that maximum
# - Performs regressions for complete snow years
# - Plots seasonal SWE maxima and SH at the SWE maximum date
# - Exports tables and figures
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = Path(
    "/Users/franzimaster/Desktop/Master_Geographie/3/Hydro/Datenanalyse"
)

OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"

# Snow year runs from July through June.
SEASON_START_MONTH = 7

STATIONS = {
    "Bella Vista": {
        "file": "1_BellaVista_10min_2015-2025.csv",
        "sh_col": "USH9 snow depth II (mm)",
        "swe_col": "SSG-2 snow water equivalent II (mm)",
        "color": "steelblue",
    },
    "Proviantdepot": {
        "file": "3_Proviantdepot_10min_2019-2025.csv",
        "sh_col": "USH9 snow depth (mm)",
        "swe_col": "SSG-2 snow water equivalent (mm)",
        "color": "forestgreen",
    },
}


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

def load_station_data(station_name, settings):
    """Load station data, parse timestamps, and remove negative values."""

    file_path = DATA_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep=None, engine="python")

    required_cols = [
        TIME_COL,
        settings["sh_col"],
        settings["swe_col"],
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{station_name}: Missing columns {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    # Supports timestamps such as:
    # 2018-03-07 14:40:00
    # 2018-03-07 14:40:00.000
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    # Convert measurement columns to numeric values.
    df[settings["sh_col"]] = pd.to_numeric(
        df[settings["sh_col"]], errors="coerce"
    )
    df[settings["swe_col"]] = pd.to_numeric(
        df[settings["swe_col"]], errors="coerce"
    )

    # Remove rows with invalid timestamps.
    df = df.dropna(subset=[TIME_COL]).copy()

    # Remove negative measurements by treating them as missing.
    df.loc[df[settings["sh_col"]] < 0, settings["sh_col"]] = np.nan
    df.loc[df[settings["swe_col"]] < 0, settings["swe_col"]] = np.nan

    # Sort and report duplicate timestamps.
    df = df.sort_values(TIME_COL)

    duplicate_count = df[TIME_COL].duplicated().sum()

    if duplicate_count:
        print(
            f"Warning: {station_name} contains "
            f"{duplicate_count} duplicate timestamps."
        )

    df = df.set_index(TIME_COL)

    # Calculate daily means from all valid observations.
    daily = df.resample("D").agg(
        **{
            "SH (mm)": (settings["sh_col"], "mean"),
            "SWE (mm)": (settings["swe_col"], "mean"),
            "SH valid measurements": (settings["sh_col"], "count"),
            "SWE valid measurements": (settings["swe_col"], "count"),
        }
    )

    daily.index.name = "Date"
    daily = daily.reset_index()

    daily["Station"] = station_name
    daily["Snow year"] = (
        daily["Date"].dt.year
        + (daily["Date"].dt.month >= SEASON_START_MONTH).astype(int)
    )

    print(
        f"{station_name}: {len(df):,} timestamped records; "
        f"{daily['Date'].min().date()} to {daily['Date'].max().date()}"
    )

    return daily


daily_data = pd.concat(
    [
        load_station_data(station_name, settings)
        for station_name, settings in STATIONS.items()
    ],
    ignore_index=True,
)

daily_data = daily_data.sort_values(
    ["Station", "Date"]
).reset_index(drop=True)


# ============================================================
# 3. DETERMINE SEASONAL SWE MAXIMA
# ============================================================

def determine_seasonal_maxima(daily):
    """
    Determine the date of maximum daily mean SWE for each station
    and snow year. SH at that date is also retained.
    """

    rows = []

    for (station, snow_year), group in daily.groupby(
        ["Station", "Snow year"], sort=True
    ):
        group = group.sort_values("Date")

        valid_swe = group.dropna(subset=["SWE (mm)"])

        # No SWE observations in this season.
        if valid_swe.empty:
            continue

        # Select the first date if the maximum occurs more than once.
        max_idx = valid_swe["SWE (mm)"].idxmax()
        max_row = valid_swe.loc[max_idx]

        # A complete season must have observations spanning the
        # nominal season boundaries: July 1 through June 30.
        season_start = pd.Timestamp(
            year=snow_year - 1, month=7, day=1
        )
        season_end = pd.Timestamp(
            year=snow_year, month=6, day=30
        )

        # Check the available daily record's temporal coverage.
        # This checks boundary coverage, not the absence of internal gaps.
        season_complete = (
            group["Date"].min() <= season_start
            and group["Date"].max() >= season_end
        )

        rows.append(
            {
                "Station": station,
                "Snow year": snow_year,
                "Season start": season_start,
                "Season end": season_end,
                "Season complete": season_complete,
                "SWE maximum date": max_row["Date"],
                "SWE maximum (mm)": max_row["SWE (mm)"],
                "SH at SWE maximum (mm)": max_row["SH (mm)"],
            }
        )

    return pd.DataFrame(rows)


seasonal_maxima = determine_seasonal_maxima(daily_data)

if seasonal_maxima.empty:
    raise ValueError(
        "No seasonal SWE maxima could be calculated. "
        "Check the SWE column names and source data."
    )

print("\nSeasonal SWE maxima:")
print(seasonal_maxima.to_string(index=False))


# ============================================================
# 4. ASSIGN ACCUMULATION AND MELTING PHASES
# ============================================================

def assign_phases(daily, maxima):
    """
    Classify dates before the seasonal SWE maximum as accumulation
    and dates after it as melting. The maximum date itself is excluded.
    """

    result = daily.copy()
    result["Phase"] = pd.NA
    result["SWE maximum date"] = pd.NaT
    result["Season complete"] = False

    maxima_lookup = maxima.set_index(
        ["Station", "Snow year"]
    )

    for idx, row in result.iterrows():
        key = (row["Station"], row["Snow year"])

        if key not in maxima_lookup.index:
            continue

        max_row = maxima_lookup.loc[key]
        max_date = max_row["SWE maximum date"]

        result.at[idx, "SWE maximum date"] = max_date
        result.at[idx, "Season complete"] = bool(
            max_row["Season complete"]
        )

        if row["Date"] < max_date:
            result.at[idx, "Phase"] = "Accumulation"
        elif row["Date"] > max_date:
            result.at[idx, "Phase"] = "Melting"
        # The SWE maximum date remains unclassified.

    return result


daily_data = assign_phases(daily_data, seasonal_maxima)


# ============================================================
# 5. PREPARE DATA FOR REGRESSION
# ============================================================

# Regressions include only:
# - complete snow years,
# - valid daily means for both variables,
# - strictly positive SH and SWE,
# - dates before or after the seasonal SWE maximum.

analysis_data = daily_data[
    daily_data["Season complete"]
    & daily_data["SH (mm)"].notna()
    & daily_data["SWE (mm)"].notna()
    & (daily_data["SH (mm)"] > 0)
    & (daily_data["SWE (mm)"] > 0)
    & daily_data["Phase"].isin(["Accumulation", "Melting"])
].copy()


# ============================================================
# 6. REGRESSION FUNCTION
# ============================================================

def calculate_regression(group):
    """Calculate linear regression statistics for SH versus SWE."""

    group = group.dropna(subset=["SH (mm)", "SWE (mm)"])

    x = group["SH (mm)"].to_numpy(dtype=float)
    y = group["SWE (mm)"].to_numpy(dtype=float)

    n = len(group)

    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {
            "n": n,
            "Slope": np.nan,
            "Intercept": np.nan,
            "R²": np.nan,
            "Pearson r": np.nan,
            "p-value": np.nan,
            "Slope standard error": np.nan,
            "Slope 95% CI lower (approx.)": np.nan,
            "Slope 95% CI upper (approx.)": np.nan,
        }

    regression = linregress(x, y)

    return {
        "n": n,
        "Slope": regression.slope,
        "Intercept": regression.intercept,
        "R²": regression.rvalue**2,
        "Pearson r": regression.rvalue,
        "p-value": regression.pvalue,
        "Slope standard error": regression.stderr,
        "Slope 95% CI lower (approx.)": (
            regression.slope - 1.96 * regression.stderr
        ),
        "Slope 95% CI upper (approx.)": (
            regression.slope + 1.96 * regression.stderr
        ),
    }


# ============================================================
# 7. POOLED REGRESSION BY STATION AND PHASE
# ============================================================

pooled_results = []

for station in STATIONS:
    for phase in ["Accumulation", "Melting"]:
        subset = analysis_data[
            (analysis_data["Station"] == station)
            & (analysis_data["Phase"] == phase)
        ]

        stats = calculate_regression(subset)

        pooled_results.append(
            {
                "Station": station,
                "Phase": phase,
                **stats,
            }
        )

pooled_results = pd.DataFrame(pooled_results)

print("\nPooled regression results:")
print(pooled_results.to_string(index=False))


# ============================================================
# 8. REGRESSION BY STATION, SNOW YEAR, AND PHASE
# ============================================================

seasonal_results = []

for (station, snow_year, phase), group in analysis_data.groupby(
    ["Station", "Snow year", "Phase"], sort=True
):
    stats = calculate_regression(group)

    seasonal_results.append(
        {
            "Station": station,
            "Snow year": snow_year,
            "Phase": phase,
            **stats,
        }
    )

seasonal_results = pd.DataFrame(seasonal_results)

print("\nRegression results by snow year:")
if seasonal_results.empty:
    print("No complete seasons with sufficient data for regression.")
else:
    print(seasonal_results.to_string(index=False))


# ============================================================
# 9. PLOT SH–SWE REGRESSIONS
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 11),
    sharex=False,
    sharey=False,
)

for row_idx, station in enumerate(STATIONS):
    station_color = STATIONS[station]["color"]

    for col_idx, phase in enumerate(["Accumulation", "Melting"]):
        ax = axes[row_idx, col_idx]

        subset = analysis_data[
            (analysis_data["Station"] == station)
            & (analysis_data["Phase"] == phase)
        ]

        ax.scatter(
            subset["SH (mm)"],
            subset["SWE (mm)"],
            s=12,
            alpha=0.35,
            color=station_color,
            edgecolors="none",
        )

        stats = calculate_regression(subset)

        if np.isfinite(stats["Slope"]):
            x_line = np.linspace(
                subset["SH (mm)"].min(),
                subset["SH (mm)"].max(),
                100,
            )
            y_line = stats["Intercept"] + stats["Slope"] * x_line

            ax.plot(
                x_line,
                y_line,
                color="black",
                linewidth=2,
            )

            annotation = (
                f"n = {stats['n']}\n"
                f"Slope = {stats['Slope']:.3f}\n"
                f"R² = {stats['R²']:.3f}\n"
                f"p = {stats['p-value']:.3g}"
            )
        else:
            annotation = f"n = {stats['n']}\nInsufficient variation"

        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "lightgray",
            },
        )

        ax.set_title(f"{station} – {phase}")
        ax.set_xlabel("Snow depth SH (mm)")
        ax.set_ylabel("Snow water equivalent SWE (mm)")
        ax.grid(True, alpha=0.25)

fig.suptitle(
    "Snow depth (SH) versus snow water equivalent (SWE)\n"
    "All available data; complete snow years only",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

regression_plot_path = (
    OUTPUT_DIR / "SH_SWE_regression_Akkumulation_Schmelze_all_data.pdf"
)

fig.savefig(regression_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 10. PLOT SEASONAL SWE MAXIMA AND SH AT THE SAME DATE
# ============================================================

fig, axes = plt.subplots(
    nrows=len(STATIONS),
    ncols=1,
    figsize=(14, 5 * len(STATIONS)),
    squeeze=False,
)

for i, (station, settings) in enumerate(STATIONS.items()):
    ax = axes[i, 0]
    ax_sh = ax.twinx()

    station_maxima = seasonal_maxima[
        seasonal_maxima["Station"] == station
    ].sort_values("SWE maximum date")

    if station_maxima.empty:
        ax.set_title(f"{station}: no seasonal SWE maxima available")
        continue

    dates = station_maxima["SWE maximum date"]
    swe_maxima = station_maxima["SWE maximum (mm)"]
    sh_at_max = station_maxima["SH at SWE maximum (mm)"]

    # Main axis: seasonal SWE maxima.
    ax.plot(
        dates,
        swe_maxima,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        color=settings["color"],
        label="Seasonal SWE maximum",
    )

    # Secondary axis: SH measured on the SWE maximum date.
    ax_sh.plot(
        dates,
        sh_at_max,
        marker=None,
        linestyle=":",
        linewidth=2.2,
        color="black",
        label="SH at SWE maximum date",
    )

    # Mark maxima from incomplete seasons.
    incomplete = ~station_maxima["Season complete"].to_numpy()

    if incomplete.any():
        ax.scatter(
            dates.iloc[np.where(incomplete)[0]],
            swe_maxima.iloc[np.where(incomplete)[0]],
            marker="x",
            s=90,
            color="red",
            zorder=5,
            label="Maximum in incomplete season",
        )

    ax.set_title(f"{station}: seasonal SWE maxima and SH")
    ax.set_xlabel("Date of seasonal SWE maximum")
    ax.set_ylabel("Seasonal maximum SWE (mm)")
    ax_sh.set_ylabel("Snow depth SH at SWE maximum date (mm)")

    ax.grid(True, alpha=0.25)

    # Combine legends from both y-axes.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_sh.get_legend_handles_labels()

    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=9,
    )

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

fig.suptitle(
    "Seasonal SWE maxima and snow depth at the SWE maximum date\n"
    "All available data; July–June snow years",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

maxima_plot_path = (
    OUTPUT_DIR / "SH_SWE_saisonale_Maxima_all_data.pdf"
)

fig.savefig(maxima_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 11. EXPORT RESULTS
# ============================================================

seasonal_maxima_path = (
    OUTPUT_DIR / "saisonale_SWE_maxima_all_data.csv"
)

pooled_results_path = (
    OUTPUT_DIR / "SH_SWE_regression_gesamt_all_data.csv"
)

seasonal_results_path = (
    OUTPUT_DIR / "SH_SWE_regression_pro_Schneejahr_all_data.csv"
)

daily_data_path = (
    OUTPUT_DIR / "SH_SWE_Tagesdaten_mit_Phase_all_data.csv"
)

seasonal_maxima.to_csv(
    seasonal_maxima_path,
    index=False,
    encoding="utf-8-sig",
)

pooled_results.to_csv(
    pooled_results_path,
    index=False,
    encoding="utf-8-sig",
)

seasonal_results.to_csv(
    seasonal_results_path,
    index=False,
    encoding="utf-8-sig",
)

daily_data.to_csv(
    daily_data_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\nAnalysis complete.")
print(f"Regression figure: {regression_plot_path}")
print(f"Seasonal maxima figure: {maxima_plot_path}")
print(f"Seasonal maxima table: {seasonal_maxima_path}")
print(f"Pooled regression table: {pooled_results_path}")
print(f"Seasonal regression table: {seasonal_results_path}")
print(f"Daily data table: {daily_data_path}")
