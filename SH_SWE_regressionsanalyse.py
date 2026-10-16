
# ============================================================
# SH-SWE REGRESSION ANALYSIS
#
# Reproducible analysis for the entire available dataset
#
# - Uses paths relative to this script
# - Reads CSV files from the repository directory
# - Creates an output folder for generated files
# - Saves all figures and result tables in the output folder
# - Parses timestamps and validates measurement columns
# - Handles duplicate timestamps transparently
# - Excludes negative measurements
# - Calculates daily means and valid-measurement counts
# - Defines snow years from July through June
# - Identifies seasonal SWE maxima
# - Evaluates seasonal data coverage
# - Classifies accumulation and melting phases
# - Performs pooled and snow-year-specific regressions
# - Exports figures and tables for reproducibility
#
# IMPORTANT:
# Conventional regression p-values and confidence intervals
# assume independent residuals. Temporal autocorrelation is
# not corrected for in this version.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import linregress, t


# ============================================================
# 1. SETTINGS
# ============================================================

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files remain in the repository directory
DATA_DIR = BASE_DIR / "input"

# Create an output folder for all generated files
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"

# Snow year runs from July through June
SEASON_START_MONTH = 7

# Expected measurement interval: 10 minutes
EXPECTED_INTERVAL_MINUTES = 10

# Minimum coverage required to classify a snow year as complete
# This methodological threshold should be justified in the thesis
MIN_SEASON_COVERAGE = 0.90

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

PHASES = ["Accumulation", "Melting"]


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

def load_station_data(station_name, settings):
    """
    Load station data, validate columns, clean measurements,
    handle duplicate timestamps, and calculate daily means.

    Negative measurements are treated as missing values.
    Duplicate timestamps with conflicting values are aggregated
    to one timestamp using the mean of valid measurements.
    """

    file_path = DATA_DIR / settings["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Ensure that the CSV file is in: {DATA_DIR}"
        )

    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
    )

    # Normalize column names
    df.columns = df.columns.str.strip()

    required_cols = [
        TIME_COL,
        settings["sh_col"],
        settings["swe_col"],
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{station_name}: Missing columns {missing_cols}.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    # --------------------------------------------------------
    # 2.1 Parse timestamps
    # --------------------------------------------------------

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    invalid_timestamps = int(df[TIME_COL].isna().sum())

    df = df.dropna(subset=[TIME_COL]).copy()

    # --------------------------------------------------------
    # 2.2 Convert measurements to numeric values
    # --------------------------------------------------------

    for col in [settings["sh_col"], settings["swe_col"]]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Count negative values before replacing them
    negative_sh_count = int(
        (df[settings["sh_col"]] < 0).sum()
    )

    negative_swe_count = int(
        (df[settings["swe_col"]] < 0).sum()
    )

    # Negative values are invalid measurements
    df.loc[
        df[settings["sh_col"]] < 0,
        settings["sh_col"],
    ] = np.nan

    df.loc[
        df[settings["swe_col"]] < 0,
        settings["swe_col"],
    ] = np.nan

    # --------------------------------------------------------
    # 2.3 Handle duplicate records
    # --------------------------------------------------------

    # Remove fully identical rows
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Count timestamps that still occur more than once
    duplicate_timestamp_mask = df[TIME_COL].duplicated(
        keep=False
    )

    duplicate_timestamp_rows = int(
        duplicate_timestamp_mask.sum()
    )

    duplicate_timestamp_count = int(
        df.loc[duplicate_timestamp_mask, TIME_COL].nunique()
    )

    if duplicate_timestamp_count:
        print(
            f"Warning: {station_name}: "
            f"{duplicate_timestamp_count} timestamps have "
            f"multiple records after exact duplicates were removed."
        )

        # Aggregate to one record per timestamp.
        # Valid observations are averaged independently for SH and SWE.
        df = (
            df.groupby(TIME_COL, as_index=False)
            .agg(
                **{
                    settings["sh_col"]: (
                        settings["sh_col"], "mean"
                    ),
                    settings["swe_col"]: (
                        settings["swe_col"], "mean"
                    ),
                }
            )
        )

    df = df.sort_values(TIME_COL).set_index(TIME_COL)

    # --------------------------------------------------------
    # 2.4 Calculate daily means and valid measurement counts
    # --------------------------------------------------------

    sh_col = settings["sh_col"]
    swe_col = settings["swe_col"]

    daily = df.resample("D").agg(
        **{
            "SH (mm)": (sh_col, "mean"),
            "SWE (mm)": (swe_col, "mean"),
            "SH valid measurements": (sh_col, "count"),
            "SWE valid measurements": (swe_col, "count"),
        }
    )

    daily.index.name = "Date"
    daily = daily.reset_index()

    daily["Station"] = station_name

    # Snow year: July-December belong to the following year;
    # January-June belong to the current year.
    daily["Snow year"] = (
        daily["Date"].dt.year
        + (
            daily["Date"].dt.month >= SEASON_START_MONTH
        ).astype(int)
    )

    # --------------------------------------------------------
    # 2.5 Print data-quality report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(station_name)
    print("=" * 60)

    print("Input file:", file_path.name)
    print("Period:", df.index.min(), "to", df.index.max())
    print("Valid timestamped records:", len(df))
    print("Invalid timestamps removed:", invalid_timestamps)
    print("Exact duplicate rows removed:", exact_duplicates)
    print(
        "Timestamps with multiple records:",
        duplicate_timestamp_count,
    )
    print("Negative SH measurements excluded:", negative_sh_count)
    print("Negative SWE measurements excluded:", negative_swe_count)
    print("Daily records:", len(daily))
    print(
        "Days with valid SH means:",
        int(daily["SH (mm)"].notna().sum()),
    )
    print(
        "Days with valid SWE means:",
        int(daily["SWE (mm)"].notna().sum()),
    )

    return daily


# Load all stations
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
# 3. DETERMINE SEASONAL SWE MAXIMA AND COVERAGE
# ============================================================

def determine_seasonal_maxima(daily):
    """
    Determine the maximum daily mean SWE for each station
    and snow year.

    Record snow depth at the date of the SWE maximum.
    A season is classified as complete only if:
      1. Its record spans both seasonal boundaries.
      2. SH daily coverage meets the defined threshold.
      3. SWE daily coverage meets the defined threshold.

    Coverage is based on valid daily means, not on the
    completeness of individual 10-minute observations.
    """

    rows = []

    for (station, snow_year), group in daily.groupby(
        ["Station", "Snow year"],
        sort=True,
    ):

        group = group.sort_values("Date")

        # July 1 of the previous calendar year through June 30
        # of the named snow year.
        season_start = pd.Timestamp(
            year=int(snow_year) - 1,
            month=7,
            day=1,
        )

        season_end = pd.Timestamp(
            year=int(snow_year),
            month=6,
            day=30,
        )

        expected_days = (
            season_end - season_start
        ).days + 1

        season_data = group.loc[
            group["Date"].between(
                season_start,
                season_end,
            )
        ].copy()

        if season_data.empty:
            continue

        # Count valid daily means for both variables
        valid_sh_days = int(
            season_data["SH (mm)"].notna().sum()
        )

        valid_swe_days = int(
            season_data["SWE (mm)"].notna().sum()
        )

        sh_coverage = valid_sh_days / expected_days
        swe_coverage = valid_swe_days / expected_days

        # Check temporal coverage at the seasonal boundaries
        boundary_coverage = (
            group["Date"].min() <= season_start
            and group["Date"].max() >= season_end
        )

        season_complete = (
            boundary_coverage
            and sh_coverage >= MIN_SEASON_COVERAGE
            and swe_coverage >= MIN_SEASON_COVERAGE
        )

        # Only valid SWE observations can define the maximum
        valid_swe = season_data.dropna(
            subset=["SWE (mm)"]
        )

        if valid_swe.empty:
            continue

        # If several dates share the maximum, select the first
        max_idx = valid_swe["SWE (mm)"].idxmax()
        max_row = valid_swe.loc[max_idx]

        rows.append(
            {
                "Station": station,
                "Snow year": int(snow_year),
                "Season start": season_start,
                "Season end": season_end,
                "Boundary coverage": boundary_coverage,
                "Valid SH days": valid_sh_days,
                "Valid SWE days": valid_swe_days,
                "Expected days": expected_days,
                "SH daily coverage (%)": sh_coverage * 100,
                "SWE daily coverage (%)": swe_coverage * 100,
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
        "Check the SWE columns and source data."
    )

print("\nSeasonal SWE maxima and coverage:")
print(seasonal_maxima.to_string(index=False))


# ============================================================
# 4. ASSIGN ACCUMULATION AND MELTING PHASES
# ============================================================

def assign_phases(daily, maxima):
    """
    Dates before the SWE maximum are classified as accumulation;
    dates after it are classified as melting.

    The maximum date itself is left unclassified.
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

    return result


daily_data = assign_phases(
    daily_data,
    seasonal_maxima,
)


# ============================================================
# 5. PREPARE DATA FOR REGRESSION
# ============================================================

# Include only:
# - seasons meeting the completeness criterion,
# - valid daily means for both variables,
# - positive SH and SWE values,
# - dates before or after the SWE maximum.
#
# Zero values are excluded because the analysis focuses on
# snow-covered conditions. This selection should be stated
# explicitly in the thesis.

analysis_data = daily_data.loc[
    daily_data["Season complete"]
    & daily_data["SH (mm)"].notna()
    & daily_data["SWE (mm)"].notna()
    & (daily_data["SH (mm)"] > 0)
    & (daily_data["SWE (mm)"] > 0)
    & daily_data["Phase"].isin(PHASES)
].copy()

print("\nRegression dataset:")
print("Number of daily observations:", len(analysis_data))

if analysis_data.empty:
    print(
        "Warning: No observations meet all regression criteria. "
        "Regression tables will contain no valid estimates."
    )


# ============================================================
# 6. REGRESSION FUNCTION
# ============================================================

def calculate_regression(group):
    """
    Fit an ordinary least-squares linear regression:

        SWE = Intercept + Slope * SH

    Conventional standard errors, p-values and confidence
    intervals are reported.

    IMPORTANT:
    These inferential statistics assume independent residuals.
    They do not correct for temporal autocorrelation.
    """

    group = group.dropna(
        subset=["SH (mm)", "SWE (mm)"]
    )

    x = group["SH (mm)"].to_numpy(dtype=float)
    y = group["SWE (mm)"].to_numpy(dtype=float)

    n = len(group)

    empty_result = {
        "n": n,
        "Slope": np.nan,
        "Intercept": np.nan,
        "R²": np.nan,
        "Pearson r": np.nan,
        "p-value": np.nan,
        "Slope standard error": np.nan,
        "Slope 95% CI lower": np.nan,
        "Slope 95% CI upper": np.nan,
    }

    if (
        n < 3
        or np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return empty_result

    regression = linregress(x, y)

    # Conventional two-sided 95% t-based confidence interval
    # for the slope, under standard OLS assumptions.
    degrees_of_freedom = n - 2
    t_critical = t.ppf(
        0.975,
        df=degrees_of_freedom,
    )

    slope_se = regression.stderr

    return {
        "n": n,
        "Slope": regression.slope,
        "Intercept": regression.intercept,
        "R²": regression.rvalue ** 2,
        "Pearson r": regression.rvalue,
        "p-value": regression.pvalue,
        "Slope standard error": slope_se,
        "Slope 95% CI lower": (
            regression.slope - t_critical * slope_se
        ),
        "Slope 95% CI upper": (
            regression.slope + t_critical * slope_se
        ),
    }


# ============================================================
# 7. POOLED REGRESSION BY STATION AND PHASE
# ============================================================

pooled_results = []

for station in STATIONS:

    for phase in PHASES:

        subset = analysis_data.loc[
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

if not analysis_data.empty:

    for (station, snow_year, phase), group in analysis_data.groupby(
        ["Station", "Snow year", "Phase"],
        sort=True,
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

seasonal_results = pd.DataFrame(
    seasonal_results,
    columns=[
        "Station",
        "Snow year",
        "Phase",
        "n",
        "Slope",
        "Intercept",
        "R²",
        "Pearson r",
        "p-value",
        "Slope standard error",
        "Slope 95% CI lower",
        "Slope 95% CI upper",
    ],
)

print("\nRegression results by snow year:")

if seasonal_results.empty:
    print("No seasonal regression groups are available.")
else:
    print(seasonal_results.to_string(index=False))


# ============================================================
# 9. PLOT SH-SWE REGRESSIONS
# ============================================================

fig, axes = plt.subplots(
    nrows=len(STATIONS),
    ncols=2,
    figsize=(14, 5.5 * len(STATIONS)),
    squeeze=False,
)

for row_idx, station in enumerate(STATIONS):

    station_color = STATIONS[station]["color"]

    for col_idx, phase in enumerate(PHASES):

        ax = axes[row_idx, col_idx]

        subset = analysis_data.loc[
            (analysis_data["Station"] == station)
            & (analysis_data["Phase"] == phase)
        ]

        # Scatter plot of daily mean values
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

            y_line = (
                stats["Intercept"]
                + stats["Slope"] * x_line
            )

            ax.plot(
                x_line,
                y_line,
                color="black",
                linewidth=2,
                label="OLS regression",
            )

            annotation = (
                f"n = {stats['n']}\n"
                f"Slope = {stats['Slope']:.3f}\n"
                f"R² = {stats['R²']:.3f}\n"
                f"p = {stats['p-value']:.3g}"
            )

        else:
            annotation = (
                f"n = {stats['n']}\n"
                "Insufficient variation"
            )

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
                "alpha": 0.85,
                "edgecolor": "lightgray",
            },
        )

        ax.set_title(f"{station} – {phase}")
        ax.set_xlabel("Snow depth SH (mm)")
        ax.set_ylabel("Snow water equivalent SWE (mm)")
        ax.grid(True, alpha=0.25)

fig.suptitle(
    "Snow depth (SH) versus snow water equivalent (SWE)\n"
    "Daily means; complete snow years; positive values only",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

regression_plot_path = (
    OUTPUT_DIR
    / "SH_SWE_regression_accumulation_melting_all_data.pdf"
)

fig.savefig(
    regression_plot_path,
    dpi=300,
    bbox_inches="tight",
)

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

    station_maxima = seasonal_maxima.loc[
        seasonal_maxima["Station"] == station
    ].sort_values("SWE maximum date")

    if station_maxima.empty:
        ax.set_title(
            f"{station}: no seasonal SWE maxima available"
        )
        continue

    dates = station_maxima["SWE maximum date"]
    swe_maxima = station_maxima["SWE maximum (mm)"]
    sh_at_max = station_maxima["SH at SWE maximum (mm)"]

    # Main axis: seasonal SWE maxima
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

    # Secondary axis: SH on the date of the SWE maximum
    ax_sh.plot(
        dates,
        sh_at_max,
        marker=None,
        linestyle=":",
        linewidth=2.2,
        color="black",
        label="SH at SWE maximum date",
    )

    # Mark maxima from seasons that do not meet the
    # completeness criterion
    incomplete = ~station_maxima["Season complete"]

    if incomplete.any():

        incomplete_dates = dates.loc[incomplete]
        incomplete_swe = swe_maxima.loc[incomplete]

        ax.scatter(
            incomplete_dates,
            incomplete_swe,
            marker="x",
            s=90,
            color="red",
            zorder=5,
            label="Maximum in incomplete season",
        )

    ax.set_title(
        f"{station}: seasonal SWE maxima and snow depth"
    )

    ax.set_xlabel("Date of seasonal SWE maximum")
    ax.set_ylabel("Seasonal maximum SWE (mm)")
    ax_sh.set_ylabel(
        "Snow depth SH at SWE maximum date (mm)"
    )

    ax.grid(True, alpha=0.25)

    # Combine legends from both y-axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_sh.get_legend_handles_labels()

    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=9,
    )

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

fig.suptitle(
    "Seasonal SWE maxima and snow depth at the SWE maximum date\n"
    "Daily mean SWE; July–June snow years",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

maxima_plot_path = (
    OUTPUT_DIR
    / "SH_SWE_seasonal_maxima_all_data.pdf"
)

fig.savefig(
    maxima_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 11. EXPORT RESULTS
# ============================================================

seasonal_maxima_path = (
    OUTPUT_DIR / "seasonal_SWE_maxima_all_data.csv"
)

pooled_results_path = (
    OUTPUT_DIR / "SH_SWE_pooled_regression_all_data.csv"
)

seasonal_results_path = (
    OUTPUT_DIR / "SH_SWE_regression_by_snow_year_all_data.csv"
)

daily_data_path = (
    OUTPUT_DIR / "SH_SWE_daily_data_with_phase_all_data.csv"
)

analysis_data_path = (
    OUTPUT_DIR / "SH_SWE_regression_input_data_all_data.csv"
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

analysis_data.to_csv(
    analysis_data_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nFigures:")
print("Regression figure:", regression_plot_path)
print("Seasonal maxima figure:", maxima_plot_path)

print("\nExported tables:")
print("Seasonal maxima:", seasonal_maxima_path)
print("Pooled regressions:", pooled_results_path)
print("Snow-year regressions:", seasonal_results_path)
print("Daily data:", daily_data_path)
print("Regression input data:", analysis_data_path)

print("\nOutput directory:", OUTPUT_DIR)
