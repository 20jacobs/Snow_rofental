
#!/usr/bin/env python3
# =============================================================================
# Windverfrachtung: Schneehöhe am exponierten und geschützten Standort,
# Bella Vista
# Daten: Universität Innsbruck (2025), v2.0
# https://doi.org/10.5880/fidgeo.2023.037
#
# Required packages:
#   pip install pandas numpy matplotlib
#
# - Analyses snow depth (SH) instead of snow water equivalent (SWE)
# - Exposed site: USH8 snow depth (mm)
# - Protected site: USH9 snow depth II (mm)
# - Converts snow depth from mm to cm
# - Calculates daily means and seasonal maxima
# - Calculates protected/exposed seasonal maximum ratios
# - Retains the original seasonal panel layout and styling
# - Saves the figure as a PNG file
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---- 0. SETTINGS: CHANGE ONLY THIS PATH -------------------------------

# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files
DATA_DIR = BASE_DIR / "input"

# Output directory
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_FILE = DATA_DIR / "1_BellaVista_10min_2015-2025.csv"

END_DATE = pd.Timestamp("2025-09-01")
START_DATE = pd.Timestamp("2019-09-01")

MIN_VALID_FRACTION = 0.80
EXPECTED_VALUES_PER_DAY = 144
MIN_VALID_DAYS = 20

# Conversion factor: mm -> cm
MM_TO_CM = 0.1

EXPOSED = "Bella Vista (exposed)"
SHELTERED = "Bella Vista (protected)"

SITE_LABELS = {
    "exposed": EXPOSED,
    "protected": SHELTERED,
}

COLORS = {
    EXPOSED: "#D55E00",    # Orange: exposed site
    SHELTERED: "#0072B2",  # Blue: protected site
}

SOURCE = (
    "Daten: Universität Innsbruck (2025), v2.0, "
    "doi:10.5880/fidgeo.2023.037"
)


# ---- 1. LOAD DATA, QUALITY CONTROL, DAILY MEANS -----------------------

def load_and_process_data():

    columns = [
        "Date and time",
        "USH8 snow depth (mm)",
        "USH9 snow depth II (mm)",
    ]

    data = pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        dtype={"Date and time": "string"},
        low_memory=False,
    )

    # Rename measurement columns for consistent processing
    data = data.rename(
        columns={
            "Date and time": "zeit",
            "USH8 snow depth (mm)": "exposed",
            "USH9 snow depth II (mm)": "protected",
        }
    )

    # Extract date only; handles timestamps ending in .000.
    data["tag"] = pd.to_datetime(
        data["zeit"].str.slice(0, 10),
        errors="coerce",
    ).dt.normalize()

    data = data.dropna(
        subset=["tag"]
    ).drop(columns=["zeit"])

    value_columns = ["exposed", "protected"]

    # Convert measurements to numeric values.
    # Exclude negative snow-depth measurements.
    # Convert valid measurements from mm to cm.
    for column in value_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        data.loc[data[column] < 0, column] = np.nan

        data[column] = data[column] * MM_TO_CM

    # Calculate daily means and valid observation counts.
    grouped = data.groupby("tag", sort=True)[value_columns]

    daily_means = grouped.mean()
    daily_counts = grouped.count()

    required_count = (
        MIN_VALID_FRACTION * EXPECTED_VALUES_PER_DAY
    )

    # Set daily means to NaN if fewer than 80% of readings are valid.
    for column in value_columns:

        daily_means.loc[
            daily_counts[column] < required_count,
            column,
        ] = np.nan

    # Convert to long format.
    daily_long = (
        daily_means.rename_axis("tag")
        .reset_index()
        .melt(
            id_vars="tag",
            value_vars=value_columns,
            var_name="standort",
            value_name="snow_depth_cm",
        )
    )

    # Restrict analysis period.
    daily_long = daily_long.loc[
        (daily_long["tag"] >= START_DATE)
        & (daily_long["tag"] < END_DATE)
    ].copy()

    daily_long["ort"] = daily_long["standort"].map(
        SITE_LABELS
    )

    # Hydrological/snow season starts on 1 September.
    daily_long["saison_start"] = (
        daily_long["tag"].dt.year
        - (daily_long["tag"].dt.month < 9).astype(int)
    )

    daily_long["saison"] = daily_long["saison_start"].map(
        lambda year: f"{year}/{str(year + 1)[-2:]}"
    )

    # Map every season onto a common standard year beginning 1 September.
    # The reference year 2000/2001 is a leap year, preserving 29 February.
    season_start_dates = pd.to_datetime(
        daily_long["saison_start"].astype(str) + "-09-01"
    )

    daily_long["datum_std"] = (
        pd.Timestamp("2000-09-01")
        + (daily_long["tag"] - season_start_dates)
    )

    return daily_long


# ---- 2. SEASONAL MAXIMA AND PROTECTED/EXPOSED RATIOS ------------------

def calculate_seasonal_maxima(data):

    seasonal = (
        data.groupby(["ort", "saison"], observed=True)
        .agg(
            n_tage=("snow_depth_cm", "count"),
            maximum=("snow_depth_cm", "max"),
        )
        .reset_index()
    )

    # Only calculate seasonal maxima when >= 240 valid daily means exist.
    seasonal.loc[
        seasonal["n_tage"] < MIN_VALID_DAYS,
        "maximum",
    ] = np.nan

    ratios = (
        seasonal.pivot(
            index="saison",
            columns="ort",
            values="maximum",
        )
        .reindex(columns=[EXPOSED, SHELTERED])
        .reset_index()
    )

    # Protected maximum snow depth divided by exposed maximum snow depth.
    # The ratio is dimensionless.
    ratios["verhaeltnis"] = (
        ratios[SHELTERED] / ratios[EXPOSED]
    )

    return seasonal, ratios


# ---- 3. PUBLICATION-READY PLOT ---------------------------------------

def make_plot(data, ratios):

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#C7CDD1",
        "axes.linewidth": 0.8,
        "xtick.color": "#424A50",
        "ytick.color": "#424A50",
        "text.color": "#20272B",
        "axes.labelcolor": "#20272B",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    seasons = sorted(data["saison"].dropna().unique())

    if not seasons:
        raise ValueError(
            "No data remain in the selected period. "
            "Please check the input file and date range."
        )

    # Three columns of seasonal panels.
    ncols = 3
    nrows = int(np.ceil(len(seasons) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 3.0  * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    # Standard year: 1 September to 1 September.
    x_start = pd.Timestamp("2000-09-01")
    x_end = pd.Timestamp("2001-09-01")

    for ax, season in zip(axes_flat, seasons):

        subset = data.loc[data["saison"] == season]

        # Plot both sites.
        for site in [EXPOSED, SHELTERED]:

            site_data = (
                subset.loc[subset["ort"] == site]
                .sort_values("datum_std")
            )

            ax.plot(
                site_data["datum_std"],
                site_data["snow_depth_cm"],
                color=COLORS[site],
                linewidth=1.1,
                alpha=0.95,
                label=site,
                solid_capstyle="round",
            )

        # Add the seasonal maximum snow-depth ratio.
        ratio_values = ratios.loc[
            ratios["saison"] == season,
            "verhaeltnis",
        ]

        ratio = (
            ratio_values.iloc[0]
            if not ratio_values.empty
            else np.nan
        )

        if pd.notna(ratio):
            ratio_text = (
                "Max. snow depth ratio "
                f"(protected/exposed): {ratio:.2f}"
            )
        else:
            ratio_text = "Max. snow depth ratio: n/a"

        # Move ratio labels closer to the top edge.
        ax.text(
            0.035,
            0.985,
            ratio_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#414A50",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#E2E6E8",
                "alpha": 0.92,
            },
        )

        ax.set_title(
            season,
            loc="left",
            fontweight="semibold",
            pad=8,
        )

        ax.set_xlim(x_start, x_end)

        # Set the common snow-depth range to 0–250 cm.
        ax.set_ylim(0, 250)

        # Month labels every two months.
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=2)
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b")
        )

        # Subtle horizontal gridlines.
        ax.grid(
            axis="y",
            color="#E8ECEF",
            linewidth=0.75,
        )
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

        ax.spines["left"].set_color("#D5DADD")
        ax.spines["bottom"].set_color("#D5DADD")
        ax.tick_params(length=0, pad=5)

    # Hide unused panels.
    for ax in axes_flat[len(seasons):]:
        ax.set_visible(False)

    # Shared axis labels.
    for row in range(nrows):
        axes[row, 0].set_ylabel("Snow depth (cm)")

    for ax in axes_flat[:len(seasons)]:
        ax.set_xlabel("Month")

    # One shared legend above the panels.
    handles = [
        plt.Line2D(
            [0], [0],
            color=COLORS[site],
            linewidth=2.0,
            label=site,
        )
        for site in [EXPOSED, SHELTERED]
    ]

    fig.legend(
    handles=handles,
    labels=[EXPOSED, SHELTERED],
    loc="upper right",
    bbox_to_anchor=(0.975, 0.965),
    ncol=1,
    frameon=False,
    handlelength=2.0,
    labelspacing=0.5,
)

    fig.suptitle(
        "Wind redistribution of snow depth",
        x=0.5,
        y=0.99,
        fontweight="bold",
    )

    # Source and measurement-method note.
    fig.text(
        0.5,
        0.018,
        (
            "Exposed: USH8 snow depth; "
            "protected: USH9 snow depth II. "
            + SOURCE
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#5F686E",
    )

    # Manually control panel placement and spacing.
    # This replaces tight_layout(), which would override these settings.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.87,
        hspace=0.22,
        wspace=0.18,
    )

    output_file = (
        OUTPUT_DIR / "abb3_schneehoehe_exponiert_geschuetzt.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file


# ---- 4. RUN ANALYSIS -------------------------------------------------

def main():

    data = load_and_process_data()

    _, ratios = calculate_seasonal_maxima(data)

    output_file = make_plot(data, ratios)

    print(f"Plot saved to: {output_file}")

    print("\nSeasonal maximum snow depth and protected/exposed ratio:")

    print(
        ratios[
            ["saison", EXPOSED, SHELTERED, "verhaeltnis"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
