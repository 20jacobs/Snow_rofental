
#!/usr/bin/env python3
# =============================================================================
# Windverfrachtung: SWE am exponierten und geschützten Standort, Bella Vista
# Daten: Universität Innsbruck (2025), v2.0
# https://doi.org/10.5880/fidgeo.2023.037
#
# Required packages:
#   pip install pandas numpy matplotlib
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---- 0. SETTINGS: CHANGE ONLY THIS PATH -------------------------------
# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files remain in the repository directory
DATA_DIR = BASE_DIR / "input"

# Create an output folder for all generated files
# Directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files are stored in the input folder
DATA_DIR = BASE_DIR / "input"

# Save generated figures in the output folder
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_FILE = DATA_DIR / "1_BellaVista_10min_2015-2025.csv"

END_DATE = pd.Timestamp("2025-09-01")
START_DATE = pd.Timestamp("2019-09-01")

MIN_VALID_FRACTION = 0.80
EXPECTED_VALUES_PER_DAY = 144
MIN_VALID_DAYS = 240

EXPOSED = "Bella Vista (exponiert)"
SHELTERED = "Bella Vista (geschützt)"

SITE_LABELS = {
    "exponiert": EXPOSED,
    "geschuetzt": SHELTERED,
}

COLORS = {
    EXPOSED: "#D55E00",   # orange: exposed snow pillow
    SHELTERED: "#0072B2", # blue: protected snow scale
}

SOURCE = (
    "Daten: Universität Innsbruck (2025), v2.0, "
    "doi:10.5880/fidgeo.2023.037"
)


# ---- 1. LOAD DATA, QUALITY CONTROL, DAILY MEANS -----------------------

def load_and_process_data():
    columns = [
        "Date and time",
        "Snow pillow snow water equivalent (mm)",
        "SSG-2 snow water equivalent II (mm)",
    ]

    data = pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        dtype={"Date and time": "string"},
        low_memory=False,
    )

    data = data.rename(
        columns={
            "Date and time": "zeit",
            "Snow pillow snow water equivalent (mm)": "exponiert",
            "SSG-2 snow water equivalent II (mm)": "geschuetzt",
        }
    )

    # Extract date only; handles timestamps ending in .000.
    data["tag"] = pd.to_datetime(
        data["zeit"].str.slice(0, 10),
        errors="coerce",
    ).dt.normalize()

    data = data.dropna(subset=["tag"]).drop(columns=["zeit"])

    value_columns = ["exponiert", "geschuetzt"]

    # Convert to numeric and set negative SWE values to zero.
    for column in value_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).clip(lower=0)

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
            daily_counts[column] < required_count, column
        ] = np.nan

    # Convert to long format.
    daily_long = (
        daily_means.rename_axis("tag")
        .reset_index()
        .melt(
            id_vars="tag",
            value_vars=value_columns,
            var_name="standort",
            value_name="swe",
        )
    )

    # Restrict analysis period.
    daily_long = daily_long.loc[
        (daily_long["tag"] >= START_DATE)
        & (daily_long["tag"] < END_DATE)
    ].copy()

    daily_long["ort"] = daily_long["standort"].map(SITE_LABELS)

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
            n_tage=("swe", "count"),
            maximum=("swe", "max"),
        )
        .reset_index()
    )

    # Only calculate seasonal maxima when >= 240 valid daily means exist.
    seasonal.loc[
        seasonal["n_tage"] < MIN_VALID_DAYS, "maximum"
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

    # Protected maximum SWE divided by exposed maximum SWE.
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
        figsize=(12.5, 2.65 * nrows),
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
                site_data["swe"],
                color=COLORS[site],
                linewidth=1.1,
                alpha=0.95,
                label=site,
                solid_capstyle="round",
            )

        # Add the seasonal maximum ratio.
        ratio_values = ratios.loc[
            ratios["saison"] == season, "verhaeltnis"
        ]

        ratio = (
            ratio_values.iloc[0]
            if not ratio_values.empty
            else np.nan
        )

        if pd.notna(ratio):
            ratio_text = (
                f"Max. SWE ratio (protected/exposed): {ratio:.1f}"
            )
        else:
            ratio_text = "Max. SWE ratio: n/a"

        ax.text(
            0.035,
            0.94,
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
        axes[row, 0].set_ylabel("SWE (mm)")

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
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.2,
    )

    fig.suptitle(
        "Wind redistribution of snow water equivalent",
        x=0.5,
        y=0.995,
        fontweight="bold",
    )

    # Source and measurement-method note.
    fig.text(
        0.5,
        0.012,
        (
            "Exposed: snow pillow; protected: snow scale. "
            + SOURCE
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#5F686E",
    )

    fig.tight_layout(
        rect=(0.025, 0.055, 0.99, 0.925),
        h_pad=1.35,
        w_pad=1.1,
    )

    output_file = (
        OUTPUT_DIR / "abb3_exponiert_geschuetzt.png"
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

    print("\nSeasonal maximum SWE and protected/exposed ratio:")

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
