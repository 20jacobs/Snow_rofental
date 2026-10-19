
# =============================================================================
# Abb. 1: Schneehöhe und SWE im oberen Rofental 2019-2025
# Daten: Univ. Innsbruck (2025), v2.0,
# https://doi.org/10.5880/fidgeo.2023.037
# =============================================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---- 0. Einstellungen: NUR diesen Pfad anpassen -----------------------------

# Ordner, in dem die drei CSV-Dateien direkt liegen
BASE_DIR = Path(__file__).resolve().parent

# Input CSV files remain in the repository directory
DATA_DIR = BASE_DIR / "input"

# Create an output folder for all generated files
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date and time"



ende = pd.Timestamp("2025-09-01")  # Daten enden September 2025
min_anteil = 0.8                    # mindestens 80 % gültige 10-min-Werte
messungen_pro_tag = 144             # 24 Stunden * 6 Messungen pro Stunde

start = pd.Timestamp("2019-09-01")


# ---- 1. Einlesen, QC, Tagesmittel -------------------------------------------

def lies_station(datei, spalten, ort):
    """
    Liest die ausgewählten Sensoren einer Station ein,
    führt die Qualitätskontrolle durch und berechnet Tagesmittel.

    Parameter
    ---------
    datei : str
        Dateiname der CSV-Datei.
    spalten : dict
        Zuordnung von HS/SWE zu den Original-Spaltennamen.
    ort : str
        Name der Station bzw. des Messstandorts.

    Rückgabe
    --------
    pandas.DataFrame mit tag, groesse, wert und ort.
    """

    pfad = DATA_DIR / datei

    # CSV einlesen: Datumsspalte als Text behandeln
    d = pd.read_csv(
        pfad,
        usecols=["Date and time", *spalten.values()],
        dtype={"Date and time": "string"},
        sep=None,
        engine="python"
    )

    # Spaltennamen vereinheitlichen
    d = d.rename(
        columns={
            "Date and time": "zeit",
            **{original: kurz for kurz, original in spalten.items()}
        }
    )

    # Nur das Datum verwenden; funktioniert auch mit .000-Zeitstempeln
    d["tag"] = pd.to_datetime(
        d["zeit"].str[:10],
        format="%Y-%m-%d",
        errors="coerce"
    )

    d = d.dropna(subset=["tag"]).drop(columns="zeit")

    # Messwerte numerisch einlesen
    for col in spalten:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Qualitätskontrolle Schneehöhe (HS)
    if "HS" in d.columns:
        d.loc[d["HS"] < 0, "HS"] = 0
        d.loc[d["HS"] > 3000, "HS"] = float("nan")

        # Umrechnung von mm in cm
        d["HS"] = d["HS"] / 10

    # Qualitätskontrolle Schneewasseräquivalent (SWE)
    # SWE bleibt in mm
    if "SWE" in d.columns:
        d.loc[d["SWE"] < 0, "SWE"] = 0

    # Ins Long-Format umwandeln
    lang = d.melt(
        id_vars="tag",
        value_vars=list(spalten.keys()),
        var_name="groesse",
        value_name="wert"
    )

    # Tagesmittel nur bei ausreichend gültigen Messwerten
    tagesmittel = (
        lang.groupby(["tag", "groesse"])["wert"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Entspricht der R-Bedingung:
    # sum(!is.na(wert)) >= min_anteil * 144
    ausreichend = (
        tagesmittel["count"]
        >= min_anteil * messungen_pro_tag
    )

    tagesmittel["wert"] = tagesmittel["mean"].where(ausreichend)

    tagesmittel = tagesmittel[["tag", "groesse", "wert"]]
    tagesmittel["ort"] = ort

    return tagesmittel


# Daten aller Stationen einlesen
ts = pd.concat(
    [
        lies_station(
            "1_BellaVista_10min_2015-2025.csv",
            {
                "HS": "USH8 snow depth (mm)",
                "SWE": "Snow pillow snow water equivalent (mm)"
            },
            "Bella Vista (exponiert)"
        ),

        lies_station(
            "1_BellaVista_10min_2015-2025.csv",
            {
                "HS": "USH9 snow depth II (mm)",
                "SWE": "SSG-2 snow water equivalent II (mm)"
            },
            "Bella Vista (geschützt)"
        ),

        lies_station(
            "2_Latschbloder_10min_2013-2025.csv",
            {
                "HS": "USH8 snow depth (mm)"
            },
            "Latschbloder"
        ),

        lies_station(
            "3_Proviantdepot_10min_2019-2025.csv",
            {
                "HS": "USH9 snow depth (mm)",
                "SWE": "SSG-2 snow water equivalent (mm)"
            },
            "Proviantdepot"
        )
    ],
    ignore_index=True
)


# Zeitraum eingrenzen und Variablen vorbereiten
orte = [
    "Bella Vista (exponiert)",
    "Bella Vista (geschützt)",
    "Latschbloder",
    "Proviantdepot"
]

ts = ts.loc[
    (ts["tag"] >= start) & (ts["tag"] < ende)
].copy()

ts["ort"] = pd.Categorical(
    ts["ort"],
    categories=orte,
    ordered=True
)

# Saison beginnt jeweils am 1. September
ts["saison_start"] = (
    ts["tag"].dt.year
    - (ts["tag"].dt.month < 9).astype(int)
)

# Deutsche Bezeichnungen für die beiden Messgrößen
groessen = {
    "HS": "Schneehöhe (cm)",
    "SWE": "Schneewasseräquivalent (mm)"
}

ts["groesse_lab"] = ts["groesse"].map(groessen)


# Datenlücke:
# Proviantdepot Schneehöhe 2023/24 (1.1.-17.7.2024)
# Diese Saison für HS nicht darstellen
luecke = (
    (ts["ort"] == "Proviantdepot")
    & (ts["groesse"] == "HS")
    & (ts["saison_start"] == 2023)
)

ts.loc[luecke, "wert"] = float("nan")


# ---- 2. Abb. 1: Zeitreihen aller Stationen ----------------------------------

farben = {
    "Bella Vista (exponiert)": "#D55E00",
    "Bella Vista (geschützt)": "#0072B2",
    "Latschbloder": "#009E73",
    "Proviantdepot": "#CC79A7"
}

quelle = (
    "Daten: Universität Innsbruck (2025), v2.0, "
    "doi:10.5880/fidgeo.2023.037"
)

# Plot-Stil
plt.rcParams.update({
    "font.size": 13,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.7
})

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 6.5),
    sharex=True
)

# Beide Messgrößen getrennt darstellen (freie y-Achsen)
for ax, groesse, titel in zip(
    axes,
    ["HS", "SWE"],
    ["Schneehöhe (cm)", "Schneewasseräquivalent (mm)"]
):
    daten_groesse = ts[ts["groesse"] == groesse]

    for ort in orte:
        daten_ort = (
            daten_groesse[daten_groesse["ort"] == ort]
            .sort_values("tag")
        )

        ax.plot(
            daten_ort["tag"],
            daten_ort["wert"],
            color=farben[ort],
            linewidth=0.5,
            label=ort
        )

    ax.set_ylabel(titel)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)

# Jahreszahlen auf der x-Achse
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[-1].set_xlim(start, ende)
axes[-1].set_xlabel("")

# Legende unterhalb der beiden Diagramme
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=2,
    frameon=False
)

fig.suptitle(
    "Schneehöhe und SWE im oberen Rofental 2019-2025",
    y=0.97
)

fig.text(
    0.01,
    0.015,
    "Tagesmittel aus 10-min-Daten. " + quelle,
    ha="left",
    va="bottom",
    fontsize=9
)

# Layout: Platz für Titel, Legende und Quellenangabe
fig.subplots_adjust(
    left=0.11,
    right=0.98,
    top=0.90,
    bottom=0.23,
    hspace=0.12
)

# Abbildung speichern
ausgabe = OUTPUT_DIR / "abb1_zeitreihen.png"

fig.savefig(
    ausgabe,
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Abbildung gespeichert unter:\n{ausgabe}")

# Abbildung anzeigen
plt.show()
