# =============================================================================
# Schneehöhe und SWE an den Stationen (R-Version des Python-Codes)
# einmalig: install.packages(c("data.table", "ggplot2", "patchwork"))
# Benötigt ggplot2 >= 3.5.0 (prüfen: packageVersion("ggplot2"))
# =============================================================================
library(data.table); library(ggplot2); library(patchwork)

dir_daten <- "C:/Users/JKnap/Documents/Master/EU Gebirgsforschung/Daten"
dir_abb   <- file.path(dirname(dir_daten), "Abbildungen")
dir.create(dir_abb, showWarnings = FALSE, recursive = TRUE)

START    <- as.POSIXct("2017-09-01", tz = "UTC")
ENDE     <- as.POSIXct("2025-10-01", tz = "UTC")
MAX_GAP  <- 20 * 60      # Sekunden: Linie bei Lücken > 20 min unterbrechen
MM_TO_CM <- 0.1

# ---- 1. Stationen -----------------------------------------------------------
bv <- "1_BellaVista_10min_2015-2025.csv"
lb <- "2_Latschbloder_10min_2013-2025.csv"
pd <- "3_Proviantdepot_10min_2019-2025.csv"

SNOW <- list(
  list(st = "Bella Vista (protected)", f = bv, col = "USH9 snow depth II (mm)"),
  list(st = "Bella Vista (exposed)",   f = bv, col = "USH8 snow depth (mm)"),
  list(st = "Latschbloder",            f = lb, col = "USH8 snow depth (mm)"),
  list(st = "Proviantdepot",           f = pd, col = "USH9 snow depth (mm)"))
SWE <- list(
  list(st = "Bella Vista (protected)", f = bv, col = "SSG-2 snow water equivalent II (mm)"),
  list(st = "Bella Vista (exposed)",   f = bv, col = "Snow pillow snow water equivalent (mm)"),
  list(st = "Proviantdepot",           f = pd, col = "SSG-2 snow water equivalent (mm)"))

farben <- c("Bella Vista (protected)" = "#0072B2", "Bella Vista (exposed)" = "#D55E00",
            "Latschbloder" = "darkorange", "Proviantdepot" = "forestgreen")

# ---- 2. Einlesen und Bereinigen ---------------------------------------------
lies <- function(s, panel) {
  d <- fread(file.path(dir_daten, s$f), select = c("Date and time", s$col),
             colClasses = list(character = "Date and time"))
  setcolorder(d, c("Date and time", s$col))      # feste Reihenfolge
  setnames(d, c("zeit", "wert"))
  d[, zeit := as.POSIXct(substr(trimws(zeit), 1, 19),
                         format = "%Y-%m-%d %H:%M:%S", tz = "UTC")]
  d <- d[!is.na(zeit)]
  d[, wert := as.numeric(wert)]
  d[wert < 0, wert := NA_real_]                  # negative Werte -> NA (wie Python)
  d <- d[, .(wert = mean(wert, na.rm = TRUE)), by = zeit]   # doppelte Zeitstempel
  d[is.nan(wert), wert := NA_real_]
  setorder(d, zeit)
  d <- d[zeit >= START]
  d[, cm := wert * MM_TO_CM]
  d[, plot_cm := cm]
  d[c(FALSE, diff(as.numeric(zeit)) > MAX_GAP), plot_cm := NA_real_]  # Lücken
  d[, c("station", "panel") := .(s$st, panel)]
  d[, .(zeit, cm, plot_cm, station, panel)]
}

P1 <- "Snow Depth"; P2 <- "Snow Water Equivalent (SWE)"
alle <- rbindlist(c(lapply(SNOW, lies, panel = P1), lapply(SWE, lies, panel = P2)))
alle[, station := factor(station, levels = names(farben))]

# ---- 3. Deskriptive Statistik (Text pro Panel) ------------------------------
st <- alle[, .(mean = mean(cm, na.rm = TRUE), max = max(cm, na.rm = TRUE),
               valid = sum(!is.na(cm)), missing = sum(is.na(cm))),
           by = .(panel, station)]
setorder(st, panel, station)
box <- st[, .(label = paste0("DESCRIPTIVE STATISTICS\n\n",
                             paste0(station, "\n  Mean: ", sprintf("%.1f", mean), " cm\n  Maximum: ",
                                    sprintf("%.1f", max), " cm\n  Valid: ", format(valid, big.mark = ","),
                                    "\n  Missing: ", format(missing, big.mark = ","), collapse = "\n\n"))),
          by = panel]

# ---- 4. Abbildung -------------------------------------------------------------
monate <- seq(START, ENDE, by = "month")                                   # Striche (jeder Monat)
jan    <- seq(as.POSIXct("2018-01-01", tz = "UTC"), ENDE, by = "year")     # Gitterlinien

# Beschriftung nur im Januar (mit Jahr), April, Juli, Oktober; sonst leer
lab_monat <- function(x) {
  m <- as.integer(format(x, "%m"))
  ifelse(m == 1, paste0("1\n", format(x, "%Y")),
         ifelse(m %in% c(4, 7, 10), as.character(m), ""))
}

panel_plot <- function(pn, titel, ylab, ymax, mit_xachse) {
  p <- ggplot(alle[panel == pn], aes(zeit, plot_cm, colour = station)) +
    geom_vline(xintercept = jan, colour = "grey88") +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    scale_colour_manual(values = farben, name = "Weather station") +
    scale_x_datetime(breaks = monate, labels = lab_monat,
                     limits = c(START, ENDE), expand = c(0, 0)) +
    coord_cartesian(ylim = c(0, ymax), expand = FALSE) +
    labs(title = titel, x = if (mit_xachse) "Month / year" else NULL, y = ylab) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
          panel.grid.major.x = element_blank(),       # vertikales Gitter: nur Januar
          panel.grid.major.y = element_line(colour = "grey88"),
          panel.grid.minor = element_blank(),
          panel.border = element_rect(colour = "grey70"),
          axis.ticks = element_line(colour = "black"),
          axis.ticks.length = unit(4, "pt"),
          axis.text.x = element_text(size = 10, lineheight = 1.1),
          legend.position = "inside",
          legend.position.inside = c(0.01, 0.98),
          legend.justification = c(0, 1),
          legend.background = element_rect(colour = "grey60", fill = alpha("white", 0.92)),
          legend.title = element_text(size = 9, face = "bold"),
          legend.text = element_text(size = 9))
  if (!mit_xachse) p <- p + theme(axis.text.x = element_blank())  # wie sharex
  p
}

# Statistik-Feld: eigenes Plot-Element rechts neben dem Panel
box_plot <- function(pn) {
  ggplot() +
    annotate("label", x = 0, y = 0.86, label = box[panel == pn, label],
             hjust = 0, vjust = 1, size = 2.8, family = "mono",
             lineheight = 1.1, label.size = 0.3) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_void()
}

abb <- (panel_plot(P1, "Snow Depth", "Snow depth (cm)", 250, FALSE) + box_plot(P1) +
          panel_plot(P2, "Snow Water Equivalent (SWE)", "SWE (cm)", 150, TRUE) + box_plot(P2)) +
  plot_layout(ncol = 2, widths = c(5, 1)) +
  plot_annotation(
    title = "Snow Depth and Snow Water Equivalent at the Weather Stations",
    theme = theme(plot.title = element_text(face = "bold", size = 17, hjust = 0.5)))

# ---- 5. Speichern (mit Ausweichordner, falls der Zielordner gesperrt ist) ----
dateiname <- "snow_depth_and_SWE_combined.png"
ziel <- file.path(dir_abb, dateiname)
ok <- tryCatch({
  ggsave(ziel, abb, width = 20, height = 12, dpi = 300, bg = "white"); TRUE
}, error = function(e) FALSE, warning = function(w) FALSE)

if (!ok) {   # z. B. Datei in anderem Programm geöffnet oder Ordner geschützt
  ziel <- file.path(path.expand("~"), dateiname)   # Dokumente-Ordner
  ggsave(ziel, abb, width = 20, height = 12, dpi = 300, bg = "white")
  message("Zielordner nicht beschreibbar. Gespeichert in: ", ziel)
} else {
  message("Gespeichert in: ", ziel)
}
abb