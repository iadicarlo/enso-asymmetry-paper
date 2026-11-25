"""
Shared configuration for the MMLEAv2 ENSO-future pipeline.

All scripts import from here so that paths, variable lists, lat/lon boxes,
and the realization-ID regex are defined exactly once.
"""

import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_DIR       = Path("/path/to/enso_impacts")
TROPICS_DIR    = REPO_DIR / "data" / "mmleav2_tropics_output"
GWI_CACHE_DIR  = REPO_DIR / "data" / "gwi_cache"
RESULTS_DIR    = REPO_DIR / "results" / "enso_metrics"
LOGS_DIR       = REPO_DIR / "logs"

for _d in (TROPICS_DIR, GWI_CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Canonical spatial box (equatorial Pacific) ────────────────────────────────
# Change here to affect all scripts: pipeline download, GWI preprocess,
# enso_metrics glob pattern, and run_enso_metrics.sh defaults.

LAT_MIN: float = -20.0
LAT_MAX: float =  20.0
LON_MIN: float = 110.0
LON_MAX: float = 285.0
# ── Widened from ±6° on 2026-04-21 to support Izumo & Colin 2022 / Izumo
# 2024 JCLI recharge index h_ind_eq+sw, whose southwest Pacific box spans
# 15°S–5°S, and to leave headroom for NPMM-style (10°–20°N) off-equatorial
# indices used in Chen 2023 (npj CAS).  Obs preprocess and Casper pipeline
# both read these constants; downstream metrics scripts reduce to the ±5°
# equatorial band where needed (e.g. Niño3.4, SST gradient boxes).

# ── Models ────────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "access_lens",
    "canesm2_lens",
    "canesm5_lens",
    "cesm1_lens",
    "cesm2_lens",
    "csiro_mk36_lens",
    "e3sm_v1_lens",
    "e3sm_v2_lens",
    "ec-earth3_lens",
    "gfdl_cm3_lens",
    "gfdl_esm2m_lens",
    "gfdl_spear_lens",
    "giss-e2",
    "ipsl_cm6a_lens",
    "miroc6_lens",
    "miroc_esm2l_lens",
    "mpi_lens",
    "mpi_lens_cmip6",
    "ukesm1-0-ll_lens",
]

# ── Variable → THREDDS frequency folder ──────────────────────────────────────
# Add new variables here. The pipeline, enso_metrics, and run scripts will
# pick them up automatically without any other changes.

VAR_FREQ_MAP: dict[str, str] = {
    # atmosphere monthly
    "pr":       "Amon",
    "tas":      "Amon",
    "psl":      "Amon",
    "ts":       "Amon",
    "tauu":     "Amon",
    "tauv":     "Amon",
    "zg500":    "Amon",
    "evspsbl":  "Amon",
    # ocean monthly
    "tos":      "Omon",
    "sos":      "Omon",   # sea-surface salinity
    "mlotst":   "Omon",   # mixed-layer depth
    "zos":      "Omon",   # sea-surface height
    "z20":      "Omon",   # 20°C isotherm depth
    # sea ice monthly
    "siconc":   "OImon",
}

ALL_VARS = list(VAR_FREQ_MAP.keys())

# ── Time bounds ───────────────────────────────────────────────────────────────
# Standard time range covering all MMLEAv2 data (historical + future).
TIME_START = "1850-01-01"
TIME_END   = "2100-12-31"

# ── THREDDS / RDA ─────────────────────────────────────────────────────────────
# RDA dataset ID and base URLs. Update here if the archive moves.
RDA_DATASET_ID       = "d651039"
THREDDS_CATALOG_BASE = f"https://thredds.rda.ucar.edu/thredds/catalog/files/{RDA_DATASET_ID}"
THREDDS_DODS_BASE    = f"https://thredds.rda.ucar.edu/thredds/dodsC/files/{RDA_DATASET_ID}"

# ── Realization ID extraction ─────────────────────────────────────────────────
# Shared by pipeline, preprocess_gwi, and enso_metrics.
# Handles three naming conventions found in MMLEAv2:
#   1. CESM2-LE non-standard : 1001.001i1p1f1
#   2. CMIP6                 : r10i1p1f1
#   3. CMIP5                 : r10i1p1

def extract_realization(filename: str) -> str:
    """Return the bare realization ID from an MMLEAv2 filename.

    Priority:
      1. CESM2-LE: \d{4}.\d{3}i\d+p\d+f\d+  (e.g. 1001.001i1p1f1)
      2. CMIP6   : r\d+i\d+p\d+f\d+           (e.g. r10i1p1f1)
      3. CMIP5   : r\d+i\d+p\d+               (e.g. r10i1p1)
    Falls back to the stemmed filename only as a last resort (will log a
    warning upstream if this happens, since it will break GWI matching).
    """
    # CESM2-LE non-standard (before CMIP6 to avoid partial r-match).
    # Some alias files prepend an "r" (pr_CESM2_cmip6_hist_ssp370_r1001.001i1p1f1_…)
    # while the canonical form omits it (pr_Amon_CESM2_..._1001.001i1p1f1_g025…)
    # → normalise both to the bare "NNNN.NNNiXpYfZ" token.
    m = re.search(r'_r?(\d{4}\.\d{3}i\d+p\d+f\d+)(?:_|\.)', filename)
    if m:
        return m.group(1)
    # CMIP6: r\d+i\d+p\d+f\d+
    m = re.search(r'_(r\d+i\d+p\d+f\d+)(?:_|\.)', filename)
    if m:
        return m.group(1)
    # CMIP5: r\d+i\d+p\d+ (no f suffix)
    m = re.search(r'_(r\d+i\d+p\d+)(?:_|\.)', filename)
    if m:
        return m.group(1)
    return filename.replace('.nc', '')


def extract_scenario(filename: str) -> str:
    """Return the future scenario label from an MMLEAv2 filename.

    Returns 'historical' for historical-only files, or the SSP/RCP token
    (e.g. 'ssp370', 'ssp585', 'rcp85') for combined future files.
    """
    tokens = re.findall(r'(ssp\d+|rcp\d+)', filename.lower())
    return tokens[-1] if tokens else "historical"
