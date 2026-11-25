"""NOAA CPC's episode rule on the relative Nino index, in standardised units.

Vendored on 2026-09-14 from the ENSOscope platform (scripts/analysis/functions.py, the block from
ENSO_DETREND onwards) so that the paper classifies events exactly as the platform does. The method is
written out in docs/07_enso_classification_change_2026-09-14.md and the companion method note. Only the
imports and two paths are adapted; the logic is the shipped code.
"""

import json as _json
import os as _os
import types as _types
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import regionmask  # only needed by ocean_only(), which masks land on a model grid
except ImportError:  # pragma: no cover
    regionmask = None

from .events import detect_enso_events, remove_seasonal_cycle_rolling_with_std  # noqa: F401

# L'Heureux et al. 2024 (J. Climate 37, 1197) prefer the relative index because
# a fitted curve cannot be reliably extrapolated in real time.
#
#   ENSO_DETREND=roni  (default)  Nino box minus the CONCURRENT 20S-20N mean
#   ENSO_DETREND=gwi              the previous method, kept for reproducibility
#
# The subtraction is same-month, no window. Each region's seasonal cycle is
# removed by the caller's deseasonalisation of the DIFFERENCE, which equals the
# difference of the deseasonalised series, so nothing else in the pipeline
# changes. The paper's scaling ratio is omitted on purpose: every threshold
# here is in standard deviations of the index itself, so the ratio cancels.

ENSO_DETREND = _os.environ.get("ENSO_DETREND", "roni").lower()
TROPICAL_BOX = ((-20.0, 20.0), (0.0, 360.0))


def _sel_box(da, lat_bounds, lon_bounds):
    """Lat/lon box that tolerates a descending latitude axis (COBE, HadISST)."""
    lat0, lat1 = lat_bounds
    lat = da["lat"].values
    lat_slice = slice(lat1, lat0) if lat[0] > lat[-1] else slice(lat0, lat1)
    return da.sel(lat=lat_slice, lon=slice(*lon_bounds))


def area_mean(da, lat_bounds, lon_bounds):
    """Cosine-weighted box mean over (lat, lon)."""
    sub = _sel_box(da, lat_bounds, lon_bounds)
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(dim=["lat", "lon"])


def relative_box_index(sst, lat_bounds, lon_bounds, trop=TROPICAL_BOX):
    """Nino-box SST minus the concurrent tropical-mean SST, same grid, same month.

    Feed the result to remove_seasonal_cycle_rolling_with_std exactly as the
    detrended box mean was fed before. Works for observations and for every
    model member alike: a model's tropical warm bias cancels in the subtraction.
    """
    return area_mean(sst, lat_bounds, lon_bounds) - area_mean(sst, *trop)


_OCEAN_MASK_CACHE = {}


def ocean_only(da):
    """Blank out land cells of a (time, lat, lon) field before relative_box_index.

    Needed for climate-model fields, not for the observed SST products. A model
    surface temperature such as CESM2's TS is the SST over ocean but the LAND
    SKIN temperature over land, which warms faster and swings much harder
    through the seasonal cycle. The three Nino boxes are entirely ocean, so this
    mask never touches them; the tropical 20S-20N mean is about a quarter land
    (Africa, South America, the Maritime Continent, India, Australia), and
    leaving that in would put a land-surface signal into the very reference the
    Nino boxes are measured against. COBE, ERSSTv5, HadISST and OISSTv2 are
    ocean-only by construction, so masking here is also what makes the model and
    observed relative indices the same quantity.

    The mask is cached per grid, so calling this once per ensemble member is
    cheap.
    """
    key = (float(da["lat"][0]), float(da["lat"][-1]), da["lat"].size,
           float(da["lon"][0]), float(da["lon"][-1]), da["lon"].size)
    m = _OCEAN_MASK_CACHE.get(key)
    if m is None:
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        m = np.isnan(land.mask(da["lon"], da["lat"]).values)
        _OCEAN_MASK_CACHE[key] = m
    return da.where(xr.DataArray(m, dims=("lat", "lon"),
                                 coords={"lat": da["lat"], "lon": da["lon"]}))


# ============================================================================
# ENSO event detection rule: the second switch (2026-09-14)
# ============================================================================
# Audit docs/audit_roni_vs_gwi_2026-09-14.md, Results 7, 8, 10 and 11.
#
#   ENSO_DETECTOR=noaa   (default)  NOAA CPC's episode rule written in
#                                   standardised units, plus multi-year
#                                   linking and cumulative intensity tiers
#   ENSO_DETECTOR=legacy            detect_enso_events + fill_small_enso_gaps
#                                   + merge_events_by_gap, exactly as shipped
#
# The legacy path is kept byte for byte because two papers were written on it.
# Nothing below touches detect_enso_events, fill_small_enso_gaps or the nested
# check_seasonal_phase_locking.
#
# THE RULE
#   1. Index. rel3(t) is the centred 3-month running mean of the deseasonalised
#      relative Nino-3.4 anomaly in degC, built by the platform's own rolling
#      30-year climatology. z3(t) = rel3(t) / sigma_rel3(m), where sigma_rel3(m)
#      is the per-calendar-month standard deviation of rel3 over the
#      standardisation baseline (1980-2014).
#   2. Threshold. t(m) = 0.5 / sigma_ONI3(m), where sigma_ONI3(m) is the
#      per-calendar-month standard deviation, over the same baseline, of the
#      3-month mean of the RAW (non-relative) deseasonalised Nino-3.4 anomaly
#      on COBE. This is algebraically NOAA's own variance adjustment: their
#      test RONI_adj = rel3 * sigma_ONI3 / sigma_rel3 >= 0.5 degC is exactly
#      z3 >= t(m). Nothing is fitted. The vector is computed once on COBE by
#      scripts/analysis/build_nino34_threshold_vector.py and shared by every
#      dataset and model, so a product with weaker variance does not silently
#      get a looser test.
#   3. Episode. At least five consecutive overlapping seasons with z3 >= t(m)
#      (El Nino) or z3 <= -t(m) (La Nina). No phase lock, no gap bridging.
#      Onset is the centre month of the first season, end the centre month of
#      the last, which is how CPC labels its seasonal table.
#   4. Linking. Two same-sign episodes are one multi-year event when the gap is
#      at most three months and no gap month reaches the opposite side of
#      +/-0.25 sd. The event keeps its sub-episodes, which are NOAA's own inner
#      boundaries.
#   5. Intensity. Read on the MONTHLY standardised Nino-3 (El Nino) or Nino-4
#      (La Nina) index inside the event window, with the shipped three-month
#      persistence, but as cumulative tiers rather than disjoint bands: extreme
#      at or beyond 1.75 sd (El Nino also needs Nino-3 precipitation at or above
#      5 mm/day for three consecutive months), else strong at or beyond 1.25 sd
#      with no upper bound, else moderate. The highest tier satisfied wins, so a
#      peak excursion can no longer break the strong run and drop an event to
#      moderate (W1e, Result 10).
#
# A run that is still going at the end of the record is kept, not discarded:
# NOAA's own table carries episodes in progress, and this platform's whole
# point is to say what phase we are in now.

ENSO_DETECTOR = _os.environ.get("ENSO_DETECTOR", "noaa").lower()

NOAA_ONI_THRESHOLD_DEGC = 0.5
NOAA_MIN_SEASONS = 5
NOAA_LINK_GAP_MONTHS = 3
NOAA_LINK_QUIET_SD = 0.25
DEFAULT_BASELINE_PERIOD = (1980, 2014)
DEFAULT_THRESHOLD_VECTOR_FILE = (
    _Path(__file__).resolve().parents[2]
    / "data" / "reference" / "nino34_threshold_vector_cobe_1980_2014.json"
)

_THRESHOLD_VECTOR_CACHE = {}


def centred_running_mean3(values):
    """Centred 3-month running mean; NaN wherever the full window is missing.

    A season is labelled by its centre month, so element i of the result is the
    season (i-1, i, i+1) and the first and last elements are always NaN. That
    is why detection on a record ending in month M stops at M-1.
    """
    values = np.asarray(values, dtype="float64")
    out = np.full(values.shape, np.nan, dtype="float64")
    for i in range(1, len(values) - 1):
        window = values[i - 1:i + 2]
        if np.all(np.isfinite(window)):
            out[i] = window.mean()
    return out


def per_calendar_month_sigma(values, time, baseline_period=DEFAULT_BASELINE_PERIOD):
    """Standard deviation per calendar month over a baseline, as a 12-vector.

    Uses the population standard deviation (ddof=0), the same convention as
    xarray's .std() in remove_seasonal_cycle_rolling_with_std, so a series
    standardised here and one standardised there are the same quantity.
    """
    values = np.asarray(values, dtype="float64")
    stamps = pd.to_datetime(np.asarray(time))
    years = stamps.year.values
    months = stamps.month.values
    lo, hi = baseline_period
    in_baseline = (years >= lo) & (years <= hi)
    sigma = np.full(12, np.nan, dtype="float64")
    for month in range(1, 13):
        selected = values[in_baseline & (months == month)]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            sigma[month - 1] = float(np.std(selected))
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        bad = [m + 1 for m in range(12) if not np.isfinite(sigma[m]) or sigma[m] <= 0]
        raise ValueError(
            f"per_calendar_month_sigma: baseline {lo}-{hi} has no spread (or no "
            f"finite values) for calendar month(s) {bad}. Standardising by it "
            f"would divide by zero and hand the detector an infinite index."
        )
    return sigma


def smoothed_standardised_index(anom_degc, time=None,
                                baseline_period=DEFAULT_BASELINE_PERIOD):
    """z3 = 3-month running mean of a degC anomaly, over its own monthly sigma.

    Parameters
    ----------
    anom_degc : xarray.DataArray or array-like
        Deseasonalised anomaly in degC, monthly, already detrended by whatever
        ENSO_DETREND selects. Not standardised: the sigma this function needs
        is the sigma of the SMOOTHED series, which is about 3 per cent smaller
        than the sigma of the monthly one and seasonally different.
    time : array-like, optional
        Month stamps. Taken from anom_degc["time"] when it is a DataArray.
    baseline_period : (int, int)
        Years over which the per-calendar-month sigma is computed.

    Returns
    -------
    (z3, sigma3) : (numpy.ndarray, numpy.ndarray)
        z3 has the length of the input, NaN in the first and last month.
        sigma3 is the 12-vector of per-calendar-month standard deviations of
        the smoothed anomaly, in degC.
    """
    if time is None:
        if not hasattr(anom_degc, "time"):
            raise ValueError("smoothed_standardised_index needs `time` for a bare array")
        time = anom_degc["time"].values
    values = np.asarray(getattr(anom_degc, "values", anom_degc), dtype="float64")
    smoothed = centred_running_mean3(values)
    sigma3 = per_calendar_month_sigma(smoothed, time, baseline_period)
    months = pd.to_datetime(np.asarray(time)).month.values
    return smoothed / sigma3[months - 1], sigma3


def load_threshold_vector(path=None):
    """The 12 monthly detection thresholds t(m), in sd of the smoothed index.

    Built once on COBE by scripts/analysis/build_nino34_threshold_vector.py and
    shared by every dataset and model on purpose, so that a product with weaker
    variance is not handed a looser test than COBE gets.
    """
    path = _Path(path) if path is not None else DEFAULT_THRESHOLD_VECTOR_FILE
    key = str(path)
    if key not in _THRESHOLD_VECTOR_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"Threshold vector {path} not found. Build it with "
                f"scripts/analysis/build_nino34_threshold_vector.py."
            )
        with open(path) as handle:
            payload = _json.load(handle)
        vector = np.asarray(payload["threshold_sd"], dtype="float64")
        if vector.shape != (12,) or not np.all(np.isfinite(vector)) or np.any(vector <= 0):
            raise ValueError(f"{path} does not hold 12 finite positive thresholds")
        _THRESHOLD_VECTOR_CACHE[key] = vector
    return _THRESHOLD_VECTOR_CACHE[key].copy()


def _lift_nested(outer, name, defaults):
    """Rebuild a nested helper of `outer` as a module-level function.

    The code object is the shipped one, not a copy of the logic, so the
    cumulative classifier below cannot drift away from the persistence test
    detect_enso_events applies. Defaults live in the outer function's frame and
    are therefore supplied here; scripts/analysis/tests/test_detector_noaa.py
    checks them against the source.
    """
    candidates = [c for c in outer.__code__.co_consts
                  if isinstance(c, _types.CodeType) and c.co_name == name]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one nested {name}, found {len(candidates)}")
    code = candidates[0]
    if code.co_freevars:
        raise RuntimeError(f"nested {name} closes over {code.co_freevars}; cannot lift")
    lifted = _types.FunctionType(code, outer.__globals__, name)
    lifted.__defaults__ = defaults
    return lifted


has_consecutive_months = _lift_nested(
    detect_enso_events, "has_consecutive_months", (None, 3, ">="))


def classify_intensity_cumulative(sign, sst_slice, pr_slice=None,
                                  extreme_sst=1.75, strong_sst=1.25,
                                  extreme_pr=5.0, min_consecutive=3):
    """Cumulative intensity tiers on the monthly standardised Nino-3 or Nino-4.

    Same thresholds and same three-consecutive-month persistence as the shipped
    classifier, but the tiers are at-or-beyond instead of disjoint bands, so the
    highest tier satisfied wins and nothing can come back unclassified.

    sign > 0 reads Nino-3 and gates extreme on Nino-3 precipitation; sign < 0
    reads Nino-4 and has no precipitation gate. Thresholds are given as positive
    numbers and mirrored for La Nina.
    """
    if sign > 0:
        sst_extreme = has_consecutive_months(
            sst_slice, extreme_sst, min_consecutive=min_consecutive, comparison=">=")
        pr_extreme = pr_slice is not None and has_consecutive_months(
            pr_slice, extreme_pr, min_consecutive=min_consecutive, comparison=">=")
        if sst_extreme and pr_extreme:
            return "extreme"
        if has_consecutive_months(sst_slice, strong_sst,
                                  min_consecutive=min_consecutive, comparison=">="):
            return "strong"
        return "moderate"

    if has_consecutive_months(sst_slice, -extreme_sst,
                              min_consecutive=min_consecutive, comparison="<="):
        return "extreme"
    if has_consecutive_months(sst_slice, -strong_sst,
                              min_consecutive=min_consecutive, comparison="<="):
        return "strong"
    return "moderate"


def _episodes_beyond_vector(z3, months, threshold_vector, min_seasons=NOAA_MIN_SEASONS):
    """Runs of at least min_seasons consecutive seasons beyond +/- t(m)."""
    t_series = np.asarray(threshold_vector, dtype="float64")[np.asarray(months) - 1]
    finite = np.isfinite(z3)
    episodes = []
    for sign in (+1, -1):
        at = np.where(finite, z3 >= t_series, False) if sign > 0 \
            else np.where(finite, z3 <= -t_series, False)
        start = None
        for i, flag in enumerate(at):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_seasons:
                    episodes.append({"first": start, "last": i - 1, "sign": sign})
                start = None
        if start is not None and len(at) - start >= min_seasons:
            episodes.append({"first": start, "last": len(at) - 1, "sign": sign})
    episodes.sort(key=lambda e: (e["first"], -e["sign"]))
    return episodes


def _link_episodes(episodes, z3, gap_months=NOAA_LINK_GAP_MONTHS,
                   quiet_sd=NOAA_LINK_QUIET_SD):
    """Chain same-sign episodes separated by a short, quiet gap into one event.

    A La Nina chain is broken by any gap month at or above +quiet_sd, an El Nino
    chain by any gap month at or below -quiet_sd, so a genuine swing to the
    other phase always terminates the event however short the gap.
    """
    chains = []
    for sign in (+1, -1):
        same = [e for e in episodes if e["sign"] == sign]
        if not same:
            continue
        current = [same[0]]
        for previous, following in zip(same, same[1:]):
            gap = following["first"] - previous["last"] - 1
            gap_values = z3[previous["last"] + 1:following["first"]]
            gap_values = gap_values[np.isfinite(gap_values)]
            if sign < 0:
                quiet = bool(np.all(gap_values <= quiet_sd))
            else:
                quiet = bool(np.all(gap_values >= -quiet_sd))
            if gap <= gap_months and quiet:
                current.append(following)
            else:
                chains.append(current)
                current = [following]
        chains.append(current)
    chains.sort(key=lambda c: c[0]["first"])
    return chains


def detect_enso_events_noaa(nino34_anom_degc, nino3_sst_std, nino3_pr, nino4_sst_std,
                            baseline_period=DEFAULT_BASELINE_PERIOD,
                            threshold_vector=None,
                            min_seasons=NOAA_MIN_SEASONS,
                            link_gap_months=NOAA_LINK_GAP_MONTHS,
                            link_quiet_sd=NOAA_LINK_QUIET_SD,
                            extreme_el_nino_sst_std_threshold=1.75,
                            extreme_el_nino_pr_threshold=5.0,
                            strong_el_nino_sst_lower=1.25,
                            extreme_min_consecutive=3,
                            verbose=True):
    """NOAA CPC's episode rule in standardised units. See the block comment above.

    Returns the SAME dictionary structure as detect_enso_events, so every
    downstream consumer is untouched, with two extra keys on each counted event:

        sub_episodes    "1998-06:2000-05;2000-09:2001-02", empty for one piece
        n_sub_episodes  number of episodes chained into the event

    Parameters
    ----------
    nino34_anom_degc : xarray.DataArray
        Deseasonalised Nino-3.4 anomaly in degC, NOT standardised.
    nino3_sst_std, nino4_sst_std, nino3_pr : xarray.DataArray
        Monthly standardised Nino-3 and Nino-4 indices and the Nino-3
        precipitation anomaly in mm/day, on the same time axis. The
        precipitation series may end early; missing months simply never satisfy
        the extreme gate.
    """
    time_values = nino34_anom_degc["time"].values
    stamps = pd.to_datetime(time_values)
    calendar_months = stamps.month.values

    z3, sigma3 = smoothed_standardised_index(
        nino34_anom_degc, time_values, baseline_period)
    if threshold_vector is None:
        threshold_vector = load_threshold_vector()
    threshold_vector = np.asarray(threshold_vector, dtype="float64")

    episodes = _episodes_beyond_vector(z3, calendar_months, threshold_vector, min_seasons)
    chains = _link_episodes(episodes, z3, link_gap_months, link_quiet_sd)

    def stamp(idx):
        return f"{stamps[idx].year:04d}-{stamps[idx].month:02d}"

    buckets = {
        "extreme_el_nino": [], "strong_el_nino": [], "moderate_el_nino": [],
        "extreme_la_nina": [], "strong_la_nina": [], "moderate_la_nina": [],
    }

    for chain in chains:
        sign = chain[0]["sign"]
        start_idx = chain[0]["first"]
        end_idx = chain[-1]["last"] + 1          # exclusive, as detect_enso_events
        window = slice(start_idx, end_idx)
        sst_slice = (nino3_sst_std if sign > 0 else nino4_sst_std).isel(time=window)
        pr_slice = nino3_pr.isel(time=window) if sign > 0 else None
        level = classify_intensity_cumulative(
            sign, sst_slice, pr_slice,
            extreme_sst=extreme_el_nino_sst_std_threshold,
            strong_sst=strong_el_nino_sst_lower,
            extreme_pr=extreme_el_nino_pr_threshold,
            min_consecutive=extreme_min_consecutive)

        magnitude = np.abs(z3[start_idx:end_idx])
        peak_offset = int(np.nanargmax(magnitude)) if np.any(np.isfinite(magnitude)) else 0
        peak_idx = start_idx + peak_offset

        sub = ";".join(f"{stamp(e['first'])}:{stamp(e['last'])}" for e in chain)
        buckets[f"{level}_{'el_nino' if sign > 0 else 'la_nina'}"].append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": str(time_values[start_idx])[:10],
            "end_date": str(time_values[end_idx - 1])[:10],
            "duration": end_idx - start_idx,
            "peak_month": int(calendar_months[peak_idx]),
            "peak_date": str(time_values[peak_idx])[:10],
            "seasonal_phase_lock_ok": True,
            "discard_reason": None,
            "original_level": level,
            "sub_episodes": sub if len(chain) > 1 else "",
            "n_sub_episodes": len(chain),
        })

    # Neutral periods: everything no counted event covers, exactly as
    # detect_enso_events builds them.
    counted = [e for key in buckets for e in buckets[key]]
    event_mask = np.zeros(len(stamps), dtype=bool)
    for event in counted:
        event_mask[event["start_idx"]:event["end_idx"]] = True

    neutral_periods = []

    def close_neutral(lo, hi):
        magnitude = np.abs(z3[lo:hi])
        peak_offset = int(np.nanargmax(magnitude)) if np.any(np.isfinite(magnitude)) else 0
        peak_idx = lo + peak_offset
        neutral_periods.append({
            "start_idx": lo,
            "end_idx": hi,
            "start_date": str(time_values[lo])[:10],
            "end_date": str(time_values[hi - 1])[:10],
            "duration": hi - lo,
            "peak_month": int(calendar_months[peak_idx]),
            "peak_date": str(time_values[peak_idx])[:10],
            "sub_episodes": "",
            "n_sub_episodes": 1,
        })

    neutral_start = None
    for i, covered in enumerate(event_mask):
        if not covered and neutral_start is None:
            neutral_start = i
        elif covered and neutral_start is not None:
            close_neutral(neutral_start, i)
            neutral_start = None
    if neutral_start is not None:
        close_neutral(neutral_start, len(event_mask))

    el_counted = (len(buckets["extreme_el_nino"]) + len(buckets["strong_el_nino"])
                  + len(buckets["moderate_el_nino"]))
    la_counted = (len(buckets["extreme_la_nina"]) + len(buckets["strong_la_nina"])
                  + len(buckets["moderate_la_nina"]))
    statistics = {
        "total_el_nino_detected": el_counted,
        "el_nino_counted": el_counted,
        "el_nino_discarded": 0,
        "total_la_nina_detected": la_counted,
        "la_nina_counted": la_counted,
        "la_nina_discarded": 0,
        "n_episodes_before_linking": len(episodes),
        "n_events_after_linking": len(chains),
        "sigma_smoothed_by_month": [round(float(v), 4) for v in sigma3],
        "threshold_vector": [round(float(v), 4) for v in threshold_vector],
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print("ENSO EVENT DETECTION (NOAA RONI rule in standardised units)")
        print(f"{'=' * 70}")
        print(f"threshold vector t(m), sd: "
              + " ".join(f"{v:.3f}" for v in threshold_vector))
        print(f"sigma of the smoothed index, degC: "
              + " ".join(f"{v:.3f}" for v in sigma3))
        print(f"episodes before linking: {len(episodes)};  "
              f"events after linking: {len(chains)}")
        for key in ("extreme_el_nino", "strong_el_nino", "moderate_el_nino",
                    "extreme_la_nina", "strong_la_nina", "moderate_la_nina"):
            print(f"\n{key}: {len(buckets[key])}")
            for event in buckets[key]:
                extra = f"  sub-episodes {event['sub_episodes']}" if event["n_sub_episodes"] > 1 else ""
                print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
                      f"({event['duration']} months, peak {event['peak_date'][:7]}){extra}")
        print(f"\nNeutral periods: {len(neutral_periods)}")

    return {
        **buckets,
        "neutral": neutral_periods,
        "discarded_el_nino": [],
        "discarded_la_nina": [],
        "statistics": statistics,
    }
