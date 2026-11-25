"""ENSO event detection and index preprocessing.

These functions are the ones used by the data-reduction pipeline on the
lorenz cluster (scripts/analysis/functions.py in the enso_impacts project),
copied here unchanged so that the observational indices are computed with
exactly the same code as the model indices.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import xarray as xr


def lowpass_filter_gwi(gwi, polynomial_degree=4):
    """
    Smooth GMST using polynomial fit to isolate forced warming trend.

    This approach removes BOTH seasonal cycle AND internal decadal variability
    (including ENSO signal in GMST) before using in detrending regression.
    The polynomial captures only the long-term anthropogenic warming trend.

    Rationale (from co-author feedback):
    - Global seasonality has nothing to do with equatorial seasonality
    - Internal variability (ENSO, PDO) contaminates the forced trend
    - Polynomial order 4 is sufficient to capture non-linear warming without
      overfitting to natural fluctuations

    Parameters
    ----------
    gwi : xarray.DataArray
        Monthly GMST timeseries (1D, time dimension)
    polynomial_degree : int, optional
        Polynomial degree for fitting (default: 4, recommended by co-author)
        - Degree 1: Linear trend (underestimates acceleration)
        - Degree 4: Captures warming acceleration + non-linearity (optimal)
        - Higher degrees: Risk of overfitting to decadal variability

    Returns
    -------
    gwi_smooth : xarray.DataArray
        Smoothed GMST (polynomial fit, no NaN values)

    Notes
    -----
    - Fits polynomial to ENTIRE timeseries (1850-2100)
    - No edge effects or NaN values (polynomial defined everywhere)
    - Removes seasonal cycle (~±0.5°C) AND decadal variability (~±0.2°C)
    - Preserves only forced anthropogenic warming trend
    - Detrending regression uses smoothed GMST:
      T(x,y,t) = α(x,y) · GMST_smooth(t) + β(x,y) + ε(x,y,t)

    Example:
    --------
    Before: GMST contains trend + ENSO + seasonal
    After:  GMST_smooth contains ONLY forced trend
    Result: Detrended SST has NO trend contamination from GMST variability

    References
    ----------
    Co-author feedback (Feb 2026): "Il faut enlever la saisonnalité de ton GWI
    avant de faire le fit car la saisonnalité globale n'a rien a voir avec la
    saisonnalité à l'équateur. Je te conseille de définir ton GWI par un fit
    polynôme d'ordre 4 pour ne pas avoir les fluctuations décennales naturelles."
    """
    # Create time axis for polynomial fit (normalized to [0,1] for numerical stability)
    time_values = np.arange(len(gwi))
    time_normalized = (time_values - time_values.mean()) / time_values.std()

    # Get GWI values (handle NaN if present, though should not be after baseline)
    gwi_values = gwi.values
    valid_mask = ~np.isnan(gwi_values)

    if valid_mask.sum() < polynomial_degree + 1:
        raise ValueError(f"Not enough valid data points ({valid_mask.sum()}) for polynomial degree {polynomial_degree}")

    # Fit polynomial (use only valid data)
    coeffs = np.polyfit(time_normalized[valid_mask], gwi_values[valid_mask], polynomial_degree)

    # Evaluate polynomial at all time points
    gwi_smooth_values = np.polyval(coeffs, time_normalized)

    # Create output DataArray
    gwi_smooth = xr.DataArray(gwi_smooth_values, coords=gwi.coords, dims=gwi.dims, attrs=gwi.attrs.copy())

    # Update attributes
    gwi_smooth.attrs["filtering"] = f"Polynomial fit (degree {polynomial_degree})"
    gwi_smooth.attrs["description"] = "Forced warming trend (seasonal + decadal variability removed)"
    gwi_smooth.attrs["method"] = "Least-squares polynomial regression over full timeseries"
    gwi_smooth.attrs["removes"] = "Seasonal cycle + internal variability (ENSO, PDO)"
    gwi_smooth.attrs["preserves"] = "Anthropogenic forced warming trend only"
    gwi_smooth.attrs["edge_handling"] = "No NaN (polynomial defined everywhere)"
    gwi_smooth.attrs["polynomial_coefficients"] = str(coeffs.tolist())

    return gwi_smooth


def compute_alpha_beta_vectorized(V, G, min_valid: int = 30):
    """Compute regression coefficients for all grid cells at once (VECTORIZED).

    NaN-safe. For each grid point (lat, lon), fits the per-timestep regression
    V[t,lat,lon] = alpha * G[t] + beta + residual using only timesteps where
    BOTH V and G are finite. Grid points with fewer than ``min_valid``
    valid timesteps get alpha = beta = NaN (residuals will then be NaN).

    This replaces a non-NaN-aware earlier version where a single fully-NaN
    timestep in V propagated NaN to every grid point's α, β via unmasked
    ``V.mean(axis=0)``, the bug that caused cesm1_lens r33i1p1's single
    missing month (April 1938) to zero-out all its ENSO indices. See
    project_corrupt_metrics_members.md.
    """
    V = np.asarray(V, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    # Valid mask per (time, lat, lon), both V and G need to be finite.
    G_exp = G[:, None, None]
    valid = np.isfinite(V) & np.isfinite(G_exp)
    n_valid = valid.sum(axis=0)  # per (lat, lon)

    # Replace NaNs with 0 in masked arrays so sums/means only count valid.
    V_m = np.where(valid, V, 0.0)
    G_m = np.where(valid, G_exp, 0.0)

    # Per-grid-point means over valid samples
    with np.errstate(invalid="ignore", divide="ignore"):
        tmean = V_m.sum(axis=0) / n_valid
        gmean = G_m.sum(axis=0) / n_valid  # (lat, lon), gmean varies by
        # grid point because the set
        # of valid timesteps may differ
        Vc = V_m - tmean[None, :, :]
        Gc = G_m - gmean[None, :, :]
        # Zero out contributions from invalid timesteps so they don't pollute
        # the sums (after mean-subtraction they have the wrong value).
        Vc = np.where(valid, Vc, 0.0)
        Gc = np.where(valid, Gc, 0.0)
        cov = (Vc * Gc).sum(axis=0) / n_valid
        varG = (Gc * Gc).sum(axis=0) / n_valid
        alpha = cov / varG
        beta = tmean - alpha * gmean

    # Invalidate grid points with too few samples or degenerate varG
    insufficient = (n_valid < min_valid) | ~np.isfinite(varG) | (varG <= 0)
    alpha = np.where(insufficient, np.nan, alpha)
    beta = np.where(insufficient, np.nan, beta)
    return alpha, beta


def remove_gwi_residuals_vectorized(data, gwi):
    """Remove GWI-correlated signal from spatial data using VECTORIZED regression.

    NaN-safe: residuals are computed per grid point using only valid
    timesteps; invalid timesteps pass through as NaN, which is the correct
    behaviour (downstream code masks NaN).
    """
    print("    Computing vectorized GWI regression residuals...")
    gwi_aligned = gwi.reindex(time=data.time, method="nearest", tolerance=timedelta(days=1))
    if gwi_aligned.isnull().all():
        raise ValueError("GWI reindexing failed - no matching times found")
    V = data.values
    G = gwi_aligned.values
    alpha, beta = compute_alpha_beta_vectorized(V, G)
    # Predicted = alpha * G + beta; residual = V - predicted. Both have NaN
    # wherever alpha/beta are NaN (insufficient-data grid points) or V is NaN.
    predicted = alpha[None, :, :] * G[:, None, None] + beta[None, :, :]
    residuals_np = V - predicted
    residuals = xr.DataArray(residuals_np, coords=data.coords, dims=data.dims, attrs=data.attrs)
    return residuals


def remove_seasonal_cycle_rolling_with_std(data, window_years=30, rolling_std=True, baseline_period=None):
    """
    Remove seasonal cycle and compute standardized anomalies.

    Can use either rolling std (adaptive to climate changes) or baseline std
    (fixed to a reference period for consistent detection across all times).

    Parameters:
    -----------
    data : xarray.DataArray
        Monthly time series data
    window_years : int, optional
        Width of the rolling window in years (default: 30, like ONI)
    rolling_std : bool, optional
        If True (default): Use rolling std for standardization (ONI methodology, adaptive)
        If False: Use baseline std from baseline_period for standardization (consistent thresholds)
    baseline_period : tuple, optional
        Tuple (start_year, end_year) for baseline period when rolling_std=False.
        Example: (1980, 2014). If rolling_std=True, this parameter is ignored.

    Returns:
    --------
    dict : Dictionary containing:
        - 'raw': xarray.DataArray - original data (before seasonal cycle removal)
        - 'anomaly': xarray.DataArray of anomalies (data - seasonal climatology)
        - 'standardized': xarray.DataArray standardized using rolling or baseline std
        - 'rolling_std': xarray.DataArray of rolling/baseline standard deviations
    """

    if rolling_std:
        print(f"  Removing seasonal cycle with {window_years}-year ROLLING window (adaptive std)...")
    else:
        if baseline_period is None:
            raise ValueError("baseline_period required when rolling_std=False")
        print(
            f"  Removing seasonal cycle with {window_years}-year window, standardizing with baseline period {baseline_period[0]}-{baseline_period[1]}..."
        )

    # Store raw data before any modification
    raw = data.copy()

    anomaly = xr.full_like(data, np.nan)
    standardized = xr.full_like(data, np.nan)
    std_values = xr.full_like(data, np.nan)

    years = data.time.dt.year.values
    unique_years = np.unique(years)
    min_year = unique_years.min()
    max_year = unique_years.max()
    half_window = window_years // 2

    print(f"    Data range: {min_year}-{max_year}")

    # ========== Step 1: Compute seasonal climatology (for all periods) ==========
    for year in unique_years:
        for month in range(1, 13):
            # Determine window bounds (centered on current year)
            year_start = year - half_window
            year_end = year + half_window

            # Handle edge cases - shift window if it extends beyond data
            if year_start < min_year:
                year_start = min_year
                year_end = min(min_year + window_years - 1, max_year)
            elif year_end > max_year:
                year_end = max_year
                year_start = max(max_year - window_years + 1, min_year)

            # Get current month's data and window data
            current_mask = (data.time.dt.year == year) & (data.time.dt.month == month)
            window_mask = (
                (data.time.dt.year >= year_start) & (data.time.dt.year <= year_end) & (data.time.dt.month == month)
            )
            window_data = data.sel(time=window_mask)

            # Require minimum number of years for robust statistics
            min_periods = min(10, window_years // 3)
            if len(window_data) >= min_periods:
                clim = window_data.mean(dim="time")

                current_value = data.sel(time=current_mask)
                if len(current_value) > 0:
                    anom = current_value - clim
                    anomaly.loc[dict(time=current_value.time)] = anom

    # ========== Step 2: Compute standardization (rolling or baseline) ==========
    if rolling_std:
        # Rolling std: recompute for each year-month
        for year in unique_years:
            for month in range(1, 13):
                year_start = year - half_window
                year_end = year + half_window

                if year_start < min_year:
                    year_start = min_year
                    year_end = min(min_year + window_years - 1, max_year)
                elif year_end > max_year:
                    year_end = max_year
                    year_start = max(max_year - window_years + 1, min_year)

                current_mask = (data.time.dt.year == year) & (data.time.dt.month == month)
                window_mask = (
                    (anomaly.time.dt.year >= year_start)
                    & (anomaly.time.dt.year <= year_end)
                    & (anomaly.time.dt.month == month)
                )
                window_anom = anomaly.sel(time=window_mask)

                min_periods = min(10, window_years // 3)
                if len(window_anom) >= min_periods:
                    std = window_anom.std(dim="time")

                    current_value_anom = anomaly.sel(time=current_mask)
                    if len(current_value_anom) > 0:
                        std_values.loc[dict(time=current_value_anom.time)] = std

                        # Standardize
                        if float(std) > 1e-10:
                            standardized.loc[dict(time=current_value_anom.time)] = current_value_anom / std
                        else:
                            standardized.loc[dict(time=current_value_anom.time)] = 0.0

    else:
        # Baseline std: compute std only from baseline period, apply to all times
        baseline_start, baseline_end = baseline_period
        baseline_mask = (anomaly.time.dt.year >= baseline_start) & (anomaly.time.dt.year <= baseline_end)
        if not bool(baseline_mask.any()):
            raise ValueError(
                f"remove_seasonal_cycle_rolling_with_std: data range "
                f"{int(anomaly.time.dt.year.values.min())}-"
                f"{int(anomaly.time.dt.year.values.max())} does not overlap "
                f"baseline {baseline_start}-{baseline_end}. This typically "
                f"means the SSP member is missing its historical segment "
                f"(same realization id), fix the upstream file pairing."
            )
        baseline_anom = anomaly.sel(time=baseline_mask)

        # Compute baseline std for each calendar month
        for month in range(1, 13):
            baseline_month_mask = baseline_anom.time.dt.month == month
            baseline_month_anom = baseline_anom.sel(time=baseline_month_mask)

            if len(baseline_month_anom) > 0:
                baseline_std_month = baseline_month_anom.std(dim="time")

                # Apply this baseline std to ALL occurrences of this month
                all_month_mask = anomaly.time.dt.month == month
                std_values.loc[dict(time=all_month_mask)] = baseline_std_month

                # Standardize using baseline std
                current_anom = anomaly.sel(time=all_month_mask)
                if float(baseline_std_month) > 1e-10:
                    standardized.loc[dict(time=all_month_mask)] = current_anom / baseline_std_month
                else:
                    standardized.loc[dict(time=all_month_mask)] = 0.0

    if rolling_std:
        return {"raw": raw, "anomaly": anomaly, "standardized": standardized, "rolling_std": std_values}
    else:
        return {"raw": raw, "anomaly": anomaly, "standardized": standardized}


def merge_events_by_gap(events_dict, nino34_anom, gap_months_threshold=3, min_duration=5, termination_months=2):
    """
    Post-process detected ENSO events to merge those separated by short gaps.

    Merges events ACROSS classification levels (extreme, strong, moderate).
    Only merges if the merged event window satisfies the minimum duration
    (5+ consecutive months) at the appropriate threshold.

    Two merge constraints are checked, both must be satisfied:
      (1) gap_months_threshold: the temporal gap between the two events
          ≤ gap_months_threshold (default: 3 months).
      (2) termination_months: inside the gap, the Niño3.4 anomaly must
          NOT stay with |value| < 0.5 σ for ≥ termination_months
          consecutive months (default: 2).  If it does, the ENSO was
          physically terminated between pulses and we keep them as two
          separate events, even if the temporal gap is short.

    Constraint (2) was added after the 2014-16 obs diagnosis: the
    "failed 2014 El Niño" dropped Niño-3.4 close to zero in summer
    2014 before the real extreme event launched in 2015.  Under the
    original pipeline these were merged into one 25-month "strong EN".
    Requiring SST termination splits them into the physically correct
    two-pulse sequence.

    Parameters
    ----------
    events_dict : dict
        Dictionary returned by detect_enso_events with keys:
        'extreme_el_nino', 'strong_el_nino', 'moderate_el_nino',
        'extreme_la_nina', 'strong_la_nina', 'moderate_la_nina', etc.
    nino34_anom : xarray.DataArray
        Nino3.4 standardized anomaly time series
    gap_months_threshold : int, optional
        Maximum gap (in months) to allow merging (default: 3).
    min_duration : int, optional
        Minimum consecutive months at threshold for valid event (default: 5).
    termination_months : int, optional
        Minimum consecutive months with |Niño-3.4 σ| < 0.5 in the gap
        that blocks a merge (default: 2).  Set to 0 to disable the
        termination check and reproduce the old behaviour.

    Returns
    -------
    dict
        Updated events_dict with merged events
    """

    def count_consecutive_at_threshold(data_slice, phase="el_nino"):
        """Count max consecutive months at/beyond ENSO threshold in a data slice."""
        if phase == "el_nino":
            threshold = 0.5
            condition = data_slice.values >= threshold
        else:  # la_nina
            threshold = -0.5
            condition = data_slice.values <= threshold

        max_consecutive = 0
        current_consecutive = 0
        for val in condition:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive

    def count_consecutive_below_half_sigma(data_slice):
        """Count MAX consecutive months with |Niño3.4 σ| < 0.5 in the gap.
        Large values of this counter indicate the ENSO was physically
        terminated in the gap, so the two surrounding events should NOT
        be merged."""
        below = np.abs(data_slice.values) < 0.5
        max_consec = 0
        cur = 0
        for b in below:
            if b:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        return max_consec

    def merge_events_across_levels(event_lists_by_level, phase):
        """
        Merge events across ALL classification levels if separated by <= gap_months_threshold.

        Parameters
        ----------
        event_lists_by_level : list of lists
            E.g., [extreme_list, strong_list, moderate_list]
        phase : str
            'el_nino' or 'la_nina'

        Returns
        -------
        dict
            Dictionary with keys 'extreme_el_nino', 'strong_el_nino', etc.
            containing merged events.
        """

        # Pool all events and sort by start_idx
        all_events = []
        for level_name, event_list in zip(["extreme", "strong", "moderate"], event_lists_by_level):
            for event in event_list:
                event["original_level"] = level_name
                all_events.append(event)

        if len(all_events) == 0:
            return {"extreme": [], "strong": [], "moderate": []}

        all_events.sort(key=lambda e: e["start_idx"])

        # Merge across levels
        merged_chronological = []
        i = 0

        while i < len(all_events):
            current_event = all_events[i]

            # Try to merge with next event(s) if gap is small
            merged_event = current_event.copy()
            j = i + 1

            while j < len(all_events):
                next_event = all_events[j]
                gap_months = next_event["start_idx"] - merged_event["end_idx"]

                if 0 < gap_months <= gap_months_threshold:
                    # Candidate for merging: validate merged window
                    merged_start = merged_event["start_idx"]
                    merged_end = next_event["end_idx"]
                    merged_slice = nino34_anom.isel(time=slice(merged_start, merged_end))

                    consecutive_at_threshold = count_consecutive_at_threshold(merged_slice, phase)

                    # Termination check: does the gap itself contain a
                    # sustained return to near-neutral Niño-3.4 (|σ|<0.5
                    # for ≥ termination_months)?  If so, the ENSO was
                    # physically terminated and we must NOT merge.
                    gap_slice = nino34_anom.isel(time=slice(merged_event["end_idx"], next_event["start_idx"]))
                    gap_has_termination = (
                        termination_months > 0 and count_consecutive_below_half_sigma(gap_slice) >= termination_months
                    )

                    if consecutive_at_threshold >= min_duration and not gap_has_termination:
                        # Merge is valid: update merged_event
                        merged_duration = merged_end - merged_start
                        abs_values = np.abs(merged_slice.values)
                        peak_idx = np.nanargmax(abs_values)
                        peak_month = int(merged_slice.time.dt.month.values[peak_idx])
                        peak_date = str(merged_slice.time.values[peak_idx])[:10]

                        merged_event = {
                            "start_idx": merged_start,
                            "end_idx": merged_end,
                            "start_date": str(nino34_anom["time"].values[merged_start])[:10],
                            "end_date": str(nino34_anom["time"].values[merged_end - 1])[:10],
                            "duration": merged_duration,
                            "peak_month": peak_month,
                            "peak_date": peak_date,
                            "seasonal_phase_lock_ok": True,
                            "discard_reason": None,
                            "merged": True,
                            "original_levels": f"{current_event.get('original_level', '?')}→...→{next_event.get('original_level', '?')}",
                        }

                        j += 1  # Try to merge next one too
                        continue

                # Gap too large or merge invalid: stop merging
                break

            merged_chronological.append(merged_event)
            i = j if j > i + 1 else i + 1  # Skip merged events

        # Redistribute merged events back to classification levels
        result = {"extreme": [], "strong": [], "moderate": []}
        for event in merged_chronological:
            # Use original level for unmerged events, or classify merged by MAX original level
            if "merged" in event and event["merged"]:
                # Extract original levels from merged composition (e.g., "moderate→strong→extreme")
                original_levels_str = event.get("original_levels", "moderate")
                original_levels_list = original_levels_str.split("→")

                # Classify merged event by HIGHEST original level among all merged events
                if "extreme" in original_levels_list:
                    max_level = "extreme"
                elif "strong" in original_levels_list:
                    max_level = "strong"
                else:
                    max_level = "moderate"

                result[max_level].append(event)
            else:
                original_level = event.get("original_level", "moderate")
                result[original_level].append(event)

        return result

    # Merge El Niño and La Niña separately
    el_nino_by_level = [events_dict["extreme_el_nino"], events_dict["strong_el_nino"], events_dict["moderate_el_nino"]]
    la_nina_by_level = [events_dict["extreme_la_nina"], events_dict["strong_la_nina"], events_dict["moderate_la_nina"]]

    el_nino_merged = merge_events_across_levels(el_nino_by_level, "el_nino")
    la_nina_merged = merge_events_across_levels(la_nina_by_level, "la_nina")

    # Build updated dictionary
    merged_dict = events_dict.copy()
    merged_dict["extreme_el_nino"] = el_nino_merged["extreme"]
    merged_dict["strong_el_nino"] = el_nino_merged["strong"]
    merged_dict["moderate_el_nino"] = el_nino_merged["moderate"]
    merged_dict["extreme_la_nina"] = la_nina_merged["extreme"]
    merged_dict["strong_la_nina"] = la_nina_merged["strong"]
    merged_dict["moderate_la_nina"] = la_nina_merged["moderate"]

    return merged_dict


def fill_small_enso_gaps(nino_index, phase="la_nina", gap_threshold_months=3):
    """
    Fill small gaps in ENSO index to merge events separated by brief reversals.

    When an ENSO event dips briefly above/below threshold for ≤gap_threshold_months,
    fill those months with the event value to keep the event continuous.

    Parameters
    ----------
    nino_index : xarray.DataArray
        Nino3.4 or similar index time series
    phase : str
        'la_nina' (≤ -0.5) or 'el_nino' (≥ 0.5)
    gap_threshold_months : int
        Max gap size to fill (default: 3)

    Returns
    -------
    xarray.DataArray
        Filled index with small gaps removed
    """

    filled = nino_index.copy()
    threshold = -0.5 if phase == "la_nina" else 0.5
    at_threshold = (nino_index.values <= threshold) if phase == "la_nina" else (nino_index.values >= threshold)

    # Find gaps
    i = 0
    while i < len(at_threshold):
        if at_threshold[i]:  # Start of event
            # Find end of continuous event
            j = i + 1
            while j < len(at_threshold) and at_threshold[j]:
                j += 1

            # Check for gap after this event
            gap_start = j
            gap_end = j
            while gap_end < len(at_threshold) and not at_threshold[gap_end]:
                gap_end += 1

            # If gap is small and there's another event after, fill it
            gap_size = gap_end - gap_start
            if 0 < gap_size <= gap_threshold_months and gap_end < len(at_threshold):
                # Fill gap with previous event's value
                fill_value = nino_index.values[gap_start - 1]
                filled.values[gap_start:gap_end] = fill_value
                # Continue from after the gap
                i = gap_end
            else:
                i = j
        else:
            i += 1

    return filled


def detect_enso_events(
    nino34_anom,
    nino3_sst_std,
    nino3_pr,
    nino4_sst_std,
    min_duration=5,
    el_nino_threshold=0.5,
    la_nina_threshold=-0.5,
    extreme_el_nino_sst_std_threshold=1.75,
    extreme_el_nino_pr_threshold=5.0,
    strong_el_nino_sst_lower=1.25,
    strong_el_nino_sst_upper=1.75,
    extreme_la_nina_sst_std_threshold=-1.75,
    strong_la_nina_sst_lower=-1.75,
    strong_la_nina_sst_upper=-1.25,
    extreme_min_consecutive=3,
):
    """
    Detect and classify ENSO events into 7 phases with strict classification criteria.

    Classification criteria (V6 - 7 phases with strict enforcement):

    Detection (using Nino-3.4 SST index):
    - El Nino: Nino-3.4 >= +0.5 for 5+ consecutive months
    - La Nina: Nino-3.4 <= -0.5 for 5+ consecutive months

    Classification (must meet explicit thresholds):
    - Extreme El Nino: Nino-3 SST >= 1.75 s.d. AND Nino-3 precip >= 5 mm/day (both 3+ consecutive months)
    - Strong El Nino:  Nino-3 SST >= 1.25 s.d. AND < 1.75 s.d. (3+ consecutive months)
    - Moderate El Nino: Nino-3 SST < 1.25 s.d. (3+ consecutive months)

    - Extreme La Nina: Nino-4 SST <= -1.75 s.d. (3+ consecutive months)
    - Strong La Nina:  Nino-4 SST > -1.75 s.d. AND <= -1.25 s.d. (3+ consecutive months)
    - Moderate La Nina: Nino-4 SST > -1.25 s.d. (3+ consecutive months)

    Discard criteria:
    - Events that fail seasonal phase-locking constraints
    - Events that don't meet ANY classification criteria (no moderate/strong/extreme match)
    - Events that reach end of time series without properly exiting threshold

    Seasonal phase-locking constraint:
    - Event must be active during at least one D(year0)J(year+1)F(year+1) season
    """

    def has_consecutive_months(data_slice, threshold_lower, threshold_upper=None, min_consecutive=3, comparison=">="):
        """
        Check if data has min_consecutive months meeting threshold criteria.

        For single threshold:
            comparison='>=': data >= threshold_lower
            comparison='<=': data <= threshold_lower
            comparison='>': data > threshold_lower (NEW)
            comparison='<': data < threshold_lower (NEW)

        For range (threshold_upper provided):
            comparison='>=': threshold_lower <= data < threshold_upper
            comparison='<=': threshold_upper < data <= threshold_lower (reversed order)
        """
        if len(data_slice) < min_consecutive:
            return False

        if threshold_upper is None:
            # Single threshold comparison
            if comparison == "<=":
                condition = data_slice.values <= threshold_lower
            elif comparison == ">=":
                condition = data_slice.values >= threshold_lower
            elif comparison == "<":
                condition = data_slice.values < threshold_lower
            elif comparison == ">":
                condition = data_slice.values > threshold_lower
            else:
                raise ValueError(f"Unsupported comparison: {comparison}")
        else:
            # Range comparison
            if comparison == ">=":
                # For El Nino: lower <= data < upper
                condition = (data_slice.values >= threshold_lower) & (data_slice.values < threshold_upper)
            elif comparison == "<=":
                # For La Nina: upper < data <= lower (note: upper is more negative than lower)
                condition = (data_slice.values > threshold_lower) & (data_slice.values <= threshold_upper)
            else:
                raise ValueError(f"Unsupported comparison: {comparison}")

        max_consecutive = 0
        current_consecutive = 0

        for val in condition:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive >= min_consecutive

    def check_seasonal_phase_locking(event_start_idx, event_end_idx, nino34_data):
        """
        Check if event satisfies seasonal phase-locking constraints:
        - Event must be active during at least one D(year0)J(year+1)F(year+1) season
        """
        event_slice = nino34_data.isel(time=slice(event_start_idx, event_end_idx))

        if len(event_slice) == 0:
            return False, None, None, "Empty event"

        months = event_slice.time.dt.month.values
        abs_values = np.abs(event_slice.values)
        peak_idx = np.nanargmax(abs_values)
        peak_month = months[peak_idx]
        peak_date = str(event_slice.time.values[peak_idx])[:10]

        # Event must be active during at least one DJF season
        has_djf = any(m in [12, 1, 2] for m in months)

        if has_djf:
            reason = "OK"
        else:
            reason = "Not active during any DJF season"

        return has_djf, peak_month, peak_date, reason

    def create_event_dict(
        start_idx, end_idx, nino34_data, peak_month=None, peak_date=None, seasonal_ok=True, discard_reason=None
    ):
        """Create event dictionary with all metadata."""
        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": str(nino34_data["time"].values[start_idx])[:10],
            "end_date": str(nino34_data["time"].values[end_idx - 1])[:10],
            "duration": end_idx - start_idx,
            "peak_month": peak_month,
            "peak_date": peak_date,
            "seasonal_phase_lock_ok": seasonal_ok,
            "discard_reason": discard_reason,
        }

    def get_peak_info(event_start_idx, event_end_idx, nino34_data):
        """Return peak month and date for a time slice (by max |anomaly|)."""
        event_slice = nino34_data.isel(time=slice(event_start_idx, event_end_idx))
        if len(event_slice) == 0:
            return None, None

        abs_values = np.abs(event_slice.values)
        peak_idx = np.nanargmax(abs_values)
        peak_month = int(event_slice.time.dt.month.values[peak_idx])
        peak_date = str(event_slice.time.values[peak_idx])[:10]
        return peak_month, peak_date

    # Initialize event lists (V5 - 7 phases)
    extreme_el_nino_events = []
    strong_el_nino_events = []
    moderate_el_nino_events = []
    extreme_la_nina_events = []
    strong_la_nina_events = []
    moderate_la_nina_events = []
    neutral_periods = []
    discarded_el_nino_events = []
    discarded_la_nina_events = []

    in_el_nino = False
    in_la_nina = False
    el_nino_start = None
    la_nina_start = None

    # Main detection loop
    for i in range(len(nino34_anom)):
        current_val = float(nino34_anom.isel(time=i))

        # El Nino detection
        if current_val >= el_nino_threshold:
            if not in_el_nino:
                in_el_nino = True
                el_nino_start = i
        elif in_el_nino:
            if (i - el_nino_start) >= min_duration:
                passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                    el_nino_start, i, nino34_anom
                )

                if passes_seasonal:
                    event_slice_sst = nino3_sst_std.isel(time=slice(el_nino_start, i))
                    event_slice_pr = nino3_pr.isel(time=slice(el_nino_start, i))

                    # Check for Extreme El Nino (SST >= 1.75 AND precip >= 5)
                    sst_extreme = has_consecutive_months(
                        event_slice_sst,
                        extreme_el_nino_sst_std_threshold,
                        min_consecutive=extreme_min_consecutive,
                        comparison=">=",
                    )
                    pr_extreme = has_consecutive_months(
                        event_slice_pr,
                        extreme_el_nino_pr_threshold,
                        min_consecutive=extreme_min_consecutive,
                        comparison=">=",
                    )

                    # Check for Strong El Nino (1.25 <= SST < 1.75)
                    sst_strong = has_consecutive_months(
                        event_slice_sst,
                        strong_el_nino_sst_lower,
                        strong_el_nino_sst_upper,
                        min_consecutive=extreme_min_consecutive,
                        comparison=">=",
                    )

                    # Check for Moderate El Nino (SST < 1.25)
                    sst_moderate = has_consecutive_months(
                        event_slice_sst, 1.25, min_consecutive=extreme_min_consecutive, comparison="<"
                    )

                    event_dict = create_event_dict(el_nino_start, i, nino34_anom, peak_month, peak_date, True, None)

                    # Classification hierarchy (strict: must meet one criterion or discard)
                    # If SST is extreme but precip is not, classify as strong
                    classified = False
                    if sst_extreme and pr_extreme:
                        extreme_el_nino_events.append(event_dict)
                        classified = True
                    elif sst_extreme and not pr_extreme:
                        strong_el_nino_events.append(event_dict)
                        classified = True
                    elif sst_strong:
                        strong_el_nino_events.append(event_dict)
                        classified = True
                    elif sst_moderate:
                        moderate_el_nino_events.append(event_dict)
                        classified = True

                    # Discard if no classification criteria met
                    if not classified:
                        event_dict["discard_reason"] = "Did not meet any El Niño classification criteria"
                        discarded_el_nino_events.append(event_dict)
                else:
                    event_dict = create_event_dict(el_nino_start, i, nino34_anom, peak_month, peak_date, False, reason)
                    discarded_el_nino_events.append(event_dict)

            in_el_nino = False
            el_nino_start = None

        # La Nina detection
        if current_val <= la_nina_threshold:
            if not in_la_nina:
                in_la_nina = True
                la_nina_start = i
        elif in_la_nina:
            if (i - la_nina_start) >= min_duration:
                passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                    la_nina_start, i, nino34_anom
                )

                if passes_seasonal:
                    event_slice_nino4 = nino4_sst_std.isel(time=slice(la_nina_start, i))

                    # Check for Extreme La Nina (SST <= -1.75)
                    is_extreme = has_consecutive_months(
                        event_slice_nino4,
                        extreme_la_nina_sst_std_threshold,
                        min_consecutive=extreme_min_consecutive,
                        comparison="<=",
                    )

                    # Check for Strong La Nina (-1.75 < SST <= -1.25)
                    is_strong = has_consecutive_months(
                        event_slice_nino4,
                        strong_la_nina_sst_lower,
                        strong_la_nina_sst_upper,
                        min_consecutive=extreme_min_consecutive,
                        comparison="<=",
                    )

                    # Check for Moderate La Nina (SST > -1.25)
                    is_moderate = has_consecutive_months(
                        event_slice_nino4, -1.25, min_consecutive=extreme_min_consecutive, comparison=">"
                    )

                    event_dict = create_event_dict(la_nina_start, i, nino34_anom, peak_month, peak_date, True, None)

                    # Classification hierarchy (strict: must meet one criterion or discard)
                    classified = False
                    if is_extreme:
                        extreme_la_nina_events.append(event_dict)
                        classified = True
                    elif is_strong:
                        strong_la_nina_events.append(event_dict)
                        classified = True
                    elif is_moderate:
                        moderate_la_nina_events.append(event_dict)
                        classified = True

                    # Discard if no classification criteria met
                    if not classified:
                        event_dict["discard_reason"] = "Did not meet any La Niña classification criteria"
                        discarded_la_nina_events.append(event_dict)
                else:
                    event_dict = create_event_dict(la_nina_start, i, nino34_anom, peak_month, peak_date, False, reason)
                    discarded_la_nina_events.append(event_dict)

            in_la_nina = False
            la_nina_start = None

    # Handle events at end of time series
    if in_el_nino and (len(nino34_anom) - el_nino_start) >= min_duration:
        # Check if El Niño event properly COMPLETED (exited below 0.5)
        # If it reaches end of time series, it's INCOMPLETE → DISCARD
        last_value = float(nino34_anom.isel(time=-1))

        if last_value < el_nino_threshold:
            # Event properly exited → can classify
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                el_nino_start, len(nino34_anom), nino34_anom
            )

            if passes_seasonal:
                event_slice_sst = nino3_sst_std.isel(time=slice(el_nino_start, None))
                event_slice_pr = nino3_pr.isel(time=slice(el_nino_start, None))

                sst_extreme = has_consecutive_months(
                    event_slice_sst,
                    extreme_el_nino_sst_std_threshold,
                    min_consecutive=extreme_min_consecutive,
                    comparison=">=",
                )
                pr_extreme = has_consecutive_months(
                    event_slice_pr,
                    extreme_el_nino_pr_threshold,
                    min_consecutive=extreme_min_consecutive,
                    comparison=">=",
                )

                sst_strong = has_consecutive_months(
                    event_slice_sst,
                    strong_el_nino_sst_lower,
                    strong_el_nino_sst_upper,
                    min_consecutive=extreme_min_consecutive,
                    comparison=">=",
                )

                # Check for Moderate El Nino (SST < 1.25)
                sst_moderate = has_consecutive_months(
                    event_slice_sst, 1.25, min_consecutive=extreme_min_consecutive, comparison="<"
                )

                event_dict = create_event_dict(
                    el_nino_start, len(nino34_anom), nino34_anom, peak_month, peak_date, True, None
                )

                # Classification hierarchy (strict: must meet one criterion or discard)
                # If SST is extreme but precip is not, classify as strong
                classified = False
                if sst_extreme and pr_extreme:
                    extreme_el_nino_events.append(event_dict)
                    classified = True
                elif sst_extreme and not pr_extreme:
                    strong_el_nino_events.append(event_dict)
                    classified = True
                elif sst_strong:
                    strong_el_nino_events.append(event_dict)
                    classified = True
                elif sst_moderate:
                    moderate_el_nino_events.append(event_dict)
                    classified = True

                # Discard if no classification criteria met
                if not classified:
                    event_dict["discard_reason"] = "Did not meet any El Niño classification criteria"
                    discarded_el_nino_events.append(event_dict)
            else:
                event_dict = create_event_dict(
                    el_nino_start, len(nino34_anom), nino34_anom, peak_month, peak_date, False, reason
                )
                discarded_el_nino_events.append(event_dict)
        else:
            # Event did NOT complete (still > 0.5 at end) → DISCARD
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                el_nino_start, len(nino34_anom), nino34_anom
            )
            event_dict = create_event_dict(
                el_nino_start,
                len(nino34_anom),
                nino34_anom,
                peak_month,
                peak_date,
                False,
                "Incomplete: event did not drop below +0.5 threshold at end of data",
            )
            discarded_el_nino_events.append(event_dict)

    if in_la_nina and (len(nino34_anom) - la_nina_start) >= min_duration:
        # Check if La Niña event properly COMPLETED (exited above -0.5)
        # If it reaches end of time series, it's INCOMPLETE → DISCARD
        last_value = float(nino34_anom.isel(time=-1))

        if last_value > la_nina_threshold:
            # Event properly exited → can classify
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                la_nina_start, len(nino34_anom), nino34_anom
            )

            if passes_seasonal:
                event_slice_nino4 = nino4_sst_std.isel(time=slice(la_nina_start, None))

                is_extreme = has_consecutive_months(
                    event_slice_nino4,
                    extreme_la_nina_sst_std_threshold,
                    min_consecutive=extreme_min_consecutive,
                    comparison="<=",
                )

                is_strong = has_consecutive_months(
                    event_slice_nino4,
                    strong_la_nina_sst_lower,
                    strong_la_nina_sst_upper,
                    min_consecutive=extreme_min_consecutive,
                    comparison="<=",
                )

                # Check for Moderate La Nina (SST > -1.25)
                is_moderate = has_consecutive_months(
                    event_slice_nino4, -1.25, min_consecutive=extreme_min_consecutive, comparison=">"
                )

                event_dict = create_event_dict(
                    la_nina_start, len(nino34_anom), nino34_anom, peak_month, peak_date, True, None
                )

                # Classification hierarchy (strict: must meet one criterion or discard)
                classified = False
                if is_extreme:
                    extreme_la_nina_events.append(event_dict)
                    classified = True
                elif is_strong:
                    strong_la_nina_events.append(event_dict)
                    classified = True
                elif is_moderate:
                    moderate_la_nina_events.append(event_dict)
                    classified = True

                # Discard if no classification criteria met
                if not classified:
                    event_dict["discard_reason"] = "Did not meet any La Niña classification criteria"
                    discarded_la_nina_events.append(event_dict)
            else:
                event_dict = create_event_dict(
                    la_nina_start, len(nino34_anom), nino34_anom, peak_month, peak_date, False, reason
                )
                discarded_la_nina_events.append(event_dict)
        else:
            # Event did NOT complete (still < -0.5 at end) → DISCARD
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                la_nina_start, len(nino34_anom), nino34_anom
            )
            event_dict = create_event_dict(
                la_nina_start,
                len(nino34_anom),
                nino34_anom,
                peak_month,
                peak_date,
                False,
                "Incomplete: event did not rise above -0.5 threshold at end of data",
            )
            discarded_la_nina_events.append(event_dict)

    # Detect neutral periods
    all_counted_events = (
        extreme_el_nino_events
        + strong_el_nino_events
        + moderate_el_nino_events
        + extreme_la_nina_events
        + strong_la_nina_events
        + moderate_la_nina_events
    )
    event_mask = np.zeros(len(nino34_anom), dtype=bool)
    for event in all_counted_events:
        event_mask[event["start_idx"] : event["end_idx"]] = True

    in_neutral = False
    neutral_start = None
    for i in range(len(event_mask)):
        if not event_mask[i]:
            if not in_neutral:
                in_neutral = True
                neutral_start = i
        else:
            if in_neutral and neutral_start is not None:
                neutral_peak_month, neutral_peak_date = get_peak_info(neutral_start, i, nino34_anom)
                event_dict = {
                    "start_idx": neutral_start,
                    "end_idx": i,
                    "start_date": str(nino34_anom["time"].values[neutral_start])[:10],
                    "end_date": str(nino34_anom["time"].values[i - 1])[:10],
                    "duration": i - neutral_start,
                    "peak_month": neutral_peak_month,
                    "peak_date": neutral_peak_date,
                }
                neutral_periods.append(event_dict)
            in_neutral = False
            neutral_start = None

    if in_neutral and neutral_start is not None:
        neutral_peak_month, neutral_peak_date = get_peak_info(neutral_start, len(nino34_anom), nino34_anom)
        event_dict = {
            "start_idx": neutral_start,
            "end_idx": len(nino34_anom),
            "start_date": str(nino34_anom["time"].values[neutral_start])[:10],
            "end_date": str(nino34_anom["time"].values[-1])[:10],
            "duration": len(nino34_anom) - neutral_start,
            "peak_month": neutral_peak_month,
            "peak_date": neutral_peak_date,
        }
        neutral_periods.append(event_dict)

    # Statistics
    total_el_nino_detected = (
        len(extreme_el_nino_events)
        + len(strong_el_nino_events)
        + len(moderate_el_nino_events)
        + len(discarded_el_nino_events)
    )
    total_la_nina_detected = (
        len(extreme_la_nina_events)
        + len(strong_la_nina_events)
        + len(moderate_la_nina_events)
        + len(discarded_la_nina_events)
    )

    statistics = {
        "total_el_nino_detected": total_el_nino_detected,
        "el_nino_counted": len(extreme_el_nino_events) + len(strong_el_nino_events) + len(moderate_el_nino_events),
        "el_nino_discarded": len(discarded_el_nino_events),
        "total_la_nina_detected": total_la_nina_detected,
        "la_nina_counted": len(extreme_la_nina_events) + len(strong_la_nina_events) + len(moderate_la_nina_events),
        "la_nina_discarded": len(discarded_la_nina_events),
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print("ENSO EVENT CLASSIFICATION (V5 - 7 PHASES with seasonal phase-locking)")
    print(f"{'=' * 70}")

    print(f"\nExtreme El Nino events: {len(extreme_el_nino_events)}")
    for event in extreme_el_nino_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nStrong El Nino events: {len(strong_el_nino_events)}")
    for event in strong_el_nino_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nModerate El Nino events: {len(moderate_el_nino_events)}")
    for event in moderate_el_nino_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nDiscarded El Nino events (failed seasonal constraint): {len(discarded_el_nino_events)}")
    for event in discarded_el_nino_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months) - {event['discard_reason']}"
        )

    print(f"\nExtreme La Nina events: {len(extreme_la_nina_events)}")
    for event in extreme_la_nina_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nStrong La Nina events: {len(strong_la_nina_events)}")
    for event in strong_la_nina_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nModerate La Nina events: {len(moderate_la_nina_events)}")
    for event in moderate_la_nina_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months, peak month: {event['peak_month']})"
        )

    print(f"\nDiscarded La Nina events (failed seasonal constraint): {len(discarded_la_nina_events)}")
    for event in discarded_la_nina_events:
        print(
            f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
            f"({event['duration']} months) - {event['discard_reason']}"
        )

    print(f"\nNeutral periods: {len(neutral_periods)}")
    for event in neutral_periods:
        if event["duration"] >= 3:
            print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} ({event['duration']} months)")

    # Time distribution
    total_months = len(nino34_anom)
    extreme_el_nino_months = sum(e["duration"] for e in extreme_el_nino_events)
    strong_el_nino_months = sum(e["duration"] for e in strong_el_nino_events)
    moderate_el_nino_months = sum(e["duration"] for e in moderate_el_nino_events)
    extreme_la_nina_months = sum(e["duration"] for e in extreme_la_nina_events)
    strong_la_nina_months = sum(e["duration"] for e in strong_la_nina_events)
    moderate_la_nina_months = sum(e["duration"] for e in moderate_la_nina_events)
    neutral_months = sum(e["duration"] for e in neutral_periods)

    print(f"\n{'=' * 70}")
    print("TIME DISTRIBUTION (counted events only)")
    print(f"{'=' * 70}")
    print(f"Total months: {total_months}")
    print(f"Extreme El Nino:  {extreme_el_nino_months} months ({100 * extreme_el_nino_months / total_months:.1f}%)")
    print(f"Strong El Nino:   {strong_el_nino_months} months ({100 * strong_el_nino_months / total_months:.1f}%)")
    print(f"Moderate El Nino: {moderate_el_nino_months} months ({100 * moderate_el_nino_months / total_months:.1f}%)")
    print(f"Extreme La Nina:  {extreme_la_nina_months} months ({100 * extreme_la_nina_months / total_months:.1f}%)")
    print(f"Strong La Nina:   {strong_la_nina_months} months ({100 * strong_la_nina_months / total_months:.1f}%)")
    print(f"Moderate La Nina: {moderate_la_nina_months} months ({100 * moderate_la_nina_months / total_months:.1f}%)")
    print(f"Neutral:          {neutral_months} months ({100 * neutral_months / total_months:.1f}%)")

    print(f"\n{'=' * 70}")
    print("SEASONAL PHASE-LOCKING STATISTICS")
    print(f"{'=' * 70}")
    print(
        f"El Nino:  {statistics['el_nino_counted']} counted, "
        f"{statistics['el_nino_discarded']} discarded "
        f"(of {statistics['total_el_nino_detected']} total detected)"
    )
    print(
        f"La Nina:  {statistics['la_nina_counted']} counted, "
        f"{statistics['la_nina_discarded']} discarded "
        f"(of {statistics['total_la_nina_detected']} total detected)"
    )

    return {
        "extreme_el_nino": extreme_el_nino_events,
        "strong_el_nino": strong_el_nino_events,
        "moderate_el_nino": moderate_el_nino_events,
        "extreme_la_nina": extreme_la_nina_events,
        "strong_la_nina": strong_la_nina_events,
        "moderate_la_nina": moderate_la_nina_events,
        "neutral": neutral_periods,
        "discarded_el_nino": discarded_el_nino_events,
        "discarded_la_nina": discarded_la_nina_events,
        "statistics": statistics,
    }
