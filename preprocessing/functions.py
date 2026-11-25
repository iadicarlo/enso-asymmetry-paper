"""
Essential functions for ENSO analysis

This module contains reusable functions for:
- ENSO event detection and classification
- Seasonal averaging and filtering
- Data preprocessing (detrending, masking)
- Population exposure analysis
- Probability ratio calculations

Author: ENSO Impacts Project
Last updated: February 2026
"""

import pandas as pd
import numpy as np
import xarray as xr
import s3fs
import regionmask
import rasterio
from scipy.stats import binned_statistic_2d
from datetime import timedelta


# ============================================================================
# GWI PROCESSING
# ============================================================================

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
    gwi_smooth = xr.DataArray(
        gwi_smooth_values,
        coords=gwi.coords,
        dims=gwi.dims,
        attrs=gwi.attrs.copy()
    )
    
    # Update attributes
    gwi_smooth.attrs['filtering'] = f'Polynomial fit (degree {polynomial_degree})'
    gwi_smooth.attrs['description'] = 'Forced warming trend (seasonal + decadal variability removed)'
    gwi_smooth.attrs['method'] = 'Least-squares polynomial regression over full timeseries'
    gwi_smooth.attrs['removes'] = 'Seasonal cycle + internal variability (ENSO, PDO)'
    gwi_smooth.attrs['preserves'] = 'Anthropogenic forced warming trend only'
    gwi_smooth.attrs['edge_handling'] = 'No NaN (polynomial defined everywhere)'
    gwi_smooth.attrs['polynomial_coefficients'] = str(coeffs.tolist())
    
    return gwi_smooth


# ============================================================================
# SEASONAL AVERAGING UTILITIES
# ============================================================================

def parse_season_string(season_str):
    """
    Parse season string (e.g., 'NDJF', 'JJA', 'MAM', 'full') into month numbers.

    Handles ambiguous letters using context and standard climatology conventions:
    - JJA = Jun-Jul-Aug (summer in NH)
    - MAM = Mar-Apr-May (spring in NH)
    - DJF = Dec-Jan-Feb (winter in NH)
    - SON = Sep-Oct-Nov (autumn in NH)

    Returns:
    --------
    list of int : Month numbers (1-12), or None for 'full'
    """
    if season_str.lower() == 'full':
        return None

    season_upper = season_str.upper()

    # Common season patterns (check these first for unambiguous parsing)
    season_patterns = {
        'JJA': [6, 7, 8],        # Jun-Jul-Aug
        'MAM': [3, 4, 5],        # Mar-Apr-May
        'SON': [9, 10, 11],      # Sep-Oct-Nov
        'DJF': [12, 1, 2],       # Dec-Jan-Feb
        'NDJF': [11, 12, 1, 2],  # Nov-Dec-Jan-Feb (ENSO peak)
        'ONDJF': [10, 11, 12, 1, 2],  # Oct-Nov-Dec-Jan-Feb (extended ENSO peak)
        'DJFM': [12, 1, 2, 3],   # Dec-Jan-Feb-Mar
        'JJAS': [6, 7, 8, 9],    # Jun-Jul-Aug-Sep
        'MJJA': [5, 6, 7, 8],    # May-Jun-Jul-Aug
        'MAMJ': [3, 4, 5, 6],    # Mar-Apr-May-Jun
    }

    if season_upper in season_patterns:
        return season_patterns[season_upper]

    # Parse character by character with context-aware rules
    months = []
    for i, char in enumerate(season_upper):
        prev_month = months[-1] if len(months) > 0 else None
        next_char = season_upper[i+1] if i+1 < len(season_upper) else None

        if char == 'J':
            # J could be Jan(1), Jun(6), or Jul(7)
            if prev_month is None or prev_month in [11, 12]:  # Start or after Nov/Dec
                months.append(1)  # January
            elif prev_month == 5:  # After May
                months.append(6)  # June
            elif prev_month == 6:  # After June
                months.append(7)  # July
            elif next_char == 'J':  # JJ at start = June-July
                months.append(6)  # First J = June
            else:
                months.append(1)  # Default to January

        elif char == 'F':
            months.append(2)  # February

        elif char == 'M':
            # M could be Mar(3) or May(5)
            if prev_month is None or prev_month in [1, 2, 12]:  # Start or winter months
                months.append(3)  # March
            elif prev_month == 4:  # After April
                months.append(5)  # May
            elif next_char == 'A':  # MA = Mar-Apr (spring)
                months.append(3)  # March
            elif next_char == 'J':  # MJ = May-Jun (late spring/early summer)
                months.append(5)  # May
            else:
                months.append(3)  # Default to March

        elif char == 'A':
            # A could be Apr(4) or Aug(8)
            if prev_month is None or prev_month in [1, 2, 3]:  # Start or winter/spring
                months.append(4)  # April
            elif prev_month in [7]:  # After July
                months.append(8)  # August
            elif next_char == 'M' and prev_month in [None, 1, 2, 12]:  # AM = Apr-May
                months.append(4)  # April
            elif next_char == 'S':  # AS = Aug-Sep
                months.append(8)  # August
            else:
                months.append(4)  # Default to April

        elif char == 'S':
            months.append(9)  # September
        elif char == 'O':
            months.append(10)  # October
        elif char == 'N':
            months.append(11)  # November
        elif char == 'D':
            months.append(12)  # December
        else:
            raise ValueError(f"Invalid month character: {char}")

    return months


def filter_event_by_season(event_dict, precip_daily_data, season_months):
    """
    Filter event to only include specified season months, respecting year transitions.

    For seasons spanning years (e.g., ONDJF = Oct-Nov-Dec-Jan-Feb), selects one
    complete seasonal cycle within the event period, choosing the year with the
    most complete coverage.

    Example:
        Event: Jun 1997 to Apr 1998
        Season: ONDJF [10,11,12,1,2]
        Result: Oct-Nov-Dec 1997 + Jan-Feb 1998 (one complete ONDJF cycle)

    Parameters:
    -----------
    event_dict : dict
        Event dictionary with 'start_date' and 'end_date'
    precip_daily_data : xr.DataArray
        Daily precipitation data
    season_months : list of int or None
        Month numbers to include (e.g., [10,11,12,1,2] for ONDJF), or None for full

    Returns:
    --------
    xr.DataArray or None : Filtered precipitation data, or None if no overlap
    """
    # Get event precipitation
    event_precip = precip_daily_data.sel(time=slice(event_dict['start_date'], event_dict['end_date']))

    if season_months is None:
        # No filtering - return full event
        return event_precip

    # Detect if season spans new year (e.g., NDJF, ONDJF, DJFM)
    # This happens when months decrease (e.g., [11,12,1,2]: 11->12 increases, 12->1 decreases)
    spans_new_year = any(season_months[i] > season_months[i+1] for i in range(len(season_months)-1))

    if not spans_new_year:
        # Simple case: season within one calendar year (e.g., JJA, MAM, SON).
        #
        # Select ONE seasonal cycle, the year with the most days, exactly as the
        # cross-year branch below does. Returning every matching month made the
        # window length scale with the event's duration, so a multi-year neutral
        # period contributed 3 or 5 JJAs (276 or 460 days) against a single
        # El Nino JJA (92 days). For MAX-type statistics (RX1DAY, RX10DAY and
        # especially CDD, which grows with window length) that inflated the
        # neutral baseline and pushed EVERY anomaly negative, for La Nina and
        # El Nino alike. DJF was unaffected because it took this branch's
        # one-cycle logic already, which is why the bias showed up in
        # MAM/JJA/SON only.
        event_months = event_precip.time.dt.month.values
        event_years = event_precip.time.dt.year.values
        mask_any = np.isin(event_months, season_months)
        if not mask_any.any():
            return None

        best_cycle, max_days = None, 0
        for yr in np.unique(event_years[mask_any]):
            mask = mask_any & (event_years == yr)
            n_days = int(mask.sum())
            if n_days > max_days:
                max_days, best_cycle = n_days, event_precip.isel(time=mask)
        return best_cycle

    else:
        # Complex case: season spans new year (e.g., ONDJF, NDJF, DJFM)
        # Strategy: Find all possible seasonal cycles and pick the one with most complete data

        # Split season into "late year" (>=7) and "early year" (<7) parts
        # Assuming seasons span at most 6 months and cross one year boundary
        late_months = [m for m in season_months if m >= 7]   # e.g., [10,11,12] for ONDJF
        early_months = [m for m in season_months if m < 7]   # e.g., [1,2] for ONDJF

        # Get years present in event
        event_years = event_precip.time.dt.year.values
        unique_years = np.unique(event_years)

        best_cycle = None
        max_days = 0

        # Try each year as the "late year" anchor
        for late_year in unique_years:
            early_year = late_year + 1

            # Select data for this seasonal cycle
            mask = np.zeros(len(event_precip.time), dtype=bool)

            # Use xarray's dt accessor which handles both standard and cftime calendars
            months = event_precip.time.dt.month.values
            years = event_precip.time.dt.year.values

            for i in range(len(event_precip.time)):
                month = months[i]
                year = years[i]

                # Include if month matches and year is correct
                if month in late_months and year == late_year:
                    mask[i] = True
                elif month in early_months and year == early_year:
                    mask[i] = True

            if mask.any():
                n_days = mask.sum()
                if n_days > max_days:
                    max_days = n_days
                    best_cycle = event_precip.isel(time=mask)

        return best_cycle  # Returns None if no valid cycle found


def check_neutral_contains_full_season(event_dict, season_months):
    """
    Check if a neutral period contains at least one complete seasonal cycle.

    For year-spanning seasons (e.g., ONDJF), checks if there exists at least one
    year where all late-year months (O,N,D) and early-year months (J,F) are present
    in consecutive years.

    Parameters:
    -----------
    event_dict : dict
        Neutral event dictionary with 'start_date' and 'end_date'
    season_months : list of int
        Month numbers that must all be present (e.g., [10,11,12,1,2] for ONDJF)

    Returns:
    --------
    bool : True if at least one complete seasonal cycle exists
    """
    if season_months is None:
        # No filtering - always valid
        return True

    # Parse start and end dates
    start_date = pd.to_datetime(event_dict['start_date'])
    end_date = pd.to_datetime(event_dict['end_date'])

    # Generate all months in the event period
    event_date_range = pd.date_range(start_date, end_date, freq='MS')  # Month start

    # Detect if season spans new year
    spans_new_year = any(season_months[i] > season_months[i+1] for i in range(len(season_months)-1))

    if not spans_new_year:
        # All required months must be present within ONE calendar year.
        #
        # Pooling months across the whole period (the previous behaviour) let
        # periods with no complete season through: Jul 1990 to Jun 1991 contains
        # a June and a July and an August, but never all three in the same year,
        # so there is no JJA to composite. Those periods then contributed a
        # partial season, which drags MAX-type statistics (RX1DAY, RX10DAY, CDD)
        # downward and biases the neutral baseline.
        months_by_year = {}
        for ts in event_date_range:
            months_by_year.setdefault(ts.year, set()).add(ts.month)
        required_months = set(season_months)
        return any(required_months.issubset(m) for m in months_by_year.values())

    else:
        # Complex case: need complete seasonal cycle spanning two consecutive years
        # Split season into late and early parts
        late_months = [m for m in season_months if m >= 7]
        early_months = [m for m in season_months if m < 7]

        # Create a dict of year -> months present
        year_months = {}
        for date in event_date_range:
            year = date.year
            month = date.month
            if year not in year_months:
                year_months[year] = set()
            year_months[year].add(month)

        # Check if any consecutive year pair has the full cycle
        sorted_years = sorted(year_months.keys())

        for i in range(len(sorted_years) - 1):
            late_year = sorted_years[i]
            early_year = sorted_years[i + 1]

            # Check if late_year has all late_months AND early_year has all early_months
            has_late = all(m in year_months[late_year] for m in late_months)
            has_early = all(m in year_months[early_year] for m in early_months)

            if has_late and has_early:
                return True

        # Also check if event starts in late year and extends into next
        # (edge case for events starting mid-year)
        if len(sorted_years) >= 2:
            # Check last year has late months and any year has early months
            last_year = sorted_years[-1]
            has_late_in_last = all(m in year_months[last_year] for m in late_months)

            # Check if any year has early months
            has_early_anywhere = any(
                all(m in year_months[y] for m in early_months)
                for y in sorted_years
            )

            if has_late_in_last and has_early_anywhere:
                # More strict: they should be consecutive
                for i in range(len(sorted_years) - 1):
                    if (all(m in year_months[sorted_years[i]] for m in late_months) and
                        all(m in year_months[sorted_years[i+1]] for m in early_months)):
                        return True

        return False


# ============================================================================
# DAILY PRECIP EXTREME INDICES
# ============================================================================

def compute_rolling_accumulated_max(event_precip, window_days=10):
    """
    Compute maximum rolling N-day accumulated precipitation for one event.

    Parameters
    ----------
    event_precip : xarray.DataArray
        Daily precipitation for one event with a ``time`` dimension.
    window_days : int, optional
        Rolling accumulation window in days (default: 10).

    Returns
    -------
    xarray.DataArray or None
        Maximum rolling accumulated precipitation (same spatial dims as input),
        or None when fewer than ``window_days`` samples are available.
    """
    if 'time' not in event_precip.dims:
        raise ValueError("event_precip must have a 'time' dimension")

    n_days = int(event_precip.sizes.get('time', 0))
    if n_days < window_days:
        return None

    rolling_accum = event_precip.rolling(time=window_days, min_periods=window_days).sum()
    return rolling_accum.max(dim='time')


# ============================================================================
# DATA MASKING AND EXTRACTION
# ============================================================================

def mask_ocean(data):
    """Mask ocean values, keeping only land"""
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land_mask.mask(data.lon, data.lat)
    return data.where(~np.isnan(mask))


def mask_latin_america(data):
    """Mask to only include Latin America region (270-327°E, -55-15°N)"""
    lat_mask = (data.lat >= -55) & (data.lat <= 15)
    lon_mask = (data.lon >= 270) & (data.lon <= 327)
    return data.where(lat_mask & lon_mask)


def extract_country_data(data, country_names):
    """
    Extract data within country polygons

    Parameters:
    -----------
    data : xarray.DataArray
        Data with lon, lat coordinates
    country_names : list of str
        Country names like ['Brazil', 'Colombia']

    Returns:
    --------
    dict : country_name -> masked DataArray
    """
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110

    result = {}
    for country_name in country_names:
        # Find country by searching through the region dict
        country_code = None
        for code, region in countries.regions.items():
            if country_name.lower() in region.name.lower():
                country_code = code
                print(f"Found {country_name}: {region.name} (code: {code})")
                break

        if country_code is not None:
            mask = countries.mask(data.lon, data.lat) == country_code
            result[country_name] = data.where(mask)
        else:
            print(f"Warning: {country_name} not found")
            # Show available countries for debugging
            print("Available countries:", [r.name for r in countries.regions.values()][:10], "...")

    return result


# ============================================================================
# S3 DATA ACCESS
# ============================================================================

def get_available_members(bucket='ncar-cesm2-lens', scenario='historical', forcing='cmip6'):
    """Get list of available member IDs from S3"""
    fs = s3fs.S3FileSystem(anon=True)

    # Load a sample variable to get member list
    zarr_path = f"s3://{bucket}/atm/monthly/cesm2LE-{scenario}-{forcing}-TS.zarr"
    store = s3fs.S3Map(root=zarr_path, s3=fs, check=False)
    ds = xr.open_zarr(store, consolidated=True)

    members = ds.member_id.values
    print(f"Found {len(members)} members")
    return members


def reindex_time(ds):
    """Reindex time by shifting dates back by 15 days"""
    ds['time'] = ds.time.get_index('time') - timedelta(days=15)
    return ds


# ============================================================================
# DETRENDING AND PREPROCESSING
# ============================================================================

def compute_gwi(tas_global):
    """Compute Global Warming Index from global mean surface air temperature."""
    weights = np.cos(np.deg2rad(tas_global.lat))
    weights_normalized = weights / weights.sum()
    weighted_tas = tas_global.weighted(xr.DataArray(weights_normalized, dims=['lat'], coords={'lat': tas_global.lat}))
    gwi = weighted_tas.mean(dim=['lat', 'lon'])
    return gwi


def compute_alpha_beta_vectorized(V, G, min_valid: int = 30):
    """Compute regression coefficients for all grid cells at once (VECTORIZED).

    NaN-safe. For each grid point (lat, lon), fits the per-timestep regression
    V[t,lat,lon] = alpha * G[t] + beta + residual using only timesteps where
    BOTH V and G are finite. Grid points with fewer than ``min_valid``
    valid timesteps get alpha = beta = NaN (residuals will then be NaN).

    This replaces a non-NaN-aware earlier version where a single fully-NaN
    timestep in V propagated NaN to every grid point's α, β via unmasked
    ``V.mean(axis=0)`` — the bug that caused cesm1_lens r33i1p1's single
    missing month (April 1938) to zero-out all its ENSO indices. See
    project_corrupt_metrics_members.md.
    """
    V = np.asarray(V, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    # Valid mask per (time, lat, lon) — both V and G need to be finite.
    G_exp = G[:, None, None]
    valid = np.isfinite(V) & np.isfinite(G_exp)
    n_valid = valid.sum(axis=0)  # per (lat, lon)

    # Replace NaNs with 0 in masked arrays so sums/means only count valid.
    V_m = np.where(valid, V, 0.0)
    G_m = np.where(valid, G_exp, 0.0)

    # Per-grid-point means over valid samples
    with np.errstate(invalid="ignore", divide="ignore"):
        tmean = V_m.sum(axis=0) / n_valid
        gmean = G_m.sum(axis=0) / n_valid   # (lat, lon) — gmean varies by
                                            # grid point because the set
                                            # of valid timesteps may differ
        Vc = V_m - tmean[None, :, :]
        Gc = G_m - gmean[None, :, :]
        # Zero out contributions from invalid timesteps so they don't pollute
        # the sums (after mean-subtraction they have the wrong value).
        Vc = np.where(valid, Vc, 0.0)
        Gc = np.where(valid, Gc, 0.0)
        cov  = (Vc * Gc).sum(axis=0) / n_valid
        varG = (Gc * Gc).sum(axis=0) / n_valid
        alpha = cov / varG
        beta  = tmean - alpha * gmean

    # Invalidate grid points with too few samples or degenerate varG
    insufficient = (n_valid < min_valid) | ~np.isfinite(varG) | (varG <= 0)
    alpha = np.where(insufficient, np.nan, alpha)
    beta  = np.where(insufficient, np.nan, beta)
    return alpha, beta


def remove_gwi_residuals_vectorized(data, gwi):
    """Remove GWI-correlated signal from spatial data using VECTORIZED regression.

    NaN-safe: residuals are computed per grid point using only valid
    timesteps; invalid timesteps pass through as NaN, which is the correct
    behaviour (downstream code masks NaN).
    """
    print(f"    Computing vectorized GWI regression residuals...")
    gwi_aligned = gwi.reindex(time=data.time, method='nearest', tolerance=timedelta(days=1))
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


def linear_detrend_timeseries(data):
    """Remove linear trend from 1D timeseries."""
    time_numeric = np.arange(len(data))
    valid_mask = ~np.isnan(data.values)
    if valid_mask.sum() < 10:
        return data
    slope, intercept = np.polyfit(time_numeric[valid_mask], data.values[valid_mask], 1)
    trend = slope * time_numeric + intercept
    detrended = data - trend
    return detrended


def remove_seasonal_cycle_rolling_with_std(data, window_years=30, rolling_std=True, 
                                           baseline_period=None):
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
        print(f"  Removing seasonal cycle with {window_years}-year window, standardizing with baseline period {baseline_period[0]}-{baseline_period[1]}...")

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
            window_mask = ((data.time.dt.year >= year_start) &
                          (data.time.dt.year <= year_end) &
                          (data.time.dt.month == month))
            window_data = data.sel(time=window_mask)

            # Require minimum number of years for robust statistics
            min_periods = min(10, window_years // 3)
            if len(window_data) >= min_periods:
                clim = window_data.mean(dim='time')

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
                window_mask = ((anomaly.time.dt.year >= year_start) &
                              (anomaly.time.dt.year <= year_end) &
                              (anomaly.time.dt.month == month))
                window_anom = anomaly.sel(time=window_mask)

                min_periods = min(10, window_years // 3)
                if len(window_anom) >= min_periods:
                    std = window_anom.std(dim='time')

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
        baseline_mask = ((anomaly.time.dt.year >= baseline_start) &
                        (anomaly.time.dt.year <= baseline_end))
        if not bool(baseline_mask.any()):
            raise ValueError(
                f"remove_seasonal_cycle_rolling_with_std: data range "
                f"{int(anomaly.time.dt.year.values.min())}-"
                f"{int(anomaly.time.dt.year.values.max())} does not overlap "
                f"baseline {baseline_start}-{baseline_end}. This typically "
                f"means the SSP member is missing its historical segment "
                f"(same realization id) — fix the upstream file pairing."
            )
        baseline_anom = anomaly.sel(time=baseline_mask)

        # Compute baseline std for each calendar month
        for month in range(1, 13):
            baseline_month_mask = baseline_anom.time.dt.month == month
            baseline_month_anom = baseline_anom.sel(time=baseline_month_mask)

            if len(baseline_month_anom) > 0:
                baseline_std_month = baseline_month_anom.std(dim='time')
                
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
        return {
            'raw': raw,
            'anomaly': anomaly,
            'standardized': standardized,
            'rolling_std': std_values
        }
    else:
        return {
            'raw': raw,
            'anomaly': anomaly,
            'standardized': standardized
        }


# ============================================================================
# ENSO EVENT POST-PROCESSING (Gap-based merging)
# ============================================================================

def merge_events_by_gap(events_dict, nino34_anom, gap_months_threshold=3,
                          min_duration=5, termination_months=2):
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
          separate events — even if the temporal gap is short.

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
    
    def count_consecutive_at_threshold(data_slice, phase='el_nino'):
        """Count max consecutive months at/beyond ENSO threshold in a data slice."""
        if phase == 'el_nino':
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
        for level_name, event_list in zip(['extreme', 'strong', 'moderate'], event_lists_by_level):
            for event in event_list:
                event['original_level'] = level_name
                all_events.append(event)
        
        if len(all_events) == 0:
            return {'extreme': [], 'strong': [], 'moderate': []}
        
        all_events.sort(key=lambda e: e['start_idx'])
        
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
                gap_months = next_event['start_idx'] - merged_event['end_idx']
                
                if 0 < gap_months <= gap_months_threshold:
                    # Candidate for merging: validate merged window
                    merged_start = merged_event['start_idx']
                    merged_end = next_event['end_idx']
                    merged_slice = nino34_anom.isel(time=slice(merged_start, merged_end))

                    consecutive_at_threshold = count_consecutive_at_threshold(merged_slice, phase)

                    # Termination check: does the gap itself contain a
                    # sustained return to near-neutral Niño-3.4 (|σ|<0.5
                    # for ≥ termination_months)?  If so, the ENSO was
                    # physically terminated and we must NOT merge.
                    gap_slice = nino34_anom.isel(
                        time=slice(merged_event['end_idx'],
                                    next_event['start_idx']))
                    gap_has_termination = (
                        termination_months > 0
                        and count_consecutive_below_half_sigma(gap_slice)
                            >= termination_months
                    )

                    if consecutive_at_threshold >= min_duration \
                            and not gap_has_termination:
                        # Merge is valid: update merged_event
                        merged_duration = merged_end - merged_start
                        abs_values = np.abs(merged_slice.values)
                        peak_idx = np.nanargmax(abs_values)
                        peak_month = int(merged_slice.time.dt.month.values[peak_idx])
                        peak_date = str(merged_slice.time.values[peak_idx])[:10]
                        
                        merged_event = {
                            'start_idx': merged_start,
                            'end_idx': merged_end,
                            'start_date': str(nino34_anom['time'].values[merged_start])[:10],
                            'end_date': str(nino34_anom['time'].values[merged_end-1])[:10],
                            'duration': merged_duration,
                            'peak_month': peak_month,
                            'peak_date': peak_date,
                            'seasonal_phase_lock_ok': True,
                            'discard_reason': None,
                            'merged': True,
                            'original_levels': f"{current_event.get('original_level', '?')}→...→{next_event.get('original_level', '?')}"
                        }
                        
                        j += 1  # Try to merge next one too
                        continue
                
                # Gap too large or merge invalid: stop merging
                break
            
            merged_chronological.append(merged_event)
            i = j if j > i + 1 else i + 1  # Skip merged events
        
        # Redistribute merged events back to classification levels
        result = {'extreme': [], 'strong': [], 'moderate': []}
        for event in merged_chronological:
            # Use original level for unmerged events, or classify merged by MAX original level
            if 'merged' in event and event['merged']:
                # Extract original levels from merged composition (e.g., "moderate→strong→extreme")
                original_levels_str = event.get('original_levels', 'moderate')
                original_levels_list = original_levels_str.split('→')
                
                # Classify merged event by HIGHEST original level among all merged events
                if 'extreme' in original_levels_list:
                    max_level = 'extreme'
                elif 'strong' in original_levels_list:
                    max_level = 'strong'
                else:
                    max_level = 'moderate'
                
                result[max_level].append(event)
            else:
                original_level = event.get('original_level', 'moderate')
                result[original_level].append(event)
        
        return result
    
    # Merge El Niño and La Niña separately
    el_nino_by_level = [
        events_dict['extreme_el_nino'],
        events_dict['strong_el_nino'],
        events_dict['moderate_el_nino']
    ]
    la_nina_by_level = [
        events_dict['extreme_la_nina'],
        events_dict['strong_la_nina'],
        events_dict['moderate_la_nina']
    ]
    
    el_nino_merged = merge_events_across_levels(el_nino_by_level, 'el_nino')
    la_nina_merged = merge_events_across_levels(la_nina_by_level, 'la_nina')
    
    # Build updated dictionary
    merged_dict = events_dict.copy()
    merged_dict['extreme_el_nino'] = el_nino_merged['extreme']
    merged_dict['strong_el_nino'] = el_nino_merged['strong']
    merged_dict['moderate_el_nino'] = el_nino_merged['moderate']
    merged_dict['extreme_la_nina'] = la_nina_merged['extreme']
    merged_dict['strong_la_nina'] = la_nina_merged['strong']
    merged_dict['moderate_la_nina'] = la_nina_merged['moderate']
    
    return merged_dict


# ============================================================================
# ENSO EVENT DETECTION
# ============================================================================

def fill_small_enso_gaps(nino_index, phase='la_nina', gap_threshold_months=3):
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
    threshold = -0.5 if phase == 'la_nina' else 0.5
    at_threshold = (nino_index.values <= threshold) if phase == 'la_nina' else (nino_index.values >= threshold)
    
    # Find gaps
    i = 0
    while i < len(at_threshold):
        if at_threshold[i]:  # Start of event
            event_start = i
            
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


def detect_enso_events(nino34_anom, nino3_sst_std, nino3_pr, nino4_sst_std,
                       min_duration=5,
                       el_nino_threshold=0.5, la_nina_threshold=-0.5,
                       extreme_el_nino_sst_std_threshold=1.75,
                       extreme_el_nino_pr_threshold=5.0,
                       strong_el_nino_sst_lower=1.25,
                       strong_el_nino_sst_upper=1.75,
                       extreme_la_nina_sst_std_threshold=-1.75,
                       strong_la_nina_sst_lower=-1.75,
                       strong_la_nina_sst_upper=-1.25,
                       extreme_min_consecutive=3):
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

    def has_consecutive_months(data_slice, threshold_lower, threshold_upper=None,
                               min_consecutive=3, comparison='>='):
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
            if comparison == '<=':
                condition = data_slice.values <= threshold_lower
            elif comparison == '>=':
                condition = data_slice.values >= threshold_lower
            elif comparison == '<':
                condition = data_slice.values < threshold_lower
            elif comparison == '>':
                condition = data_slice.values > threshold_lower
            else:
                raise ValueError(f"Unsupported comparison: {comparison}")
        else:
            # Range comparison
            if comparison == '>=':
                # For El Nino: lower <= data < upper
                condition = (data_slice.values >= threshold_lower) & (data_slice.values < threshold_upper)
            elif comparison == '<=':
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

    def create_event_dict(start_idx, end_idx, nino34_data, peak_month=None, peak_date=None,
                          seasonal_ok=True, discard_reason=None):
        """Create event dictionary with all metadata."""
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_date': str(nino34_data['time'].values[start_idx])[:10],
            'end_date': str(nino34_data['time'].values[end_idx-1])[:10],
            'duration': end_idx - start_idx,
            'peak_month': peak_month,
            'peak_date': peak_date,
            'seasonal_phase_lock_ok': seasonal_ok,
            'discard_reason': discard_reason
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
                    el_nino_start, i, nino34_anom)

                if passes_seasonal:
                    event_slice_sst = nino3_sst_std.isel(time=slice(el_nino_start, i))
                    event_slice_pr = nino3_pr.isel(time=slice(el_nino_start, i))

                    # Check for Extreme El Nino (SST >= 1.75 AND precip >= 5)
                    sst_extreme = has_consecutive_months(
                        event_slice_sst, extreme_el_nino_sst_std_threshold,
                        min_consecutive=extreme_min_consecutive, comparison='>=')
                    pr_extreme = has_consecutive_months(
                        event_slice_pr, extreme_el_nino_pr_threshold,
                        min_consecutive=extreme_min_consecutive, comparison='>=')

                    # Check for Strong El Nino (1.25 <= SST < 1.75)
                    sst_strong = has_consecutive_months(
                        event_slice_sst, strong_el_nino_sst_lower, strong_el_nino_sst_upper,
                        min_consecutive=extreme_min_consecutive, comparison='>=')

                    # Check for Moderate El Nino (SST < 1.25)
                    sst_moderate = has_consecutive_months(
                        event_slice_sst, 1.25,
                        min_consecutive=extreme_min_consecutive, comparison='<')

                    event_dict = create_event_dict(el_nino_start, i, nino34_anom,
                                                   peak_month, peak_date, True, None)

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
                        event_dict['discard_reason'] = "Did not meet any El Niño classification criteria"
                        discarded_el_nino_events.append(event_dict)
                else:
                    event_dict = create_event_dict(el_nino_start, i, nino34_anom,
                                                   peak_month, peak_date, False, reason)
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
                    la_nina_start, i, nino34_anom)

                if passes_seasonal:
                    event_slice_nino4 = nino4_sst_std.isel(time=slice(la_nina_start, i))

                    # Check for Extreme La Nina (SST <= -1.75)
                    is_extreme = has_consecutive_months(
                        event_slice_nino4, extreme_la_nina_sst_std_threshold,
                        min_consecutive=extreme_min_consecutive, comparison='<=')

                    # Check for Strong La Nina (-1.75 < SST <= -1.25)
                    is_strong = has_consecutive_months(
                        event_slice_nino4, strong_la_nina_sst_lower, strong_la_nina_sst_upper,
                        min_consecutive=extreme_min_consecutive, comparison='<=')

                    # Check for Moderate La Nina (SST > -1.25)
                    is_moderate = has_consecutive_months(
                        event_slice_nino4, -1.25,
                        min_consecutive=extreme_min_consecutive, comparison='>')

                    event_dict = create_event_dict(la_nina_start, i, nino34_anom,
                                                   peak_month, peak_date, True, None)

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
                        event_dict['discard_reason'] = "Did not meet any La Niña classification criteria"
                        discarded_la_nina_events.append(event_dict)
                else:
                    event_dict = create_event_dict(la_nina_start, i, nino34_anom,
                                                   peak_month, peak_date, False, reason)
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
                el_nino_start, len(nino34_anom), nino34_anom)

            if passes_seasonal:
                event_slice_sst = nino3_sst_std.isel(time=slice(el_nino_start, None))
                event_slice_pr = nino3_pr.isel(time=slice(el_nino_start, None))

                sst_extreme = has_consecutive_months(
                    event_slice_sst, extreme_el_nino_sst_std_threshold,
                    min_consecutive=extreme_min_consecutive, comparison='>=')
                pr_extreme = has_consecutive_months(
                    event_slice_pr, extreme_el_nino_pr_threshold,
                    min_consecutive=extreme_min_consecutive, comparison='>=')

                sst_strong = has_consecutive_months(
                    event_slice_sst, strong_el_nino_sst_lower, strong_el_nino_sst_upper,
                    min_consecutive=extreme_min_consecutive, comparison='>=')

                # Check for Moderate El Nino (SST < 1.25)
                sst_moderate = has_consecutive_months(
                    event_slice_sst, 1.25,
                    min_consecutive=extreme_min_consecutive, comparison='<')

                event_dict = create_event_dict(el_nino_start, len(nino34_anom), nino34_anom,
                                               peak_month, peak_date, True, None)

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
                    event_dict['discard_reason'] = "Did not meet any El Niño classification criteria"
                    discarded_el_nino_events.append(event_dict)
            else:
                event_dict = create_event_dict(el_nino_start, len(nino34_anom), nino34_anom,
                                               peak_month, peak_date, False, reason)
                discarded_el_nino_events.append(event_dict)
        else:
            # Event did NOT complete (still > 0.5 at end) → DISCARD
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                el_nino_start, len(nino34_anom), nino34_anom)
            event_dict = create_event_dict(el_nino_start, len(nino34_anom), nino34_anom,
                                           peak_month, peak_date, False,
                                           "Incomplete: event did not drop below +0.5 threshold at end of data")
            discarded_el_nino_events.append(event_dict)

    if in_la_nina and (len(nino34_anom) - la_nina_start) >= min_duration:
        # Check if La Niña event properly COMPLETED (exited above -0.5)
        # If it reaches end of time series, it's INCOMPLETE → DISCARD
        last_value = float(nino34_anom.isel(time=-1))

        if last_value > la_nina_threshold:
            # Event properly exited → can classify
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                la_nina_start, len(nino34_anom), nino34_anom)

            if passes_seasonal:
                event_slice_nino4 = nino4_sst_std.isel(time=slice(la_nina_start, None))

                is_extreme = has_consecutive_months(
                    event_slice_nino4, extreme_la_nina_sst_std_threshold,
                    min_consecutive=extreme_min_consecutive, comparison='<=')

                is_strong = has_consecutive_months(
                    event_slice_nino4, strong_la_nina_sst_lower, strong_la_nina_sst_upper,
                    min_consecutive=extreme_min_consecutive, comparison='<=')

                # Check for Moderate La Nina (SST > -1.25)
                is_moderate = has_consecutive_months(
                    event_slice_nino4, -1.25,
                    min_consecutive=extreme_min_consecutive, comparison='>')

                event_dict = create_event_dict(la_nina_start, len(nino34_anom), nino34_anom,
                                               peak_month, peak_date, True, None)

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
                    event_dict['discard_reason'] = "Did not meet any La Niña classification criteria"
                    discarded_la_nina_events.append(event_dict)
            else:
                event_dict = create_event_dict(la_nina_start, len(nino34_anom), nino34_anom,
                                               peak_month, peak_date, False, reason)
                discarded_la_nina_events.append(event_dict)
        else:
            # Event did NOT complete (still < -0.5 at end) → DISCARD
            passes_seasonal, peak_month, peak_date, reason = check_seasonal_phase_locking(
                la_nina_start, len(nino34_anom), nino34_anom)
            event_dict = create_event_dict(la_nina_start, len(nino34_anom), nino34_anom,
                                           peak_month, peak_date, False,
                                           "Incomplete: event did not rise above -0.5 threshold at end of data")
            discarded_la_nina_events.append(event_dict)

    # Detect neutral periods
    all_counted_events = (extreme_el_nino_events + strong_el_nino_events + moderate_el_nino_events +
                          extreme_la_nina_events + strong_la_nina_events + moderate_la_nina_events)
    event_mask = np.zeros(len(nino34_anom), dtype=bool)
    for event in all_counted_events:
        event_mask[event['start_idx']:event['end_idx']] = True

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
                    'start_idx': neutral_start,
                    'end_idx': i,
                    'start_date': str(nino34_anom['time'].values[neutral_start])[:10],
                    'end_date': str(nino34_anom['time'].values[i-1])[:10],
                    'duration': i - neutral_start,
                    'peak_month': neutral_peak_month,
                    'peak_date': neutral_peak_date
                }
                neutral_periods.append(event_dict)
            in_neutral = False
            neutral_start = None

    if in_neutral and neutral_start is not None:
        neutral_peak_month, neutral_peak_date = get_peak_info(neutral_start, len(nino34_anom), nino34_anom)
        event_dict = {
            'start_idx': neutral_start,
            'end_idx': len(nino34_anom),
            'start_date': str(nino34_anom['time'].values[neutral_start])[:10],
            'end_date': str(nino34_anom['time'].values[-1])[:10],
            'duration': len(nino34_anom) - neutral_start,
            'peak_month': neutral_peak_month,
            'peak_date': neutral_peak_date
        }
        neutral_periods.append(event_dict)

    # Statistics
    total_el_nino_detected = (len(extreme_el_nino_events) + len(strong_el_nino_events) +
                              len(moderate_el_nino_events) + len(discarded_el_nino_events))
    total_la_nina_detected = (len(extreme_la_nina_events) + len(strong_la_nina_events) +
                              len(moderate_la_nina_events) + len(discarded_la_nina_events))

    statistics = {
        'total_el_nino_detected': total_el_nino_detected,
        'el_nino_counted': len(extreme_el_nino_events) + len(strong_el_nino_events) + len(moderate_el_nino_events),
        'el_nino_discarded': len(discarded_el_nino_events),
        'total_la_nina_detected': total_la_nina_detected,
        'la_nina_counted': len(extreme_la_nina_events) + len(strong_la_nina_events) + len(moderate_la_nina_events),
        'la_nina_discarded': len(discarded_la_nina_events),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"ENSO EVENT CLASSIFICATION (V5 - 7 PHASES with seasonal phase-locking)")
    print(f"{'='*70}")

    print(f"\nExtreme El Nino events: {len(extreme_el_nino_events)}")
    for event in extreme_el_nino_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nStrong El Nino events: {len(strong_el_nino_events)}")
    for event in strong_el_nino_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nModerate El Nino events: {len(moderate_el_nino_events)}")
    for event in moderate_el_nino_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nDiscarded El Nino events (failed seasonal constraint): {len(discarded_el_nino_events)}")
    for event in discarded_el_nino_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months) - {event['discard_reason']}")

    print(f"\nExtreme La Nina events: {len(extreme_la_nina_events)}")
    for event in extreme_la_nina_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nStrong La Nina events: {len(strong_la_nina_events)}")
    for event in strong_la_nina_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nModerate La Nina events: {len(moderate_la_nina_events)}")
    for event in moderate_la_nina_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months, peak month: {event['peak_month']})")

    print(f"\nDiscarded La Nina events (failed seasonal constraint): {len(discarded_la_nina_events)}")
    for event in discarded_la_nina_events:
        print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} "
              f"({event['duration']} months) - {event['discard_reason']}")

    print(f"\nNeutral periods: {len(neutral_periods)}")
    for event in neutral_periods:
        if event['duration'] >= 3:
            print(f"  {event['start_date'][:7]} to {event['end_date'][:7]} ({event['duration']} months)")

    # Time distribution
    total_months = len(nino34_anom)
    extreme_el_nino_months = sum(e['duration'] for e in extreme_el_nino_events)
    strong_el_nino_months = sum(e['duration'] for e in strong_el_nino_events)
    moderate_el_nino_months = sum(e['duration'] for e in moderate_el_nino_events)
    extreme_la_nina_months = sum(e['duration'] for e in extreme_la_nina_events)
    strong_la_nina_months = sum(e['duration'] for e in strong_la_nina_events)
    moderate_la_nina_months = sum(e['duration'] for e in moderate_la_nina_events)
    neutral_months = sum(e['duration'] for e in neutral_periods)

    print(f"\n{'='*70}")
    print(f"TIME DISTRIBUTION (counted events only)")
    print(f"{'='*70}")
    print(f"Total months: {total_months}")
    print(f"Extreme El Nino:  {extreme_el_nino_months} months ({100*extreme_el_nino_months/total_months:.1f}%)")
    print(f"Strong El Nino:   {strong_el_nino_months} months ({100*strong_el_nino_months/total_months:.1f}%)")
    print(f"Moderate El Nino: {moderate_el_nino_months} months ({100*moderate_el_nino_months/total_months:.1f}%)")
    print(f"Extreme La Nina:  {extreme_la_nina_months} months ({100*extreme_la_nina_months/total_months:.1f}%)")
    print(f"Strong La Nina:   {strong_la_nina_months} months ({100*strong_la_nina_months/total_months:.1f}%)")
    print(f"Moderate La Nina: {moderate_la_nina_months} months ({100*moderate_la_nina_months/total_months:.1f}%)")
    print(f"Neutral:          {neutral_months} months ({100*neutral_months/total_months:.1f}%)")

    print(f"\n{'='*70}")
    print(f"SEASONAL PHASE-LOCKING STATISTICS")
    print(f"{'='*70}")
    print(f"El Nino:  {statistics['el_nino_counted']} counted, "
          f"{statistics['el_nino_discarded']} discarded "
          f"(of {statistics['total_el_nino_detected']} total detected)")
    print(f"La Nina:  {statistics['la_nina_counted']} counted, "
          f"{statistics['la_nina_discarded']} discarded "
          f"(of {statistics['total_la_nina_detected']} total detected)")

    return {
        'extreme_el_nino': extreme_el_nino_events,
        'strong_el_nino': strong_el_nino_events,
        'moderate_el_nino': moderate_el_nino_events,
        'extreme_la_nina': extreme_la_nina_events,
        'strong_la_nina': strong_la_nina_events,
        'moderate_la_nina': moderate_la_nina_events,
        'neutral': neutral_periods,
        'discarded_el_nino': discarded_el_nino_events,
        'discarded_la_nina': discarded_la_nina_events,
        'statistics': statistics
    }


def fill_small_iod_gaps(dmi_3mo, phase='piod', threshold=0.4, gap_threshold_months=2):
    """
    Fill small gaps (≤ gap_threshold_months) in the IOD index so that brief
    threshold reversals during an active IOD don't fragment one event into two.

    Mirror of `fill_small_enso_gaps` but tuned for IOD: shorter event lifecycle
    (3-6 months total) means a 3-month gap can swallow most of an event, so
    default gap is 2 months. Phase = 'piod' uses >= +threshold, phase = 'niod'
    uses <= -threshold.

    Parameters
    ----------
    dmi_3mo : xr.DataArray
        DMI 3-month running mean, monthly.
    phase : {'piod', 'niod'}
    threshold : float
        DMI detection threshold (default 0.4 — Jain 2025 SOP).
    gap_threshold_months : int
        Maximum gap size to fill (default 2).

    Returns
    -------
    xr.DataArray
        Filled DMI series.
    """
    filled = dmi_3mo.copy()
    if phase == 'piod':
        at_threshold = dmi_3mo.values >= threshold
    elif phase == 'niod':
        at_threshold = dmi_3mo.values <= -threshold
    else:
        raise ValueError(f"phase must be 'piod' or 'niod', got {phase!r}")

    i = 0
    n = len(at_threshold)
    while i < n:
        if at_threshold[i]:
            # Walk to end of contiguous in-event run
            j = i + 1
            while j < n and at_threshold[j]:
                j += 1
            # Walk to end of next contiguous out-of-event gap
            gap_start = j
            gap_end = j
            while gap_end < n and not at_threshold[gap_end]:
                gap_end += 1
            gap_size = gap_end - gap_start
            if 0 < gap_size <= gap_threshold_months and gap_end < n:
                # Fill the gap with the value just before the gap so that the
                # in-event run is uninterrupted at detection time.
                fill_value = dmi_3mo.values[gap_start - 1]
                filled.values[gap_start:gap_end] = fill_value
                i = gap_end
            else:
                i = j
        else:
            i += 1
    return filled


def merge_iod_events_by_gap(events_dict, dmi_3mo, gap_months_threshold=2,
                              min_duration=3,
                              piod_threshold=0.4, niod_threshold=-0.4):
    """
    Post-process detected IOD events to merge those separated by short gaps.

    Mirror of `merge_events_by_gap` adapted for IOD's two phases (pIOD merges
    extreme + moderate sub-classes). nIOD has a single class. Merging only
    succeeds if the merged window still contains >= min_duration consecutive
    months of DMI past threshold.

    Parameters
    ----------
    events_dict : dict
        Output of `detect_iod_events`. Keys 'extreme_piod', 'moderate_piod',
        'niod' are merged. Other keys pass through unchanged.
    dmi_3mo : xr.DataArray
    gap_months_threshold : int
        Maximum gap (in months) to bridge (default 2).
    min_duration : int
        Minimum consecutive months at threshold for merged validity (default 3).
    """

    def count_consecutive(slice_vals, threshold, comparison):
        if comparison == '>=':
            cond = slice_vals >= threshold
        else:
            cond = slice_vals <= threshold
        best = 0
        run = 0
        for v in cond:
            if v:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    def merge_one_phase(event_lists_by_level, threshold, comparison):
        """event_lists_by_level: list of lists (e.g. [extreme, moderate]).
        Returns merged classification per level (highest wins)."""
        all_events = []
        for level_name, lst in zip(['extreme', 'moderate'], event_lists_by_level):
            for ev in lst:
                e = dict(ev)
                e['original_level'] = level_name
                all_events.append(e)
        if not all_events:
            return {'extreme': [], 'moderate': []}
        all_events.sort(key=lambda e: e['start_idx'])

        merged_chronological = []
        i = 0
        while i < len(all_events):
            cur = dict(all_events[i])
            j = i + 1
            while j < len(all_events):
                nxt = all_events[j]
                gap = nxt['start_idx'] - cur['end_idx']
                if 0 < gap <= gap_months_threshold:
                    new_start = cur['start_idx']
                    new_end = nxt['end_idx']
                    merged_slice = dmi_3mo.isel(time=slice(new_start, new_end))
                    if count_consecutive(merged_slice.values, threshold, comparison) >= min_duration:
                        abs_vals = np.abs(merged_slice.values)
                        peak_idx = int(np.nanargmax(abs_vals))
                        cur = {
                            'start_idx': new_start,
                            'end_idx': new_end,
                            'start_date': str(dmi_3mo.time.values[new_start])[:10],
                            'end_date': str(dmi_3mo.time.values[new_end - 1])[:10],
                            'duration': new_end - new_start,
                            'peak_month': int(merged_slice.time.dt.month.values[peak_idx]),
                            'peak_date': str(merged_slice.time.values[peak_idx])[:10],
                            'seasonal_phase_lock_ok': True,
                            'discard_reason': None,
                            'merged': True,
                            'original_levels': '+'.join(
                                e['original_level'] for e in all_events[i:j+1]
                            ),
                        }
                        j += 1
                        continue
                break
            merged_chronological.append(cur)
            i = j if j > i + 1 else i + 1

        result = {'extreme': [], 'moderate': []}
        for ev in merged_chronological:
            if ev.get('merged'):
                lvls = ev.get('original_levels', 'moderate').split('+')
                level = 'extreme' if 'extreme' in lvls else 'moderate'
            else:
                level = ev.get('original_level', 'moderate')
            result[level].append(ev)
        return result

    def merge_niod(niod_list):
        if not niod_list:
            return []
        sorted_list = sorted(niod_list, key=lambda e: e['start_idx'])
        merged = []
        i = 0
        while i < len(sorted_list):
            cur = dict(sorted_list[i])
            j = i + 1
            while j < len(sorted_list):
                nxt = sorted_list[j]
                gap = nxt['start_idx'] - cur['end_idx']
                if 0 < gap <= gap_months_threshold:
                    new_start = cur['start_idx']
                    new_end = nxt['end_idx']
                    merged_slice = dmi_3mo.isel(time=slice(new_start, new_end))
                    if count_consecutive(merged_slice.values, niod_threshold, '<=') >= min_duration:
                        abs_vals = np.abs(merged_slice.values)
                        peak_idx = int(np.nanargmin(merged_slice.values))
                        cur = {
                            'start_idx': new_start,
                            'end_idx': new_end,
                            'start_date': str(dmi_3mo.time.values[new_start])[:10],
                            'end_date': str(dmi_3mo.time.values[new_end - 1])[:10],
                            'duration': new_end - new_start,
                            'peak_month': int(merged_slice.time.dt.month.values[peak_idx]),
                            'peak_date': str(merged_slice.time.values[peak_idx])[:10],
                            'seasonal_phase_lock_ok': True,
                            'discard_reason': None,
                            'merged': True,
                        }
                        j += 1
                        continue
                break
            merged.append(cur)
            i = j if j > i + 1 else i + 1
        return merged

    piod_merged = merge_one_phase(
        [events_dict.get('extreme_piod', []), events_dict.get('moderate_piod', [])],
        piod_threshold, '>=',
    )
    niod_merged = merge_niod(events_dict.get('niod', []))

    out = dict(events_dict)
    out['extreme_piod'] = piod_merged['extreme']
    out['moderate_piod'] = piod_merged['moderate']
    out['niod'] = niod_merged
    return out


def detect_iod_events(dmi_3mo, setio_pr_anom,
                      min_duration=3,
                      piod_threshold=0.4, niod_threshold=-0.4,
                      extreme_piod_pr_threshold=-3.0,
                      extreme_min_consecutive=3,
                      require_son=True,
                      include_open_events=True):
    """
    Detect and classify IOD events into 4 phases.

    Mirrors `detect_enso_events` structure. Thresholds calibrated in results/iod_investigation/.

    Classification criteria:

    Detection (using 3-month running mean DMI):
    - pIOD: DMI_3mo >= +0.4 for 3+ consecutive months
    - nIOD: DMI_3mo <= -0.4 for 3+ consecutive months

    Classification:
    - Extreme pIOD: detection passed AND SETIO precip anomaly <= -3.0 mm/day
                     for 3+ consecutive months within the event window
    - Moderate pIOD: detection passed but precip threshold not met
    - nIOD: detection passed (single class, no sub-levels)

    Discard criteria:
    - Events that fail SON phase-locking (if require_son=True)
    - Events that reach end of time series without exiting threshold

    Seasonal phase-locking constraint (obs/model mode):
    - Event must be active during at least one SON (Sep-Oct-Nov) month

    Parameters
    ----------
    dmi_3mo : xr.DataArray
        Dipole Mode Index (WTIO - SETIO SST anomaly), 3-month running mean,
        monthly frequency, dim 'time'.
    setio_pr_anom : xr.DataArray
        Precipitation anomaly over SETIO box (90E-110E, 10S-0), monthly,
        mm/day, dim 'time', aligned with dmi_3mo.
    min_duration : int
        Minimum months at threshold for event detection (default 3).
    piod_threshold, niod_threshold : float
        DMI detection thresholds (default +/- 0.4, Jain 2025 SOP).
    extreme_piod_pr_threshold : float
        SETIO precip anomaly threshold for extreme pIOD (default -3.0 mm/day).
    extreme_min_consecutive : int
        Months precip must consecutively satisfy threshold (default 3 for
        obs/model; set to 2 for forecast/hindcast).
    require_son : bool
        If True, event must include >=1 SON month. Default True for obs/model;
        set False for forecast mode.

    Returns
    -------
    dict with keys:
        'extreme_piod', 'moderate_piod', 'niod', 'neutral',
        'discarded_piod', 'discarded_niod', 'statistics'
    Each event entry mirrors the `create_event_dict` schema used by ENSO.
    """

    def has_consecutive_months(data_slice, threshold, min_consecutive=3,
                                comparison='>='):
        if len(data_slice) < min_consecutive:
            return False
        if comparison == '>=':
            cond = data_slice.values >= threshold
        elif comparison == '<=':
            cond = data_slice.values <= threshold
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")
        run = 0
        best = 0
        for v in cond:
            if v:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best >= min_consecutive

    def check_son_phase_locking(start_idx, end_idx, dmi_data):
        """Event must include >=1 month in SON. Returns (ok, peak_month, peak_date, reason)."""
        event_slice = dmi_data.isel(time=slice(start_idx, end_idx))
        if len(event_slice) == 0:
            return False, None, None, "Empty event"
        months = event_slice.time.dt.month.values
        abs_vals = np.abs(event_slice.values)
        peak_idx = int(np.nanargmax(abs_vals))
        peak_month = int(months[peak_idx])
        peak_date = str(event_slice.time.values[peak_idx])[:10]
        has_son = any(m in (9, 10, 11) for m in months)
        if has_son:
            return True, peak_month, peak_date, "OK"
        return False, peak_month, peak_date, "Not active during any SON season"

    def create_event_dict(start_idx, end_idx, dmi_data,
                           peak_month=None, peak_date=None,
                           seasonal_ok=True, discard_reason=None):
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_date': str(dmi_data['time'].values[start_idx])[:10],
            'end_date': str(dmi_data['time'].values[end_idx - 1])[:10],
            'duration': end_idx - start_idx,
            'peak_month': peak_month,
            'peak_date': peak_date,
            'seasonal_phase_lock_ok': seasonal_ok,
            'discard_reason': discard_reason,
        }

    # ── event lists ────────────────────────────────────────────────────────
    extreme_piod_events = []
    moderate_piod_events = []
    niod_events = []
    neutral_periods = []
    discarded_piod_events = []
    discarded_niod_events = []

    in_piod = False
    in_niod = False
    piod_start = None
    niod_start = None

    # ── main detection loop ───────────────────────────────────────────────
    for i in range(len(dmi_3mo)):
        current_val = float(dmi_3mo.isel(time=i))

        # pIOD detection
        if current_val >= piod_threshold:
            if not in_piod:
                in_piod = True
                piod_start = i
        elif in_piod:
            if (i - piod_start) >= min_duration:
                if require_son:
                    passes_son, peak_month, peak_date, reason = check_son_phase_locking(
                        piod_start, i, dmi_3mo)
                else:
                    passes_son = True
                    event_slice_dmi = dmi_3mo.isel(time=slice(piod_start, i))
                    peak_idx = int(np.nanargmax(event_slice_dmi.values))
                    peak_month = int(event_slice_dmi.time.dt.month.values[peak_idx])
                    peak_date = str(event_slice_dmi.time.values[peak_idx])[:10]
                    reason = "OK"

                if passes_son:
                    event_slice_pr = setio_pr_anom.isel(time=slice(piod_start, i))
                    pr_extreme = has_consecutive_months(
                        event_slice_pr, extreme_piod_pr_threshold,
                        min_consecutive=extreme_min_consecutive, comparison='<=')
                    event_dict = create_event_dict(piod_start, i, dmi_3mo,
                                                    peak_month, peak_date, True, None)
                    if pr_extreme:
                        extreme_piod_events.append(event_dict)
                    else:
                        moderate_piod_events.append(event_dict)
                else:
                    event_dict = create_event_dict(piod_start, i, dmi_3mo,
                                                    peak_month, peak_date, False, reason)
                    discarded_piod_events.append(event_dict)
            in_piod = False
            piod_start = None

        # nIOD detection
        if current_val <= niod_threshold:
            if not in_niod:
                in_niod = True
                niod_start = i
        elif in_niod:
            if (i - niod_start) >= min_duration:
                if require_son:
                    passes_son, peak_month, peak_date, reason = check_son_phase_locking(
                        niod_start, i, dmi_3mo)
                else:
                    passes_son = True
                    event_slice_dmi = dmi_3mo.isel(time=slice(niod_start, i))
                    peak_idx = int(np.nanargmin(event_slice_dmi.values))
                    peak_month = int(event_slice_dmi.time.dt.month.values[peak_idx])
                    peak_date = str(event_slice_dmi.time.values[peak_idx])[:10]
                    reason = "OK"

                if passes_son:
                    event_dict = create_event_dict(niod_start, i, dmi_3mo,
                                                    peak_month, peak_date, True, None)
                    niod_events.append(event_dict)
                else:
                    event_dict = create_event_dict(niod_start, i, dmi_3mo,
                                                    peak_month, peak_date, False, reason)
                    discarded_niod_events.append(event_dict)
            in_niod = False
            niod_start = None

    # ── End-of-series finalization (open events) ─────────────────────────
    # If the time series ends while an event is still active, `in_piod` /
    # `in_niod` remain True. By default we keep these "open" events so that
    # ongoing IODs at the operational month are not silently dropped. The
    # same criteria (min_duration, SON gate, extreme classification) apply.
    # Set include_open_events=False to drop them (ENSO obs legacy behaviour).
    n_total = len(dmi_3mo)
    if include_open_events:
        if in_piod and (n_total - piod_start) >= min_duration:
            i = n_total  # event closes at end-of-series
            if require_son:
                passes_son, peak_month, peak_date, reason = check_son_phase_locking(
                    piod_start, i, dmi_3mo)
            else:
                passes_son = True
                event_slice_dmi = dmi_3mo.isel(time=slice(piod_start, i))
                peak_idx = int(np.nanargmax(event_slice_dmi.values))
                peak_month = int(event_slice_dmi.time.dt.month.values[peak_idx])
                peak_date = str(event_slice_dmi.time.values[peak_idx])[:10]
                reason = "OK (open at end-of-series)"
            if passes_son:
                event_slice_pr = setio_pr_anom.isel(time=slice(piod_start, i))
                pr_extreme = has_consecutive_months(
                    event_slice_pr, extreme_piod_pr_threshold,
                    min_consecutive=extreme_min_consecutive, comparison='<=')
                event_dict = create_event_dict(piod_start, i, dmi_3mo,
                                                peak_month, peak_date, True, None)
                event_dict['open_ended'] = True
                if pr_extreme:
                    extreme_piod_events.append(event_dict)
                else:
                    moderate_piod_events.append(event_dict)
            else:
                event_dict = create_event_dict(piod_start, i, dmi_3mo,
                                                peak_month, peak_date, False, reason)
                event_dict['open_ended'] = True
                discarded_piod_events.append(event_dict)

        if in_niod and (n_total - niod_start) >= min_duration:
            i = n_total
            if require_son:
                passes_son, peak_month, peak_date, reason = check_son_phase_locking(
                    niod_start, i, dmi_3mo)
            else:
                passes_son = True
                event_slice_dmi = dmi_3mo.isel(time=slice(niod_start, i))
                peak_idx = int(np.nanargmin(event_slice_dmi.values))
                peak_month = int(event_slice_dmi.time.dt.month.values[peak_idx])
                peak_date = str(event_slice_dmi.time.values[peak_idx])[:10]
                reason = "OK (open at end-of-series)"
            if passes_son:
                event_dict = create_event_dict(niod_start, i, dmi_3mo,
                                                peak_month, peak_date, True, None)
                event_dict['open_ended'] = True
                niod_events.append(event_dict)
            else:
                event_dict = create_event_dict(niod_start, i, dmi_3mo,
                                                peak_month, peak_date, False, reason)
                event_dict['open_ended'] = True
                discarded_niod_events.append(event_dict)

    # ── neutral period stitching ──────────────────────────────────────────
    # Everything not inside a detected pIOD/nIOD active window is neutral.
    active_mask = np.zeros(n_total, dtype=bool)
    for events in (extreme_piod_events, moderate_piod_events, niod_events):
        for ev in events:
            active_mask[ev['start_idx']:ev['end_idx']] = True
    i = 0
    while i < n_total:
        if not active_mask[i]:
            j = i
            while j < n_total and not active_mask[j]:
                j += 1
            if (j - i) >= min_duration:
                neutral_periods.append(create_event_dict(i, j, dmi_3mo,
                                                          None, None, True, None))
            i = j
        else:
            i += 1

    statistics = {
        'total_piod_detected': len(extreme_piod_events) + len(moderate_piod_events) +
                                len(discarded_piod_events),
        'piod_counted': len(extreme_piod_events) + len(moderate_piod_events),
        'piod_discarded': len(discarded_piod_events),
        'total_niod_detected': len(niod_events) + len(discarded_niod_events),
        'niod_counted': len(niod_events),
        'niod_discarded': len(discarded_niod_events),
    }

    return {
        'extreme_piod': extreme_piod_events,
        'moderate_piod': moderate_piod_events,
        'niod': niod_events,
        'neutral': neutral_periods,
        'discarded_piod': discarded_piod_events,
        'discarded_niod': discarded_niod_events,
        'statistics': statistics,
    }


# ============================================================================
# POPULATION ANALYSIS
# ============================================================================

def load_and_regrid_population(pop_file, target_lons, target_lats):
    """
    Load 1km population raster and regrid to CESM grid using conservative interpolation.

    Conservative interpolation: sum all high-res population within each coarse grid cell.

    Parameters:
    -----------
    pop_file : str
        Path to population GeoTIFF
    target_lons : array
        CESM longitude grid (0-360)
    target_lats : array
        CESM latitude grid (-90 to 90)

    Returns:
    --------
    pop_regrid : xr.DataArray
        Population on CESM grid
    """
    print(f"Loading population data from {pop_file}...")

    # Read population raster
    with rasterio.open(pop_file) as src:
        pop_data = src.read(1)  # Read first band
        transform = src.transform

        # Get coordinate arrays for each pixel
        height, width = pop_data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to geographic coordinates
        lons_1km, lats_1km = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        lons_1km = np.array(lons_1km)
        lats_1km = np.array(lats_1km)
        pop_flat = pop_data.flatten()

        # Filter out nodata values
        nodata = src.nodata
        if nodata is not None:
            valid_mask = pop_flat != nodata
        else:
            valid_mask = ~np.isnan(pop_flat)

        lons_1km = lons_1km[valid_mask]
        lats_1km = lats_1km[valid_mask]
        pop_flat = pop_flat[valid_mask]

        # Filter negative population values
        valid_mask = pop_flat >= 0
        lons_1km = lons_1km[valid_mask]
        lats_1km = lats_1km[valid_mask]
        pop_flat = pop_flat[valid_mask]

    print(f"  Population raster shape: {pop_data.shape}")
    print(f"  Valid population pixels: {len(pop_flat):,}")
    print(f"  Total population: {pop_flat.sum():,.0f}")

    # Convert longitude from -180/180 to 0/360 if needed
    if lons_1km.min() < 0:
        lons_1km = (lons_1km + 360) % 360

    # Create bin edges for CESM grid
    # Assume regular spacing and create cell boundaries
    lon_spacing = np.diff(target_lons).mean()
    lat_spacing = np.diff(target_lats).mean()

    lon_edges = np.concatenate([
        [target_lons[0] - lon_spacing/2],
        (target_lons[:-1] + target_lons[1:]) / 2,
        [target_lons[-1] + lon_spacing/2]
    ])

    lat_edges = np.concatenate([
        [target_lats[0] - lat_spacing/2],
        (target_lats[:-1] + target_lats[1:]) / 2,
        [target_lats[-1] + lat_spacing/2]
    ])

    print(f"  Regridding to CESM grid ({len(target_lats)} x {len(target_lons)})...")

    # Conservative regridding: sum population in each CESM grid cell
    pop_regrid, _, _, _ = binned_statistic_2d(
        lats_1km, lons_1km, pop_flat,
        statistic='sum',
        bins=[lat_edges, lon_edges]
    )

    # Handle bins with no data (set to 0 instead of NaN)
    pop_regrid = np.nan_to_num(pop_regrid, nan=0.0)

    print(f"  Regridded total population: {pop_regrid.sum():,.0f}")

    # Create xarray DataArray
    pop_xr = xr.DataArray(
        pop_regrid,
        coords={'lat': target_lats, 'lon': target_lons},
        dims=['lat', 'lon'],
        attrs={'units': 'people', 'source': 'FuturePop SSP3 2025'}
    )

    return pop_xr


# ============================================================================
# PROBABILITY RATIO ANALYSIS
# ============================================================================

def probability_ratio_analysis(phase_events, neutral_events,
                                percentile_threshold=0.90,
                                pr_increase_threshold=1.3,
                                pr_decrease_threshold=0.7,
                                value_key='rx1day',
                                tail='upper'):
    """
    Compute probability ratio for upper/lower-tail events relative to neutral.

    This approach asks: "Does this ENSO phase make extreme rainfall more likely?"
    rather than requiring individual events to agree on sign.

    Parameters:
    -----------
    phase_events : list of dicts with index DataArrays
        Events from the ENSO phase being analyzed
    neutral_events : list of dicts with index DataArrays
        Neutral period events for baseline
    percentile_threshold : float
        Percentile of neutral distribution to use as threshold (default 90th)
    pr_increase_threshold : float
        PR above this = robust increase in extreme rainfall (default 1.3 = +30%)
    pr_decrease_threshold : float
        PR below this = robust decrease in extreme rainfall (default 0.7 = -30%)
    value_key : str
        Dictionary key used to extract the event index from each record.
        Examples: 'rx1day', 'rx10day'.
    tail : str
        Tail to evaluate. 'upper' uses exceedance (> threshold),
        'lower' uses non-exceedance (<= threshold). Default: 'upper'.

    Returns:
    --------
    prob_ratio : xr.DataArray - probability ratio at each pixel
    robust_increase : xr.DataArray (bool) - pixels with robust increase
    robust_decrease : xr.DataArray (bool) - pixels with robust decrease
    p_exceed_phase : xr.DataArray - exceedance probability during ENSO phase
    p_exceed_neutral : xr.DataArray - exceedance probability during neutral
    threshold : xr.DataArray - the threshold value (90th percentile of neutral)
    mean_phase : xr.DataArray - mean RX1DAY during ENSO phase
    mean_neutral : xr.DataArray - mean RX1DAY during neutral
    mean_anomaly : xr.DataArray - difference (mean_phase - mean_neutral)
    """
    print(f"    Computing probability ratio analysis...")
    print(f"    Phase events: {len(phase_events)}, Neutral events: {len(neutral_events)}")

    # Stack all events
    phase_rx1day = xr.concat([e[value_key] for e in phase_events], dim='event')
    neutral_rx1day = xr.concat([e[value_key] for e in neutral_events], dim='event')

    # Compute mean RX1DAY for each phase (for the absolute pattern figure)
    mean_phase = phase_rx1day.mean(dim='event')
    mean_neutral = neutral_rx1day.mean(dim='event')
    mean_anomaly = mean_phase - mean_neutral

    if tail not in {'upper', 'lower'}:
        raise ValueError("tail must be 'upper' or 'lower'")

    # Threshold = percentile of neutral distribution (at each pixel)
    threshold = neutral_rx1day.quantile(percentile_threshold, dim='event')

    # Event probability in each phase
    if tail == 'upper':
        p_exceed_phase = (phase_rx1day > threshold).mean(dim='event')
        p_exceed_neutral = (neutral_rx1day > threshold).mean(dim='event')
    else:
        p_exceed_phase = (phase_rx1day <= threshold).mean(dim='event')
        p_exceed_neutral = (neutral_rx1day <= threshold).mean(dim='event')

    # Probability ratio.
    # Use a floor on p_exceed_neutral so that pixels where the neutral
    # distribution is so discrete that only 1–2 events exceed the P90
    # threshold don't produce artificially extreme ratios (e.g. 1000%+).
    # A floor of 0.05 means we require at least ~5% of neutral events to
    # exceed the threshold before trusting the ratio; pixels below this
    # floor are set to NaN (shown as grey / no data on the map).
    p_neutral_floor = 0.05
    valid_mask = p_exceed_neutral >= p_neutral_floor
    safe_neutral = xr.where(valid_mask, p_exceed_neutral, np.nan)
    prob_ratio = xr.where(valid_mask, p_exceed_phase / safe_neutral, np.nan)

    # Robust masks
    robust_increase = prob_ratio > pr_increase_threshold
    robust_decrease = prob_ratio < pr_decrease_threshold

    n_inc = int(robust_increase.sum())
    n_dec = int(robust_decrease.sum())
    print(f"    Robust increase (PR>{pr_increase_threshold}): {n_inc} pixels")
    print(f"    Robust decrease (PR<{pr_decrease_threshold}): {n_dec} pixels")

    return (prob_ratio, robust_increase, robust_decrease,
            p_exceed_phase, p_exceed_neutral, threshold,
            mean_phase, mean_neutral, mean_anomaly)
