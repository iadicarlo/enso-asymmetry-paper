"""
ENSO Nonstationarity Analysis - Step 1: Preprocessing & Event Detection

Detects ENSO events across all ensemble members, assigns GWI warming bins,
and saves results for downstream figure generation.

This script runs ONCE and provides data for all subsequent figure scripts.

HARMONIZED WITH functions.py - Uses exact same workflow as monolithic script.

Output:
    - enso_events_all_members.csv: Event catalog with metadata (member_id column for filtering)
    - regional_timeseries_all_members.nc: Regional indices (member dimension, 12 variables)
      * 3 regions (nino3, nino34, nino4) × 4 components per region:
        - raw_sst: Original SST before any processing
        - detrended: ε = after GWI detrending (before seasonal cycle removal)
        - anom: ε* = after seasonal cycle removal (30-yr rolling window)
        - std: Standardized ε* (using rolling std)
    - precipitation_timeseries_all_members.nc: Niño-3 precipitation ANOMALY ONLY (member dimension)
    - gwi_timeseries_all_members.nc: GWI for all members (member dimension)
    - preprocessing_config.json: Metadata

For visualization purposes:
  - raw_sst vs detrended shows the trend component: α(lon,lat) × GMT(t) + β(lon,lat)
  - detrended vs anom shows the seasonal cycle being removed
  - anom vs std shows the standardization using rolling std
  
Usage in figure scripts:
    # Load events and filter
    events = pd.read_csv('enso_events_all_members.csv')
    el_nino_box3 = events[(events['phase'] == 'extreme_el_nino') & (events['gwi_box'] == 3)]
    
    # Load timeseries and select member or compute ensemble mean
    ds = xr.open_dataset('regional_timeseries_all_members.nc')
    member_ts = ds.sel(member='cmip6_1001_001')        # Single member
    ensemble_mean = ds.mean(dim='member')              # Ensemble mean
    
    # Extract event data using indices from events CSV
    for _, event in el_nino_box3.iterrows():
        member_data = ds.sel(member=event['member_id'])
        event_ts = member_data.isel(time=slice(event['start_idx'], event['end_idx']))
    
    # Visualize detrending components
    nino3_raw = member_ts['nino3_raw_sst']
    nino3_detrended = member_ts['nino3_detrended']
    trend_removed = nino3_raw - nino3_detrended  # This is α×GMT + β

Performance optimizations:
    - Uses integer slicing instead of date comparisons (10-100x faster)
    - Pre-computes DJF mask once per member
    - Threading backend for S3 I/O parallelization
    - Loads only required spatial subsets (tropical regions for SST/precip)

Author: ENSO Impacts Project  
Date: February 2026
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import s3fs
import datetime
import argparse
import logging
import sys
from joblib import Parallel, delayed
import json

# Import utility functions (GROUND TRUTH)
# Ensure script can run directly (not as package)
analysis_dir = Path(__file__).resolve().parents[1]
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

from functions import (
    get_available_members,
    compute_gwi,
    lowpass_filter_gwi,
    remove_gwi_residuals_vectorized,
    remove_seasonal_cycle_rolling_with_std,
    detect_enso_events,
    reindex_time
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Preprocess ENSO events for all ensemble members')
parser.add_argument('--n_members', type=int, default=50,
                    help='Number of ensemble members to process')
parser.add_argument('--n_jobs', type=int, default=8,
                    help='Number of parallel workers')
args = parser.parse_args()

# Setup logging
log_dir = Path('/path/to/enso_impacts/logs/analysis')
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'preprocess_events_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout (not stderr)
    ]
)
logger = logging.getLogger(__name__)

output_dir = Path('/path/to/enso_impacts/results/enso_nonstationarity')
output_dir.mkdir(parents=True, exist_ok=True)

# Data configuration (HARMONIZED WITH MONOLITHIC SCRIPT)
SCENARIOS = ['historical', 'ssp370']
START_YEARS = [1850, 2015]
END_YEARS = [2014, 2100]
S3_BUCKET = 'ncar-cesm2-lens'
S3_COMPONENT = 'atm'
S3_FORCING = 'cmip6'

# NEW GWI BINNING (3 bins instead of 4)
# Bin boundaries: (inclusive lower, exclusive upper] except last bin
GWI_BOX_EDGES = [0.0, 1.0, 2.0, 10.0]
GWI_BOX_LABELS = [
    'Box 1: GWI ≤ 1°C', 
    'Box 2: 1.0°C < GWI < 2.0°C',
    'Box 3: GWI ≥ 2.0°C'
]

# Regional definitions (EXACT MATCH WITH MONOLITHIC)
NINO_REGIONS = {
    'nino3': {'lat': (-5, 5), 'lon': (210, 270)},
    'nino34': {'lat': (-5, 5), 'lon': (190, 240)},
    'nino4': {'lat': (-5, 5), 'lon': (160, 210)}
}

logger.info("="*70)
logger.info("ENSO EVENT PREPROCESSING - STEP 1")
logger.info("="*70)
logger.info(f"Members: {args.n_members}")
logger.info(f"Parallel workers: {args.n_jobs}")
logger.info(f"GWI bins: {len(GWI_BOX_LABELS)}")
logger.info(f"  {GWI_BOX_LABELS}")
logger.info(f"Output: {output_dir}")
logger.info("="*70)

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def assign_gwi_box(gwi_value, gwi_edges):
    """
    Assign GWI box using new 3-bin structure.
    
    Box boundaries (consistent with user request):
    - Box 1: gwi <= 1°C
    - Box 2: 1.0°C < gwi < 2.0°C
    - Box 3: gwi >= 2.0°C
    
    Args:
        gwi_value: GWI value relative to 1850-1900 baseline (DJF mean)
        gwi_edges: [0.0, 1.0, 2.0, 10.0]
    
    Returns:
        int: Box number (1-3) or 4 for outliers (should be rare)
    """
    if gwi_value <= gwi_edges[1]:
        return 1  # gwi <= 1.0
    elif gwi_value <= gwi_edges[2]:
        return 2  # 1.0 < gwi <= 2.0
    elif gwi_value < gwi_edges[3]:
        return 3  # 2.0 < gwi < 10.0
    else:
        return 4  # gwi >= 10.0 


def process_member(member_idx, member_str, total_members):
    """
    Process single ensemble member using EXACT workflow from monolithic script.
    
    Steps (harmonized with functions.py):
    1. Load TREFHT → compute GWI → baseline relative to 1850-1900
    2. Load TS (SST) tropical region → detrend with GWI
    3. Extract Niño 3/3.4/4 regional means
    4. Apply remove_seasonal_cycle_rolling_with_std → get anomaly + standardized + rolling_std
    5. Load PRECC + PRECL for Niño-3 → detrend → seasonal cycle removal → anomaly
    6. Detect events using detect_enso_events(nino34_anom, nino3_std, nino3_pr, nino4_std)
    7. Filter neutral events (duration >= 5 months)
    8. Assign GWI boxes
    
    Returns:
        member_data: dict with all timeseries and events
    """
    logger.info(f"\n[Worker {member_idx}/{total_members}] Starting {member_str}")
    
    try:
        # Initialize S3 filesystem
        fs = s3fs.S3FileSystem(anon=True)
        
        # ========== Load TREFHT (for GWI computation) ==========
        trefht_data_list = []
        for scenario, start, end in zip(SCENARIOS, START_YEARS, END_YEARS):
            trefht_zarr_path = f"s3://{S3_BUCKET}/{S3_COMPONENT}/monthly/cesm2LE-{scenario}-{S3_FORCING}-TREFHT.zarr"
            trefht_store = s3fs.S3Map(root=trefht_zarr_path, s3=fs, check=False)
            ds_trefht = xr.open_zarr(trefht_store, consolidated=True)
            trefht = ds_trefht['TREFHT'].sel(member_id=member_str, time=slice(f'{start}', f'{end}'))
            trefht_data_list.append(trefht)
        
        trefht_combined = xr.concat(trefht_data_list, dim='time')
        trefht_combined = trefht_combined - 273.15  # Convert K → °C
        trefht_combined = reindex_time(trefht_combined)
        
        # Compute GWI and baseline to 1850-1900 = 0
        gwi_monthly = compute_gwi(trefht_combined)
        gwi_baseline = gwi_monthly.sel(time=slice('1850', '1900')).mean()
        gwi_relative = gwi_monthly - gwi_baseline
        
        # ✓ REQUIREMENT: 1850-1900 baseline now exactly 0
        # Verify baseline is correct
        baseline_check = gwi_relative.sel(time=slice('1850', '1900')).mean().values
        if abs(baseline_check) > 1e-10:
            logger.warning(f"  WARNING: GWI baseline not zero: {baseline_check:.2e}°C (small floating point error)")
        else:
            logger.info(f"  ✓ GWI baseline (1850-1900) set to 0.0000°C")
        
        # Apply polynomial smoothing to GMST (order 4, removes seasonal + decadal variability)
        # This isolates ONLY the forced anthropogenic warming trend
        # Rationale: Global seasonality/variability ≠ equatorial seasonality/variability
        gwi_smooth = lowpass_filter_gwi(gwi_relative, polynomial_degree=4)
        
        # ✓ VERIFICATION: Check polynomial fit quality
        residual_std = (gwi_relative - gwi_smooth).std().values
        signal_std = gwi_relative.std().values
        smoothness_ratio = residual_std / signal_std
        
        logger.info(f"  Loaded TREFHT: {trefht_combined.shape}")
        logger.info(f"  GWI baseline (1850-1900): {gwi_baseline.values:.3f}°C")
        logger.info(f"  Applied polynomial smoothing (degree 4)")
        logger.info(f"  ✓ Smoothness: {smoothness_ratio:.1%} residual/signal (removes seasonal + decadal)")
        logger.info(f"  ✓ No NaN values (polynomial defined everywhere)")
        
        # ========== Load TS (SST) - tropical region only ==========
        ts_data_list = []
        for scenario, start, end in zip(SCENARIOS, START_YEARS, END_YEARS):
            ts_zarr_path = f"s3://{S3_BUCKET}/{S3_COMPONENT}/monthly/cesm2LE-{scenario}-{S3_FORCING}-TS.zarr"
            ts_store = s3fs.S3Map(root=ts_zarr_path, s3=fs, check=False)
            ds_ts = xr.open_zarr(ts_store, consolidated=True)
            # OPTIMIZATION: Load only tropical region
            ts = ds_ts['TS'].sel(
                member_id=member_str,
                lat=slice(-40, 40),
                lon=slice(40, 320),
                time=slice(f'{start}', f'{end}')
            )
            ts_data_list.append(ts)
        
        sst_combined = xr.concat(ts_data_list, dim='time')
        sst_combined = sst_combined - 273.15  # Convert K → °C
        sst_combined = reindex_time(sst_combined)
        # Detrend using SMOOTHED GWI (polynomial fit, removes forced trend only)
        sst_detrended = remove_gwi_residuals_vectorized(sst_combined, gwi_smooth)
        
        logger.info(f"  Loaded SST: {sst_combined.shape}")
        
        # ========== Extract regional indices (with standardization) ==========
        # Store all components: raw SST, detrended (ε), anomaly (ε*), standardized
        regional_indices = {}
        regional_sst_components = {}  # For storing raw vs detrended for visualization
        
        for region_name, bounds in NINO_REGIONS.items():
            # Extract region from DETRENDED SST
            region_sst_detrended = sst_detrended.sel(
                lat=slice(bounds['lat'][0], bounds['lat'][1]),
                lon=slice(bounds['lon'][0], bounds['lon'][1])
            )
            region_index_detrended = region_sst_detrended.mean(dim=['lat', 'lon'])
            
            # Also extract from ORIGINAL SST for comparison
            region_sst_raw = sst_combined.sel(
                lat=slice(bounds['lat'][0], bounds['lat'][1]),
                lon=slice(bounds['lon'][0], bounds['lon'][1])
            )
            region_index_raw = region_sst_raw.mean(dim=['lat', 'lon'])
            
            # Apply rolling seasonal cycle removal (returns dict with raw, anomaly, standardized, rolling_std)
            result = remove_seasonal_cycle_rolling_with_std(region_index_detrended, window_years=30, 
                                                             rolling_std=False, baseline_period=(1980, 2014))   
            regional_indices[region_name] = {
                'raw_sst': region_index_raw,           # Original SST before any processing
                'detrended': result['raw'],             # ε: Detrended SST before seasonal cycle removal
                'anomaly': result['anomaly'],           # ε*: After seasonal cycle removal
                'standardized': result['standardized']  # ε* / σ
            }
            
            logger.info(f"    {region_name}: raw_sst, detrended (ε), anom (ε*), std")
        
        
        # ========== Load precipitation (PRECC + PRECL) - Niño-3 region only ==========
        precc_data_list = []
        precl_data_list = []
        for scenario, start, end in zip(SCENARIOS, START_YEARS, END_YEARS):
            precc_zarr_path = f"s3://{S3_BUCKET}/{S3_COMPONENT}/monthly/cesm2LE-{scenario}-{S3_FORCING}-PRECC.zarr"
            precc_store = s3fs.S3Map(root=precc_zarr_path, s3=fs, check=False)
            ds_precc = xr.open_zarr(precc_store, consolidated=True)
            precc = ds_precc['PRECC'].sel(
                member_id=member_str,
                lat=slice(-5, 5), lon=slice(210, 270),
                time=slice(f'{start}', f'{end}')
            )
            precc_data_list.append(precc)
            
            precl_zarr_path = f"s3://{S3_BUCKET}/{S3_COMPONENT}/monthly/cesm2LE-{scenario}-{S3_FORCING}-PRECL.zarr"
            precl_store = s3fs.S3Map(root=precl_zarr_path, s3=fs, check=False)
            ds_precl = xr.open_zarr(precl_store, consolidated=True)
            precl = ds_precl['PRECL'].sel(
                member_id=member_str,
                lat=slice(-5, 5), lon=slice(210, 270),
                time=slice(f'{start}', f'{end}')
            )
            precl_data_list.append(precl)
        
        prect_combined = xr.concat([precc + precl for precc, precl in zip(precc_data_list, precl_data_list)], dim='time')
        prect_combined = prect_combined * 86400 * 1000  # Convert m/s → mm/day
        prect_combined = reindex_time(prect_combined)
        # Detrend precipitation using SMOOTHED GWI
        precip_detrended = remove_gwi_residuals_vectorized(prect_combined, gwi_smooth)
        nino3_pr_index = precip_detrended.mean(dim=['lat', 'lon'])
        pr_result = remove_seasonal_cycle_rolling_with_std(nino3_pr_index, window_years=30)
        
        logger.info(f"  Loaded precipitation: {prect_combined.shape}")
        
        # ========== Detect ENSO events (with polynomial-smoothed GWI) ==========
        # Note: Polynomial fit produces no NaN values (defined everywhere)
        # Minimal verification for data quality
        nan_counts = {
            'nino34': regional_indices['nino34']['standardized'].isnull().sum().values,
            'nino3_sst': regional_indices['nino3']['standardized'].isnull().sum().values,
            'nino3_pr': pr_result['anomaly'].isnull().sum().values,
            'nino4': regional_indices['nino4']['standardized'].isnull().sum().values,
        }
        
        if sum(nan_counts.values()) > 50:  # Allow small edge effects from seasonal removal
            logger.warning(f"  WARNING: Significant NaN detected (>50 months): {nan_counts}")
        
        events_dict = detect_enso_events(
            nino34_anom=regional_indices['nino34']['standardized'], # Standardized Niño-3.4 anomaly used for event detection instead of raw anomaly like previously
            nino3_sst_std=regional_indices['nino3']['standardized'],
            nino3_pr=pr_result['anomaly'], # Anomaly used for precipitation in event detection -> will maybe consider using standardized in future if needed but for now keep consistent with monolithic script
            nino4_sst_std=regional_indices['nino4']['standardized'],
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
            extreme_min_consecutive=3
        )
        
        # Filter neutral events (minimum duration)
        events_dict['neutral'] = [e for e in events_dict['neutral'] if e['duration'] >= 5]
        
        # ========== Assign GWI boxes (use SMOOTHED GWI) ==========
        # Pre-compute DJF mask once (much faster than repeated date comparisons)
        time_months = gwi_smooth.time.dt.month.values
        djf_mask = np.isin(time_months, [12, 1, 2])
        gwi_data = gwi_smooth.values  # Use polynomial-smoothed GWI for box assignment
        
        # Assign GWI box to each event using DJF mean
        all_events = []
        for phase_name in ['extreme_el_nino', 'strong_el_nino', 'moderate_el_nino',
                          'extreme_la_nina', 'strong_la_nina', 'moderate_la_nina', 'neutral']:
            for event in events_dict[phase_name]:
                # Use event indices directly (much faster than date comparisons)
                start_idx = event['start_idx']
                end_idx = event['end_idx']
                
                # Slice GWI data and apply pre-computed DJF mask
                event_gwi = gwi_data[start_idx:end_idx]
                event_djf_mask = djf_mask[start_idx:end_idx]
                gwi_djf_values = event_gwi[event_djf_mask]
                
                if len(gwi_djf_values) > 0:
                    # Compute DJF mean GWI
                    event['gwi_djf_mean_relative'] = float(np.mean(gwi_djf_values))
                    gwi_val = event['gwi_djf_mean_relative']
                    
                    # Assign to new 4-bin structure
                    event['gwi_box'] = assign_gwi_box(gwi_val, GWI_BOX_EDGES)
                    event['phase'] = phase_name
                    event['member_id'] = member_str
                    all_events.append(event)
                # Events without DJF months are excluded
        
        logger.info(f"  [Worker {member_idx}/{total_members}] ✓ {member_str} complete ({len(all_events)} events)")
        
        return {
            'member_id': member_str,
            'regional_indices': regional_indices,
            'precipitation': pr_result,
            'gwi_relative': gwi_relative,
            'events': all_events
        }
        
    except Exception as e:
        logger.error(f"  [Worker {member_idx}/{total_members}] ✗ {member_str} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

logger.info("\nFetching available ensemble members...")
members = get_available_members(
    bucket=S3_BUCKET,
    forcing=S3_FORCING,
    scenario='historical'
)[:args.n_members]

logger.info(f"Processing {len(members)} members: {members[:5]}... (showing first 5)\n")

# Parallel processing
logger.info("="*70)
logger.info(f"PARALLEL PROCESSING: {len(members)} members using {args.n_jobs} threads")
logger.info("="*70)

results = Parallel(n_jobs=args.n_jobs, backend='threading')(
    delayed(process_member)(i+1, member, len(members))
    for i, member in enumerate(members)
)

# Filter successful results
results = [r for r in results if r is not None]
logger.info(f"\n✓ Successfully processed {len(results)}/{len(members)} members")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("SAVING OUTPUTS")
logger.info("="*70)

# 1. Save all events as CSV
all_events = []
for result in results:
    all_events.extend(result['events'])

events_df = pd.DataFrame(all_events)
events_csv = output_dir / 'enso_events_all_members.csv'
events_df.to_csv(events_csv, index=False)
logger.info(f"✓ Saved events catalog: {events_csv}")
logger.info(f"  Total events: {len(events_df)}")
logger.info(f"  Columns: {list(events_df.columns)}")

# Event statistics
logger.info("\nEvent statistics by phase:")
for phase in sorted(events_df['phase'].unique()):
    count = len(events_df[events_df['phase'] == phase])
    logger.info(f"  {phase}: {count} events")

logger.info("\nEvent statistics by GWI box:")
for box in sorted(events_df['gwi_box'].unique()):
    count = len(events_df[events_df['gwi_box'] == box])
    label = GWI_BOX_LABELS[box-1]
    logger.info(f"  Box {box} ({label}): {count} events")

# 2. Save regional timeseries as NetCDF (raw_sst, detrended ε, anomaly ε*, standardized)
logger.info("\nSaving regional timeseries...")
regional_ds_list = []
for result in results:
    member_id = result['member_id']
    indices = result['regional_indices']
    
    # Create dataset for this member with all components
    # For each region: raw_sst, detrended (ε), anom (ε*), std
    ds_dict = {}
    for region in NINO_REGIONS.keys():
        ds_dict[f'{region}_raw_sst'] = indices[region]['raw_sst']       # Original SST
        ds_dict[f'{region}_detrended'] = indices[region]['detrended']   # ε: After GWI detrending
        ds_dict[f'{region}_anom'] = indices[region]['anomaly']          # ε*: After seasonal cycle removal
        ds_dict[f'{region}_std'] = indices[region]['standardized']      # Standardized ε*
    
    ds = xr.Dataset(ds_dict)
    ds = ds.expand_dims(member=[member_id])
    regional_ds_list.append(ds)

regional_ds = xr.concat(regional_ds_list, dim='member')
regional_nc = output_dir / 'regional_timeseries_all_members.nc'
regional_ds.to_netcdf(regional_nc)
logger.info(f"✓ Saved regional timeseries: {regional_nc}")
logger.info(f"  Total variables: {len(regional_ds.data_vars)} (3 regions × 4 components)")
logger.info(f"  Variables: {list(regional_ds.data_vars)}")
logger.info(f"  Components: raw_sst, detrended (ε), anom (ε*), std")
logger.info(f"  Shape: {regional_ds.dims}")

# 3. Save precipitation timeseries (ANOMALY ONLY - standardized not used)
logger.info("\nSaving precipitation timeseries...")
precip_ds_list = []
for result in results:
    member_id = result['member_id']
    pr = result['precipitation']
    
    # Only save anomaly (used in detect_enso_events)
    ds = xr.Dataset({
        'nino3_pr_anom': pr['anomaly']
    })
    ds = ds.expand_dims(member=[member_id])
    precip_ds_list.append(ds)

precip_ds = xr.concat(precip_ds_list, dim='member')
precip_nc = output_dir / 'precipitation_timeseries_all_members.nc'
precip_ds.to_netcdf(precip_nc)
logger.info(f"✓ Saved precipitation timeseries: {precip_nc}")
logger.info(f"  Variable: nino3_pr_anom (anomaly only)")
logger.info(f"  Shape: {precip_ds.dims}")

# 4. Save GWI timeseries as NetCDF
logger.info("\nSaving GWI timeseries...")
gwi_ds_list = []
for result in results:
    member_id = result['member_id']
    gwi = result['gwi_relative']
    
    ds = xr.Dataset({'gwi': gwi})
    ds = ds.expand_dims(member=[member_id])
    gwi_ds_list.append(ds)

gwi_ds = xr.concat(gwi_ds_list, dim='member')
gwi_nc = output_dir / 'gwi_timeseries_all_members.nc'
gwi_ds.to_netcdf(gwi_nc)
logger.info(f"✓ Saved GWI timeseries: {gwi_nc}")
logger.info(f"  Shape: {gwi_ds.dims}")

# 5. Save configuration metadata
config = {
    'n_members': len(results),
    'gwi_box_edges': GWI_BOX_EDGES,
    'gwi_box_labels': GWI_BOX_LABELS,
    'nino_regions': {k: str(v) for k, v in NINO_REGIONS.items()},
    'scenarios': SCENARIOS,
    'time_range': f"{START_YEARS[0]}-{END_YEARS[-1]}",
    'processing_date': datetime.datetime.now().isoformat(),
    'harmonized_with': 'functions.py v5 (7-phase classification with seasonal phase-locking)'
}

config_json = output_dir / 'preprocessing_config.json'
with open(config_json, 'w') as f:
    json.dump(config, f, indent=2)
logger.info(f"✓ Saved configuration: {config_json}")

# ============================================================================
# SUMMARY
# ============================================================================

logger.info("\n" + "="*70)
logger.info("PREPROCESSING COMPLETE")
logger.info("="*70)
logger.info(f"Members processed: {len(results)}")
logger.info(f"Total events: {len(events_df)}")
logger.info(f"Time range: {START_YEARS[0]}-{END_YEARS[-1]}")
logger.info(f"GWI bins: {len(GWI_BOX_LABELS)}")
logger.info(f"\nOutputs:")
logger.info(f"  1. {events_csv.name}")
logger.info(f"  2. {regional_nc.name} (3 regions × 4 components: raw_sst, detrended, anom, std)")
logger.info(f"  3. {precip_nc.name} (nino3_pr_anom only)")
logger.info(f"  4. {gwi_nc.name}")
logger.info(f"  5. {config_json.name}")
logger.info(f"\nNext steps:")
logger.info(f"  - Create visualization script for detrending components (raw_sst vs detrended vs trend)")
logger.info(f"  - Run figure generation scripts (2_generate_figure*.py)")
logger.info("="*70)
