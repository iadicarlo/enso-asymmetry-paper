#!/bin/bash
# Rebuild every table and figure of the paper from the lorenz products under the current classification
# (ENSO_CLASSIFICATION, default noaa). Run from the project root with the uv environment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# observed maps on the 2.5 degree archive grid (reads COBE-SST2 over OPeNDAP, run once): scripts/build_obs_maps_g025.py
echo "classification: ${ENSO_CLASSIFICATION:-noaa}"
# figS3_rainfall_response also writes obs_rain_response.csv, which constrain reads
for s in build_obs build_tables selection rain_response figS3_rainfall_response constrain screened smile_use extra_stats member_trends composite_stats variable_inventory family_test; do
    echo "=== $s"; $PY scripts/$s.py 2>&1 | grep -viE "warning|rolling window|data range|^\s*$" | tail -4
done
for f in fig1_large_ensembles fig2_phases fig3 fig4_summary figS1_evaluation figS2_relation_tests figS3_rainfall_response; do
    printf "=== %-16s" "$f"; $PY scripts/$f.py 2>&1 | grep -E "^saved|Error|Traceback" | tr '\n' ' ' | cut -c1-90; echo
done
# side analyses for the co-authors (not in the paper): scripts/side_wind_feedback.py
echo "done"
