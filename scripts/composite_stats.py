"""Numbers quoted in the text for the co-author review of 2026-09-21 that no other table holds.

- ensembles whose median asymmetry change differs from zero by more than two standard errors (1.253 s / sqrt(n))
- the driest-six and wettest-six composites of Figure 4d: median over the six ensembles of the per-model median
  change of the El Nino and La Nina contributions (percent) and of A, with a two-level bootstrap (ensembles, then
  members; 4000 resamples, fixed seed) giving 90 and 95 percent intervals

Output: results/tables/composite_stats.csv (statistic, value, lo90, hi90, lo95, hi95, n)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.plotting import TABLES  # noqa: E402

DRY = ["mpi_lens", "mpi_lens_cmip6", "e3sm_v1_lens", "csiro_mk36_lens", "ipsl_cm6a_lens", "gfdl_spear_lens"]
WET = ["cesm2_lens", "access_lens", "miroc6_lens", "canesm2_lens", "giss-e2", "miroc_esm2l_lens"]

mem = pd.read_csv(TABLES / "members_delta.csv")
mem = mem.assign(pEN=100 * mem.d_C_EN / mem.C_EN_ref, pLN=100 * mem.d_C_LN / mem.C_LN_ref)
obs = pd.read_csv(TABLES / "obs_multi.csv")
o6 = obs[(obs.sst == "ERSSTv6") & (obs.pr == "GPCP")].iloc[0]
rows = []

z = mem.groupby("model").d_A.agg(lambda v: np.median(v) / (1.253 * np.std(v, ddof=1) / np.sqrt(len(v))))
rows.append(dict(statistic="ensembles with d_A below -2 SE", value=int((z < -2).sum()), n=len(z)))
rows.append(dict(statistic="ensembles with d_A above +2 SE", value=int((z > 2).sum()), n=len(z)))

rng = np.random.default_rng(0)
for name, models in (("dry six", DRY), ("wet six", WET)):
    for col in ("pEN", "pLN", "d_A"):
        groups = {m: mem.loc[mem.model == m, col].values for m in models}
        point = np.median([np.median(v) for v in groups.values()])
        boot = [
            np.median([np.median(rng.choice(groups[m], len(groups[m]))) for m in rng.choice(models, len(models))])
            for _ in range(4000)
        ]
        lo90, hi90, lo95, hi95 = np.percentile(boot, [5, 95, 2.5, 97.5])
        rows.append(dict(statistic=f"{name} {col}", value=point, lo90=lo90, hi90=hi90, lo95=lo95, hi95=hi95, n=6))

# the comparisons of members with observations over 1980 to 2025 are in obs_vs_members.csv (member_trends.py)
out = pd.DataFrame(rows)
out.to_csv(TABLES / "composite_stats.csv", index=False, float_format="%.4f")
print(out.round(3).to_string(index=False))
