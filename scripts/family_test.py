"""One ensemble per model family: the cross-model relation for every combination that keeps a single ensemble of the
CESM, CanESM, E3SM, GFDL, MIROC and MPI families (11 ensembles, 96 combinations). Writes genealogy_sensitivity.csv."""

import itertools
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.plotting import TABLES  # noqa: E402

FAMILIES = [
    ["cesm1_lens", "cesm2_lens"],
    ["canesm2_lens", "canesm5_lens"],
    ["e3sm_v1_lens", "e3sm_v2_lens"],
    ["gfdl_cm3_lens", "gfdl_esm2m_lens", "gfdl_spear_lens"],
    ["miroc6_lens", "miroc_esm2l_lens"],
    ["mpi_lens", "mpi_lens_cmip6"],
]
mod = pd.read_csv(TABLES / "models_delta.csv", index_col=0)
single = [m for m in mod.index if m not in sum(FAMILIES, [])]
rows = []
for combo in itertools.product(*FAMILIES):
    d = mod.loc[single + list(combo)]
    rr, pr = stats.pearsonr(d.pr_nino3_ref, d.d_A)
    rc, pc = stats.pearsonr(d.cold_tongue_sst_ref, d.d_A)
    rows.append(dict(n=len(d), r_rain=rr, p_rain=pr, r_ct=rc, p_ct=pc, r_grad=stats.pearsonr(d.grad_per_K, d.d_A)[0]))
out = pd.DataFrame(rows)
out.to_csv(TABLES / "genealogy_sensitivity.csv", index=False, float_format="%.4f")
print(f"{len(out)} combinations of {out.n.iloc[0]} ensembles; r rainfall {out.r_rain.max():+.3f} to {out.r_rain.min():+.3f}, "
      f"median {out.r_rain.median():+.3f}, negative in {int((out.r_rain < 0).sum())}, p < 0.05 in {100 * (out.p_rain < 0.05).mean():.1f} percent")
