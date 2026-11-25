"""Which variable exists for which ensemble, for how many of the paper's members, over which years, and where.

For every (ensemble, scenario) of the roster and every variable the paper or the planned feedback analysis needs,
count the roster members with at least ten years of data in the reduced lorenz products (data/lorenz/<scenario>/),
and record the first and last year with data. The heat fluxes are not in MMLEAv2 at all; their sources are listed
in docs/13_variable_inventory.md.

Output: results/tables/variable_inventory.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import DATA, PRETTY, ROOT  # noqa: E402

SOURCES = {
    "sst (Nino boxes)": ("box_anoms", "nino34_anom"),
    "pr (Nino-3)": ("extra_vars", "nino3_pr_anom"),
    "tauu (Nino-4)": ("extra_vars", "nino4_tauu_anom"),
    "tauu (Nino-3)": ("extra_vars", "nino3_tauu_anom"),
    "gwi (tas)": ("metrics", "gwi"),
    "zos gradient": ("metrics", "meanstate_monthly_h_eq_sw_zos"),
    "z20 tilt": ("metrics", "meanstate_monthly_h_eq_sw_z20"),
}
roster = pd.read_csv(ROOT / "results/tables/members_delta.csv", usecols=["model", "rid", "scenario"])
rows = []
for (model, scen), g in roster.groupby(["model", "scenario"]):
    handles = {}
    for var, (kind, name) in SOURCES.items():
        f = DATA / scen / f"{model}_{kind}.nc"
        if kind not in handles:
            handles[kind] = xr.open_dataset(f) if f.exists() else None
        d = handles[kind]
        have, firsts, lasts = 0, [], []
        if d is not None and name in d:
            ens = [str(e) for e in d.ens.values]
            t = pd.DatetimeIndex(d.time.values)
            arr = d[name].values
            for rid in g.rid:
                if rid not in ens:
                    continue
                x = arr[ens.index(rid)]
                ok = np.isfinite(x) & (x != 0)
                if ok.sum() >= 120:
                    have += 1
                    firsts.append(t[ok][0].year)
                    lasts.append(t[ok][-1].year)
        rows.append(
            dict(
                ensemble=PRETTY.get(model, model),
                model=model,
                scenario=scen,
                variable=var,
                roster_members=len(g),
                members_with_data=have,
                first_year=min(firsts) if firsts else None,
                last_year=max(lasts) if lasts else None,
                last_year_min=min(lasts) if lasts else None,
            )
        )
    for d in handles.values():
        if d is not None:
            d.close()
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/tables/variable_inventory.csv", index=False)
gap = out[(out.members_with_data < out.roster_members) | (out.last_year_min < 2095)]
pd.set_option("display.width", 200)
print(gap.to_string(index=False))
