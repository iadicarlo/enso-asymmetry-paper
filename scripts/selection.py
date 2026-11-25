"""ENSO-fidelity subsets of the large ensembles and the cross-model relation within each.

Criteria (reference bin, GWI < 1 K, per-model median across members; observed
values are the ERSSTv6 + GPCP estimates over 1980-2025):
  ext_events   extreme El Niño and extreme La Niña both occur at least half as often as observed
  en_dominated present-day asymmetry index A > 0, as observed
  skew         Niño-3 skewness at least half the observed value
  cold_tongue  cold-tongue SST bias within 1 K of observed (1980-2014 mean)
  all_three    ext_events and skew and cold_tongue
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import ROOT, PRETTY  # noqa: E402
from ensoasym.bootstrap import two_level_bootstrap  # noqa: E402

T = ROOT / "results" / "tables"
mod = pd.read_csv(T / "models_delta.csv", index_col=0)
mem = pd.read_csv(T / "members_delta.csv")
mem["grad_per_K"] = mem["d_sst_gradient"] / mem["dgwi"]
mem["wpct_per_K"] = mem["d_wp_minus_ct"] / mem["dgwi"]
obs = pd.read_csv(T / "obs_multi.csv")
o = obs[(obs.sst == "ERSSTv6") & (obs.pr == "GPCP")].iloc[0]

crit = pd.DataFrame(index=mod.index)
crit["rate_ext_EN"] = mod["rate_ext_EN_ref"]
crit["rate_ext_LN"] = mod["rate_ext_LN_ref"]
crit["skew_n3"] = mod["skew_n3_ref"]
crit["A_ref"] = mod["A_ref"]
crit["ct_bias"] = mod["cold_tongue_sst_ref"] - o["cold_tongue_ref"]
crit["ext_events"] = (crit.rate_ext_EN >= 0.5 * o.rate_ext_EN) & (crit.rate_ext_LN >= 0.5 * o.rate_ext_LN)
crit["en_dominated"] = crit.A_ref > 0
crit["skew"] = crit.skew_n3 >= 0.5 * o.skew_n3
crit["cold_tongue"] = crit.ct_bias.abs() < 1.0
crit["all_three"] = crit["ext_events"] & crit["skew"] & crit["cold_tongue"]
crit["pretty"] = [PRETTY[m] for m in crit.index]
crit.to_csv(T / "model_fidelity.csv", float_format="%.4f")
print(
    "obs (ERSSTv6+GPCP): ext EN %.3f/yr, ext LN %.3f/yr, skew3 %.2f, CT %.2f C"
    % (o.rate_ext_EN, o.rate_ext_LN, o.skew_n3, o.cold_tongue_ref)
)
print(
    crit[
        [
            "pretty",
            "rate_ext_EN",
            "rate_ext_LN",
            "skew_n3",
            "A_ref",
            "ct_bias",
            "ext_events",
            "en_dominated",
            "skew",
            "cold_tongue",
            "all_three",
        ]
    ]
    .round(3)
    .to_string()
)

rows = []
for name in ["all", "ext_events", "en_dominated", "skew", "cold_tongue", "all_three"]:
    keep = list(crit.index) if name == "all" else list(crit.index[crit[name]])
    sub = mem[mem.model.isin(keep)]
    for x, y in [
        ("pr_nino3_ref", "d_A"),
        ("cold_tongue_sst_ref", "d_A"),
        ("grad_per_K", "d_A"),
        ("wpct_per_K", "d_A"),
        ("grad_per_K", "d_A_strong"),
    ]:
        if len(keep) < 5:
            continue
        b = two_level_bootstrap(sub, x, y, n_boot=1000)
        b["subset"] = name
        b["models"] = ",".join(keep)
        rows.append(b)
        print(
            f"{name:13s} n={b['n']:2d} {y:10s} vs {x:11s} r={b['r']:+.2f} [{b['r_lo']:+.2f},{b['r_hi']:+.2f}] rho={b['rho']:+.2f} p_perm={b['p_perm']:.3f} slope={b['slope']:+.3f}"
        )
pd.DataFrame(rows).to_csv(T / "selection_bootstrap.csv", index=False, float_format="%.4f")
