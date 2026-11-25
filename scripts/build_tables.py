"""Build every table the figures need from the lorenz metrics files.

Outputs (results/tables/):
  members_delta.csv      per-member ref/warm statistics and their change
  models_delta.csv       per-model medians, standard errors, member counts
  gwl_trajectory.csv     per-member statistics in 1 K wide GWI bins every 0.25 K
  obs_summary.csv        observed values computed with the same code
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import MODELS, PRETTY, REF, WARM, ROOT, MIN_MEMBERS, load_members  # noqa: E402
from ensoasym.indices import model_delta_table, gwl_trajectory, summarise, contributions  # noqa: E402

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_COLS = [
    "A_ref",
    "A_warm",
    "d_A",
    "A_strong_ref",
    "d_A_strong",
    "skew_n3_ref",
    "skew_n34_ref",
    "d_skew_n3",
    "rate_ext_EN_ref",
    "rate_ext_LN_ref",
    "rate_str_EN_ref",
    "rate_str_LN_ref",
    "d_rate_ext_EN",
    "d_rate_ext_LN",
    "z20_tilt_ref",
    "d_z20_tilt",
    "C_EN_ref",
    "C_LN_ref",
    "d_C_EN",
    "d_C_LN",
    "d_C_EN_strong",
    "d_C_LN_strong",
    "f_EN_ref",
    "f_LN_ref",
    "d_f_EN",
    "d_f_LN",
    "x2_EN_ref",
    "x2_LN_ref",
    "d_x2_EN",
    "d_x2_LN",
    "var_ref",
    "d_var",
    "sst_gradient_ref",
    "d_sst_gradient",
    "zos_gradient_ref",
    "d_zos_gradient",
    "wp_minus_ct_ref",
    "d_wp_minus_ct",
    "cold_tongue_sst_ref",
    "d_cold_tongue_sst",
    "cold_tongue_sst_warm",
    "pr_nino3_ref",
    "pr_nino3_warm",
    "d_pr_nino3",
    "ctex_ref",
    "warm_pool_sst_ref",
    "d_warm_pool_sst",
    "ctex_ref",
    "d_ctex",
    "dgwi",
]


def main():
    members, trajs, periods = [], [], []
    centres = np.arange(0.5, 4.76, 0.25)
    for model in MODELS:
        mem = load_members(model)
        df = model_delta_table(mem, REF, WARM)
        print(f"{model:18s} members {len(mem):3d} usable {len(df):3d}")
        if len(df) < MIN_MEMBERS:
            print(f"   -> dropped (fewer than {MIN_MEMBERS} usable members)")
            continue
        members.append(df)
        trajs.append(gwl_trajectory(mem, centres, width=1.0, min_months=120))
        periods.append(model_delta_table(mem, (1985, 2014), (2070, 2099), periods=True))
    mem_df = pd.concat(members, ignore_index=True)
    mem_df.to_csv(OUT / "members_delta.csv", index=False, float_format="%.5f")
    per_df = pd.concat(periods, ignore_index=True)
    per_df.to_csv(OUT / "members_delta_periods.csv", index=False, float_format="%.5f")
    cols = [c for c in SUMMARY_COLS if c in mem_df.columns]
    mod = summarise(mem_df, cols)
    mod["pretty"] = [PRETTY[m] for m in mod.index]
    # normalised warming-pattern metric: gradient change per degree of GWI change
    mod["grad_per_K"] = mod["d_sst_gradient"] / mod["dgwi"]
    mod["wpct_per_K"] = mod["d_wp_minus_ct"] / mod["dgwi"]
    mod.to_csv(OUT / "models_delta.csv", float_format="%.5f")
    tr = pd.concat(trajs, ignore_index=True)
    tr.to_csv(OUT / "gwl_trajectory.csv", index=False, float_format="%.5f")

    # cross-model relations with two-level bootstrap and permutation tests
    from ensoasym.bootstrap import two_level_bootstrap, model_bootstrap_median

    mem_df["grad_per_K"] = mem_df["d_sst_gradient"] / mem_df["dgwi"]
    mem_df["wpct_per_K"] = mem_df["d_wp_minus_ct"] / mem_df["dgwi"]
    per_df["grad_per_K"] = per_df["d_sst_gradient"] / per_df["dgwi"]
    boot = []
    pairs = [
        ("grad_per_K", "d_A"),
        ("wpct_per_K", "d_A"),
        ("d_sst_gradient", "d_A"),
        ("grad_per_K", "d_A_strong"),
        ("d_zos_gradient", "d_A_strong"),
        ("d_z20_tilt", "d_A_strong"),
        ("d_zos_gradient", "d_A"),
        ("d_z20_tilt", "d_A"),
        ("d_ctex", "d_A"),
        ("grad_per_K", "d_C_EN"),
        ("grad_per_K", "d_C_LN"),
        ("d_skew_n3", "d_A"),
        ("grad_per_K", "d_skew_n3"),
        ("pr_nino3_ref", "d_A"),
        ("pr_nino3_warm", "d_A"),
        ("cold_tongue_sst_ref", "d_A"),
        ("d_pr_nino3", "d_A"),
        ("pr_nino3_ref", "d_A_strong"),
        ("cold_tongue_sst_ref", "d_A_strong"),
        ("pr_nino3_ref", "grad_per_K"),
        ("cold_tongue_sst_ref", "grad_per_K"),
        ("pr_nino3_ref", "d_C_EN"),
        ("pr_nino3_ref", "d_C_LN"),
    ]
    print("\nCross-model relations (per-model medians, two-level bootstrap 95%% CI, permutation p)")
    for x, y in pairs:
        b = two_level_bootstrap(mem_df, x, y)
        b["design"] = "gwi_bins"
        boot.append(b)
        print(
            f"  {y:11s} vs {x:15s} n={b['n']:2d} r={b['r']:+.2f} [{b['r_lo']:+.2f},{b['r_hi']:+.2f}] "
            f"rho={b['rho']:+.2f} [{b['rho_lo']:+.2f},{b['rho_hi']:+.2f}] p_perm={b['p_perm']:.3f}"
        )
    for x, y in [("grad_per_K", "d_A"), ("d_sst_gradient", "d_A")]:
        b = two_level_bootstrap(per_df, x, y)
        b["design"] = "periods_2070-2099_vs_1985-2014"
        boot.append(b)
        print(
            f"  [fixed periods] {y} vs {x} n={b['n']} r={b['r']:+.2f} [{b['r_lo']:+.2f},{b['r_hi']:+.2f}] p_perm={b['p_perm']:.3f}"
        )
    pd.DataFrame(boot).to_csv(OUT / "bootstrap.csv", index=False, float_format="%.4f")
    # multi-model medians of the per-model percent changes with model-bootstrap intervals
    mm = []
    for q in ("d_C_EN", "d_C_LN", "d_A", "d_var"):
        base = q.replace("d_", "") + "_ref"
        if q != "d_A":
            pct = 100 * mod[q] / mod[base]
            med, lo, hi = model_bootstrap_median(pct.values)
            mm.append(dict(quantity=q + "_pct", median=med, lo=lo, hi=hi))
        med, lo, hi = model_bootstrap_median(mod[q].values)
        mm.append(dict(quantity=q, median=med, lo=lo, hi=hi))
    mm = pd.DataFrame(mm)
    mm.to_csv(OUT / "multimodel_medians.csv", index=False, float_format="%.4f")
    print(mm.to_string(index=False))

    # observations with the same code
    obs = xr.open_dataset(ROOT / "data/obs/obs_enso_metrics_1979_2025.nc")
    x = obs["nino34_std"].values.astype(float)
    ph = obs["phase_indicator"].values.astype(int)
    ok = np.isfinite(x)
    c = contributions(x[ok], ph[ok])
    om = xr.open_dataset(ROOT / "data/obs/obs_metrics_1979_2025_g025.nc")
    grad = om["meanstate_sst_gradient"].to_series()
    grad.index = om["year"].values.astype(int) if "year" in om.coords else grad.index
    g = obs["gwi"].to_series()
    g_year = g.groupby(pd.DatetimeIndex(g.index).year).mean()
    common = grad.index.intersection(g_year.index)
    slope_grad_gwi = stats.linregress(g_year.loc[common].values, grad.loc[common].values)
    slope_grad_t = stats.linregress(np.array(common, float), grad.loc[common].values)
    rows = {
        "A_obs": c["A"],
        "A_strong_obs": c["A_strong"],
        "C_EN_obs": c["C_EN"],
        "C_LN_obs": c["C_LN"],
        "var_obs": c["var"],
        "n_months": int(ok.sum()),
        "grad_mean": float(grad.loc[common].mean()),
        "grad_trend_per_decade": slope_grad_t.slope * 10,
        "grad_trend_se_per_decade": slope_grad_t.stderr * 10,
        "grad_per_K_gwi": slope_grad_gwi.slope,
        "grad_per_K_gwi_se": slope_grad_gwi.stderr,
        "gwi_span": float(g_year.loc[common].max() - g_year.loc[common].min()),
        "years": f"{common.min()}-{common.max()}",
    }
    pd.Series(rows).to_csv(OUT / "obs_summary.csv", header=False)
    print("\nObs:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
