# Present-day eastern Pacific mean state is linked to projected ENSO asymmetry changes

Code and reduced data for the paper "Present-day eastern Pacific mean state is linked to projected ENSO asymmetry
changes" by I. Abdelkader Di Carlo, Y. Y. Planton, N. Maher, P. Trascasa-Castro, M. Lengaigne and J. Vialard
(submitted to Geophysical Research Letters).

The paper uses 18 single-model initial-condition large ensembles (623 members) of the Multi-Model Large Ensemble
Archive version 2 (MMLEAv2). It measures how the balance between the El Niño and La Niña contributions to Niño-3.4
variance changes above 3 K of global warming, and relates that change to the present-day eastern Pacific state of
each model.

## What is here

| Folder | Content |
| --- | --- |
| `src/ensoasym/` | Library: reading the reduced files, ENSO classification, indices, bootstrap, plotting helpers |
| `scripts/` | One script per table set and per figure, and `rebuild_all.sh`, which runs them in order |
| `results/tables/` | The tables behind every number of the paper (CSV) |
| `figures/` | The figures of the paper and of the supporting information |
| `preprocessing/` | The scripts that reduced the MMLEAv2 fields to the per-member files, as run on the cluster |
| `data/` | The reduced data, not in the GitHub repository (see below) |

## Data

The reduced data (about 550 MB) is archived on Zenodo: DATA_DOI. Download it and unpack it at the root of this
repository, so that `data/lorenz/` and `data/obs/` sit next to `src/` and `scripts/`.

- `data/lorenz/<scenario>/<model>_metrics.nc`, `_box_anoms.nc`, `_extra_vars.nc`, `_nonlinear_indices.nc`: monthly
  indices and box averages of each member (Niño-3.4 and Niño-3 SST, rainfall, the zonal SST gradient, the global
  warming index), one file per ensemble and scenario (RCP8.5, SSP3-7.0 or SSP5-8.5).
- `data/lorenz/trop_mean_global/`: tropical-mean SST of each member, used for the relative Niño-3.4 index.
- `data/lorenz/fidelity/`: present-day climatologies of each ensemble and the member maps used in Figure 4.
- `data/obs/`: the same indices from ERSSTv5, ERSSTv6, COBE-SST2, HadISST, GPCP and CMAP, and the observed global
  warming index from Berkeley Earth.
- `data/reference/`: the threshold vector of the ENSO classification.

The model fields themselves are available from the NCAR Research Data Archive (MMLEAv2, dataset d651039). The
observational products are available from the NOAA Physical Sciences Laboratory, the Met Office Hadley Centre and
Berkeley Earth.

## Reproducing the tables and figures

```bash
git clone https://github.com/iadicarlo/enso-asymmetry-paper.git
cd enso-asymmetry-paper
# unpack the Zenodo data archive here
uv sync
bash scripts/rebuild_all.sh
```

Python 3.11. Without `uv`: `python -m venv .venv`, then `.venv/bin/pip install -e .`. The full rebuild takes about
five minutes on a laptop; the first run stores the ENSO classification of each member in `data/lorenz/noaa_cache/`. Figure 4 needs the Natural Earth land outlines, which cartopy downloads on first use.

| Figure | Script | Content |
| --- | --- | --- |
| Fig. 1 | `scripts/fig1_large_ensembles.py` | Asymmetry change of every member and ensemble, and detection against ensemble size |
| Fig. 2 | `scripts/fig2_phases.py` | El Niño and La Niña variance contributions against global warming |
| Fig. 3 | `scripts/fig3.py` | Asymmetry change against present-day Niño-3 rainfall and against the forced zonal SST-gradient change |
| Fig. 4 | `scripts/fig4_summary.py` | Driest and wettest ensembles, the rainfall axis, and the two predictors |
| Fig. S1 | `scripts/figS1_evaluation.py` | Each ensemble compared with observations |
| Fig. S2 | `scripts/figS2_relation_tests.py` | The relation under alternative choices |
| Fig. S3 | `scripts/figS3_rainfall_response.py` | Rainfall response per K of SST anomaly |

The table scripts run before the figures: `build_obs.py` (observed indices), `build_tables.py` (per-member and
per-ensemble changes, bootstrap correlations), `selection.py` and `screened.py` (subsets of ensembles),
`rain_response.py`, `constrain.py` (partial correlations and the estimate at the observed state), `smile_use.py`
(variance partition between internal variability and forced response), `extra_stats.py`, `member_trends.py`,
`composite_stats.py` and `variable_inventory.py`.

## Preprocessing

`preprocessing/` holds the scripts that produced the reduced files from the MMLEAv2 monthly fields on a computing
cluster: the regional extraction (`mmleav2_tropics_pipeline.py`), the global warming index
(`preprocess_gwi_ensemble.py`, `preprocess_gwi_scenario.py`), the member indices (`enso_metrics_ensemble.py`,
`enso_metrics_extra_vars.py`, `enso_meanstate_nonlinear_indices.py`), the tropical-mean SST
(`preprocess_trop_mean_tos.py`), the climatologies and maps (`enso_fidelity_maps.py`, `enso_member_maps.py`) and the
observed indices (`build_obs_enso_metrics.py`, `compute_obs_metrics.py`). They are given for reference. They expect
the archive on local disk and paths set in `config.py`, and they are not needed to reproduce the paper from the
reduced data. The script that wrote the `_box_anoms.nc` files is not included; these files hold monthly SST anomalies averaged
over the Niño-3, Niño-3.4 and Niño-4 boxes and the western and eastern equatorial Pacific.

## License

Code: MIT. Data and tables: CC BY 4.0.

## Citation

If you use this code or data, please cite the paper and the Zenodo records.
