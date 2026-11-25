"""Shared I/O helpers for SMILEs per-model figures.

The MMLEAv2 metrics files are organised by scenario (ssp585 / ssp370 /
rcp85), but the per-bin GWI-pooled estimators we use for figK / figM
(methods §2.4c) treat scenario as invisible — every "hot bin" sample is
a month with smoothed GWI > 3 °C, regardless of whether it sits in
SSP585 around 2080 or SSP370 around 2095.

So the natural unit of analysis is **one realisation, scenario chosen
to maximise GWI > 3 °C coverage**, not "the SSP585 file as a whole".
This module provides ``load_member_roster`` which builds that
deduplicated roster, and ``iter_member_arrays`` which iterates over
(rid, scenario, x, ph, g) tuples for downstream stats.

See ``project_smiles_member_pooling.md`` for the rule and rationale.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

# Module-level log + accumulator so callers can inspect dropped members
# at the end of a script run (e.g. to flag which model+member triples
# need a metrics rebuild).
_log = logging.getLogger("smiles_io")
DROPPED_ZERO_VARIANCE: list[tuple[str, str, str]] = []   # (model, scenario, rid)


def reset_drop_log() -> None:
    """Clear the in-process record of dropped members (call before main loop)."""
    DROPPED_ZERO_VARIANCE.clear()


def report_dropped() -> None:
    """Print dropped-member summary to stderr (call at end of main loop)."""
    if not DROPPED_ZERO_VARIANCE:
        return
    print(f"\n[smiles_io] {len(DROPPED_ZERO_VARIANCE)} member(s) dropped — "
          f"corrupt nino34_std (identically zero):", flush=True)
    for model, scenario, rid in DROPPED_ZERO_VARIANCE:
        print(f"  - {model} / {scenario} / {rid}  → rebuild recommended")


# Default scenario preference order — used to break ties when multiple
# scenarios have the same hot-bin coverage for a realisation.
SCENARIO_PREF = {"ssp585": 0, "ssp370": 1, "rcp85": 2}

# Default GWI window used for "does this realisation cover a hot bin?"
# (matches methods §2.4c: hot = GWI > 3 °C).
DEFAULT_HOT_LO = 3.0
DEFAULT_REF_LO = 0.0
DEFAULT_REF_HI = 1.0
DEFAULT_MIN_BIN_SAMPLES = 10


def load_member_roster(
    model: str,
    *,
    results_dir: Path,
    ref_lo: float = DEFAULT_REF_LO,
    ref_hi: float = DEFAULT_REF_HI,
    hot_lo: float = DEFAULT_HOT_LO,
    min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES,
) -> list[tuple[str, str, xr.Dataset, int]]:
    """Build a deduplicated per-realisation roster across scenarios.

    For each realisation, picks the scenario that gives the deepest
    hot-bin (>``hot_lo`` °C) coverage. Ties broken by SCENARIO_PREF
    (SSP585 first — canonical high-warming scenario).

    Parameters
    ----------
    model
        MMLEAv2 model name (matches metrics file basename, e.g.
        ``cesm2_lens``).
    results_dir
        Path to ``results/enso_metrics`` (containing the per-scenario
        subdirectories).
    ref_lo, ref_hi
        GWI bounds for the reference bin (default [0, 1] °C).
    hot_lo
        GWI lower bound for the hot bin (default >3 °C, no upper bound).
    min_bin_samples
        Drop a realisation if either bin has fewer than this many finite
        samples. Default 10 (rapidity SE becomes meaningless below).

    Returns
    -------
    list of (realisation_id, scenario, dataset, ens_index) tuples.
    Empty list if no realisations qualify.

    Notes
    -----
    Dataset handles in the returned roster MUST be closed by the caller
    when done. Use ``close_roster()`` to do this safely.
    """
    candidates: dict[str, list] = {}
    for scenario in ("ssp585", "ssp370", "rcp85"):
        path = results_dir / scenario / f"{model}_metrics.nc"
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        ens_ids = (list(ds.coords["ens"].values)
                   if "ens" in ds.coords else ["m0"])
        gwi   = ds["gwi"].values
        nino  = ds["nino34_std"].values
        if gwi.ndim == 1:
            gwi  = gwi[np.newaxis, :]
            nino = nino[np.newaxis, :]
        for k, rid in enumerate(ens_ids):
            g = gwi[k]
            n_ref = int((np.isfinite(g) & (g >= ref_lo) & (g < ref_hi)).sum())
            n_hot = int((np.isfinite(g) & (g >= hot_lo)).sum())
            if n_hot < min_bin_samples or n_ref < min_bin_samples:
                continue
            # Sanity gate: skip members whose nino34_std is identically
            # zero (or has variance < 1e-6) across all finite months.
            # Such members are corrupt artefacts of a previous metrics
            # build (see project_corrupt_metrics_members.md). Log them so
            # the user sees which (model, scenario, rid) triples need
            # a metrics-pipeline rerun.
            x = nino[k]
            x_finite = x[np.isfinite(x)]
            if x_finite.size == 0 or float(np.std(x_finite)) < 1e-6:
                triple = (model, scenario, str(rid))
                if triple not in DROPPED_ZERO_VARIANCE:
                    DROPPED_ZERO_VARIANCE.append(triple)
                _log.warning(
                    "Dropping corrupt member: %s / %s / %s — nino34_std "
                    "has zero variance (artefact of stale metrics build)",
                    model, scenario, rid,
                )
                continue
            candidates.setdefault(str(rid), []).append(
                (n_hot, SCENARIO_PREF[scenario], scenario, ds, k)
            )
        # Note: we deliberately keep `ds` open here because some realisations
        # in this scenario may end up being the ones we keep. We close
        # unused datasets after dedup.

    # Per realisation: pick (max n_hot, then preferred scenario)
    roster = []
    chosen_datasets: set[int] = set()
    all_datasets: set[int] = set()
    for rid, opts in candidates.items():
        opts.sort(key=lambda o: (-o[0], o[1]))
        n_hot, _, scenario, ds, k = opts[0]
        roster.append((rid, scenario, ds, k))
        chosen_datasets.add(id(ds))
        for o in opts:
            all_datasets.add(id(o[3]))
    # Close datasets that contributed no kept realisations
    for rid, opts in candidates.items():
        for _, _, _, ds, _ in opts:
            if id(ds) not in chosen_datasets:
                try:
                    ds.close()
                except Exception:
                    pass
                chosen_datasets.add(id(ds))   # mark as already-closed
    return roster


def close_roster(roster: list[tuple[str, str, xr.Dataset, int]]) -> None:
    """Close all unique dataset handles in the roster (idempotent)."""
    seen: set[int] = set()
    for _, _, ds, _ in roster:
        if id(ds) in seen:
            continue
        seen.add(id(ds))
        try:
            ds.close()
        except Exception:
            pass


def iter_member_arrays(
    roster: list[tuple[str, str, xr.Dataset, int]],
    *,
    var_nino: str = "nino34_std",
    var_phase: str = "phase_indicator",
    var_gwi: str = "gwi",
) -> Iterator[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (realisation_id, scenario, nino_1d, phase_1d, gwi_1d) per member.

    Phase is cast to float so downstream NaN-aware masks work uniformly.
    """
    for rid, scenario, ds, k in roster:
        nino  = np.atleast_2d(ds[var_nino].values)[k]
        phase = np.atleast_2d(ds[var_phase].values)[k].astype(float)
        gwi   = np.atleast_2d(ds[var_gwi].values)[k]
        yield rid, scenario, nino, phase, gwi


def member_count(roster) -> int:
    """Convenience: number of unique realisations in the roster."""
    return len(roster)
