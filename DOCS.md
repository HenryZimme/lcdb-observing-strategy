# Technical Documentation

`lcdb-observing-strategy` — last updated 2026-03

---

## Module reference

### `src/data_loader.py`

Loads and cleans the LCDB `lc_summary.csv`.

**Key behaviors:**
- Auto-detects the header row by scanning the first 60 lines for `Number`, `Name`, `Period` co-occurrence (LCDB files have a ~15-line preamble of metadata)
- Drops rows where `Number` is non-numeric (these are continuation metadata rows)
- Replaces LCDB null sentinel (`-9.99`) with `np.nan`
- Coerces `Period`, `Diam`, `Albedo` to float

**Functions:**

| Function | Returns | Notes |
|---|---|---|
| `mount_drive()` | None | No-op outside Colab |
| `find_lcdb_file(default_path, search_depth)` | str path | Walks MyDrive up to `search_depth` levels |
| `detect_header_row(file_path, max_scan=60)` | int | 0-based row index |
| `load_lcdb(file_path, local_path)` | pd.DataFrame | Main entry point |

---

### `src/filters.py`

Defines the three filter levels. All return copies (no mutation of input).

**Filter hierarchy:**

```
full_df    — entire LCDB (n ≈ 33,000)
  └── diam_df  — 35 km ≤ Diam ≤ 100 km
        └── ap_df  — diam_df + Albedo ≤ 0.075  (Cindygraber-like)
```

**Why albedo 0.075?** Albedo constrains asteroid taxonomy. Cindygraber has albedo = 0.039,
consistent with C/D-type dark asteroids. The 0.075 cutoff encompasses the C, D, and B complexes
while excluding the S-type silicate asteroids (albedo ≈ 0.20). See Mahlke et al. (2022).

**Constants:**

| Constant | Value | Source |
|---|---|---|
| `CINDYGRABER_DIAM_KM` | 38.46 km | JPL SBDB |
| `ALBEDO_MAX` | 0.075 | taxonomic cutoff |
| `SPIN_BARRIER` | 2.2 h | tensile-strength limit (Scheeres & Sanchez 2018) |

---

### `src/plotting.py`

All plot functions:
- Accept an optional `ax` argument (create their own figure if `None`)
- Return the axes for further annotation
- Accept an optional `save_path` (saves at 300 dpi if provided)
- Call `set_pub_style()` internally

**`set_pub_style()`** applies Minor Planet Bulletin-compatible rcParams: sans-serif font, inward ticks, 300 dpi save.

**`add_period_stats(data, ax, show_spin_barrier)`** overlays:
- Median (solid purple)
- Q1, Q3 (dotted green)
- Spin barrier at 2.2h (dashed red, optional)

---

### `src/observing_strategy.py`

Core campaign planning module.

#### Phase-coverage model

`calculate_nights_to_solve(period, ...)` tracks 96 phase bins (3.75° each). A night "solves" when:
- ≥ 85% of bins are marked `True` **AND**
- total observed time ≥ 2 × period (two full cycles)

Weather is simulated as a Bernoulli draw per night (`clear_fraction=0.20` by default). The RNG seed is set per trial so Monte Carlo runs are reproducible.

**Why 96 bins?** 360° / 96 = 3.75° resolution. This is finer than typical photometric phase resolution for ~15 mag asteroids and ensures the coverage criterion is meaningful.

**Why 85%?** Standard in the literature for "sufficient" phase coverage to distinguish the dominant period from aliases (Harris 1989; Warner et al. 2009).

**Why 2 full cycles?** A single cycle can be ambiguous; two cycles provide independent verification of the period and allow alias rejection.

#### Monte Carlo weather simulation

`run_mc_weather_simulation(periods_array, n_trials=1000)` runs `k=1000` independent trials per asteroid, each with a different RNG seed, and returns `mean_nights` and `std_nights` across trials. The ±1σ band in the CDF plot comes from these.

#### CDF and ROI

`compute_cdf_roi(nights_series)` returns a DataFrame with:

| Column | Definition |
|---|---|
| `cumulative_probability` | fraction of population solved by night N |
| `efficiency` | `cumulative_probability / N` — marginal value per night invested |
| `marginal_gain` | `Δ cumulative_probability` — incremental gain from adding one more night |

The **peak efficiency night** is where `efficiency` is maximized — the optimal single-night stopping point if you could only observe one more night.

The **80% night** is the first night where `cumulative_probability ≥ 0.80`.

---

### `src/lightcurve_sim.py`

Synthetic lightcurve and period-recovery diagnostics.

#### Lightcurve model

```
mag(t) = -0.15 × cos(2π·t/P) - 0.10 × cos(4π·t/P)
```

This is a 2nd-order Fourier approximation of a double-peaked lightcurve (two reflectivity maxima per rotation). The 0.15 and 0.10 amplitudes produce a realistic ~0.3 mag amplitude variation.

#### Period recovery methods

**Lomb-Scargle (LS):** Standard unevenly-sampled power spectrum. Best period = max power frequency. Well-understood behavior; the `astropy.timeseries.LombScargle` implementation is used.

**Fourier RMS scan:** For each trial period P on a grid, fits a 2nd-order Fourier model via least squares and computes the residual RMS. Best period = minimum RMS. More robust to aliases than LS for short observation baselines but slower.

#### Diagnostic grid

`plot_period_evolution()` produces an `(n_nights × 4)` grid:
- Column 0: lightcurve phased at LS best period
- Column 1: lightcurve phased at RMS best period
- Column 2: LS periodogram
- Column 3: RMS scan

This shows visually how both methods converge as nights accumulate.

---

### `src/convergence.py`

Tracks the Lomb-Scargle estimate night-by-night and derives a stopping rule.

**`run_convergence_simulation(ap_df, ...)`** records for each night:
- `estimated_period`: LS best-fit period
- `abs_error`: |estimated - true|
- `delta_p`: |estimated_period(night N) - estimated_period(night N-1)|

**`find_stability_threshold(conv_df, target_accuracy=0.80)`** scans delta-P thresholds from large to small. The "optimal threshold" is the largest value at which ≥80% of converged estimates have `abs_error < 0.05h`. This gives the operational rule:

> *Stop extending the campaign when the night-to-night change in best-fit period drops below X hours.*
>
> **Note:** I used an 80% threshold, because ~ 20% of the sample had significant aliasing. As seen earlier, aliasing can be mitigated by running a RMS fit on asteroids with multiple high power LS peaks. 

In practice this should be combined with a visual inspection of the phased lightcurve.

---

### `src/bootstrap_analysis.py`

Bootstraps the `nights_required` distribution to estimate CDF uncertainty from finite sample size.

**Why bootstrap and not just MC weather?** The MC weather simulation captures uncertainty in *which nights are clear*. The bootstrap additionally captures uncertainty in *which asteroid the target resembles* — since the ap_df has only 698 entries, random sampling noise in the population CDF is non-negligible.

`run_bootstrap_cdf(nights_data, n_boot=1000)` resamples with replacement and returns 2.5/97.5 percentile bands across epochs. The 95% CI represents the range of CDFs consistent with the observed population sample.

---

## Simulation parameters

| Parameter | Value | Justification |
|---|---|---|
| Observing hours/night | 7.0 h | Typical usable dark time at PAO in winter |
| Clear fraction | 0.20 | ~20% clear nights assumed (pessimistic for Massachusetts winter) |
| Phase bins | 96 | 3.75° resolution |
| Min phase coverage | 85% | Standard in MPB literature |
| Min periods covered | 2 | Alias rejection |
| Max nights (cap) | 100 | Simulation upper bound |
| MC trials | 1000 | Stable mean/std |
| Bootstrap epochs | 1000 | Stable 95% CI |
| Noise scale | 0.01–0.05 mag | Realistic for 16 mag asteroid, 60s exposures |
| Obs/night | 30–50 | ~5 min cadence over 7h |

---

## Known limitations

1. **Lightcurve model is simplified.** The double-peaked Fourier model does not capture tumbling, binary companions, or surface albedo variation — all of which Cindygraber may exhibit.

2. **Phase coverage ≠ period recovery.** The observing_strategy model tells you when you've sampled enough phase; it does not directly tell you whether the LS/RMS method will return the correct period. The convergence module addresses this separately.

3. **Weather model is memoryless.** Real weather has serial correlation (cloudy nights cluster). The Bernoulli model is a simplification.

4. **Lomb-Scargle grid is finite.** The frequency grid resolution limits period precision. In real analysis, MPO Canopus performs a more exhaustive search.

5. **ap_df sample size.** Only 698 asteroids match the Cindygraber-like filter. The bootstrap CI reflects this uncertainty.

---

## Output files

All outputs are written to `outputs/`:

| File | Description |
|---|---|
| `fig_period_dist_full.png` | LCDB full period distribution |
| `fig_period_dist_filtered.png` | Filtered (ap_df) period distribution |
| `fig_scatter_period_diam.png` | Period vs diameter scatter |
| `fig_cdf_roi_mc.png` | CDF/ROI with MC weather bands |
| `fig_cdf_roi_deterministic.png` | CDF/ROI without weather |
| `fig_period_evolution_grid.png` | 4×4 LS/RMS diagnostic grid |
| `fig_convergence_spaghetti.png` | Convergence spaghetti plot |
| `fig_bootstrap_cdf.png` | Final bootstrapped CDF |
| `bootstrap_roi_metrics.csv` | Bootstrapped ROI data table |
