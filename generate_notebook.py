"""
generate_notebook.py
--------------------
builds lcdb_observing_strategy.ipynb programmatically.
each cell corresponds to a logical analysis stage.
"""

import json
import textwrap

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }


cells = [

# --- title ---
cell("""# LCDB Observing Strategy: Cindygraber (MPC 7605)
**Rotational period determination and campaign planning from the Asteroid Lightcurve Database (LCDB)**

Author: Henry Zimmerman · Phillips Academy Observatory · PHY530 (2026)

This notebook walks through the full analysis pipeline:
1. Load & clean the LCDB `lc_summary.csv`
2. Filter for Cindygraber-like asteroids (diameter + albedo)
3. Visualize period distributions
4. Model observing campaign length (phase-coverage simulation)
5. Monte Carlo weather uncertainty
6. CDF / ROI analysis
7. Synthetic lightcurve and period-recovery diagnostics
8. LS convergence simulation and delta-P stopping criterion
9. Bootstrapped final CDF with 95% CI
""", "markdown"),

# --- setup ---
cell("""# install dependencies (colab)
!pip install astropy scikit-learn seaborn --quiet"""),

cell("""import sys, os
sys.path.insert(0, 'src')   # so we can import our modules directly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')"""),

# --- 1. load data ---
cell("## 1. Load LCDB Data", "markdown"),
cell("""from data_loader import load_lcdb

# set local_path if running outside colab, e.g. local_path='./lc_summary.csv'
df = load_lcdb()
df.head(3)"""),

# --- 2. filters ---
cell("## 2. Apply Filters", "markdown"),
cell("""from filters import apply_filters, summarize_period_stats

full_df, diam_df, ap_df = apply_filters(df)

summarize_period_stats(full_df, label='full lcdb')
summarize_period_stats(diam_df, label='diameter filtered (35-100 km)')
summarize_period_stats(ap_df, label='cindygraber-like (ap_df)')"""),

# --- 3. distributions ---
cell("## 3. Period Distribution Plots", "markdown"),
cell("""from plotting import plot_period_distribution, plot_period_vs_diam

# full lcdb distribution
ax = plot_period_distribution(
    full_df,
    title='Period Frequency Distribution for LCDB Asteroids',
    save_path='outputs/fig_period_dist_full.png'
)
plt.show()"""),

cell("""# cindygraber-like population (ap_df)
ax = plot_period_distribution(
    ap_df,
    title='Period Frequency Distribution for Filtered LCDB Asteroids\\n'
          '(35 km < Diam < 100 km, Albedo ≤ 0.075)',
    save_path='outputs/fig_period_dist_filtered.png'
)
plt.show()"""),

cell("""# period vs diameter scatter
plot_period_vs_diam(
    df, title='Asteroid Period vs Diameter (Hue: Albedo)',
    save_path='outputs/fig_scatter_period_diam.png'
)
plt.show()"""),

# --- 4. phase coverage ---
cell("## 4. Phase Coverage Simulation", "markdown"),
cell("""from observing_strategy import calculate_nights_to_solve, compute_cdf_roi

ap_df = ap_df.copy()

# deterministic simulation (fixed seed, single weather draw)
results = ap_df['Period'].apply(
    calculate_nights_to_solve, clear_fraction=0.20
)
ap_df[['nights_required', 'clear_nights']] = pd.DataFrame(
    results.tolist(), index=ap_df.index
)

print(ap_df[['Name', 'Period', 'nights_required']].head(10))"""),

# --- 5. MC weather ---
cell("## 5. Monte Carlo Weather Simulation (k=1000)", "markdown"),
cell("""from observing_strategy import run_mc_weather_simulation

mean_nights, std_nights = run_mc_weather_simulation(
    ap_df['Period'].values, n_trials=1000, clear_fraction=0.20
)

ap_df['mc_mean_nights'] = mean_nights
ap_df['mc_std_nights']  = std_nights

# compute optimistic / pessimistic bands
nights_mean = ap_df['mc_mean_nights'].dropna()
nights_std  = ap_df['mc_std_nights'].dropna()
nights_opt  = (nights_mean - nights_std).clip(lower=1)
nights_pess = nights_mean + nights_std

print(ap_df[['Name', 'Period', 'mc_mean_nights', 'mc_std_nights']].head(10))"""),

# --- 6. CDF/ROI ---
cell("## 6. CDF and ROI Metrics", "markdown"),
cell("""from observing_strategy import compute_cdf_roi, plot_cdf_roi

roi_df = compute_cdf_roi(nights_mean)

# compute band cdfs from pessimistic/optimistic
from observing_strategy import MAX_NIGHTS
x = np.arange(1, MAX_NIGHTS + 1)
n_total = len(nights_opt)
cdf_opt  = np.array([(nights_opt  <= n).sum() / n_total for n in x])
cdf_pess = np.array([(nights_pess <= n).sum() / n_total for n in x])

fig, ax1, ax2 = plot_cdf_roi(
    roi_df,
    title='Observing Strategy Metrics: CDF vs ROI for Filtered LCDB Asteroids w/ MC Weather',
    cdf_lower=cdf_pess,
    cdf_upper=cdf_opt,
    save_path='outputs/fig_cdf_roi_mc.png'
)
plt.show()"""),

cell("""# deterministic (no-weather) cdf for comparison
roi_df_det = compute_cdf_roi(ap_df['nights_required'].dropna())

fig, ax1, ax2 = plot_cdf_roi(
    roi_df_det,
    title='Observing Strategy Metrics: CDF vs ROI for Filtered LCDB Asteroids',
    save_path='outputs/fig_cdf_roi_deterministic.png'
)
plt.show()"""),

# --- 7. lightcurve sim ---
cell("## 7. Synthetic Lightcurve and Period Recovery", "markdown"),
cell("""from lightcurve_sim import (
    build_synthetic_datasets, run_ls_rms_analysis, plot_period_evolution
)

# true period: median of the ap_df population
true_period = ap_df['Period'].median()
print(f'using true period: {true_period:.2f} h')

np.random.seed(42)
datasets = build_synthetic_datasets(
    true_period=true_period,
    max_nights=4,
    obs_per_night=30,
    noise_scale=0.05,
)

print('\\nperiod recovery results:')
analysis_results = run_ls_rms_analysis(datasets, period_min=2.0, period_max=24.0)"""),

cell("""fig = plot_period_evolution(
    datasets, analysis_results, true_period,
    save_path='outputs/fig_period_evolution_grid.png'
)
plt.show()"""),

# --- 8. convergence ---
cell("## 8. LS Convergence and Stopping Criterion", "markdown"),
cell("""from convergence import (
    run_convergence_simulation, find_stability_threshold,
    plot_convergence_spaghetti
)

conv_df = run_convergence_simulation(
    ap_df, max_nights=15, obs_per_night=50, noise_scale=0.01
)

ax = plot_convergence_spaghetti(
    conv_df, save_path='outputs/fig_convergence_spaghetti.png'
)
plt.show()"""),

cell("""# find the delta-P threshold that predicts 95% accuracy
threshold = find_stability_threshold(conv_df, target_accuracy=0.95)"""),

# --- 9. bootstrap ---
cell("## 9. Bootstrapped CDF with 95% CI", "markdown"),
cell("""from bootstrap_analysis import run_bootstrap_cdf, plot_bootstrap_cdf

nights_int = ap_df['nights_required'].dropna().astype(int).values

roi_boot = run_bootstrap_cdf(nights_int, n_boot=1000)

fig, ax1, ax2 = plot_bootstrap_cdf(
    roi_boot,
    title='Bootstrapped Observing Strategy Optimization (n=1000)',
    save_path='outputs/fig_bootstrap_cdf.png'
)
plt.show()"""),

cell("""# export final metrics to csv
roi_boot.to_csv('outputs/bootstrap_roi_metrics.csv')
print("saved: outputs/bootstrap_roi_metrics.csv")"""),

]


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = "notebooks/lcdb_observing_strategy.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"notebook written to {out_path}")
