"""
observing_strategy.py
---------------------
core analysis module for the cindygraber observing campaign.

contains:
  - calculate_nights_to_solve : deterministic phase-coverage model per asteroid
  - run_mc_weather_simulation : monte carlo weather uncertainty (k=1000 trials)
  - compute_cdf_roi            : cumulative distribution + efficiency/marginal-gain metrics
  - plot_cdf_roi               : final annotated cdf/roi dual-axis figure

the phase-coverage model tracks 96 phase bins and requires:
  - >= 85% of bins covered  (min_phase_coverage)
  - >= 2 full period cycles elapsed  (min_periods_covered)
both conditions must be satisfied simultaneously before a night is counted as "solved".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


# --- simulation defaults ---
MAX_NIGHTS           = 100
OBSERVING_HOURS      = 7.0    # hours per night
NIGHT_INTERVAL       = 24.0   # hours between night starts
CLEAR_FRACTION       = 0.20   # fraction of nights with clear weather
MIN_PHASE_COVERAGE   = 0.85   # fraction of phase bins that must be covered
MIN_PERIODS_COVERED  = 2.0    # minimum number of complete periods elapsed
N_PHASE_BINS         = 96


def calculate_nights_to_solve(
    period,
    bins=N_PHASE_BINS,
    max_nights=MAX_NIGHTS,
    observing_hours=OBSERVING_HOURS,
    night_interval=NIGHT_INTERVAL,
    min_phase_coverage=MIN_PHASE_COVERAGE,
    min_periods_covered=MIN_PERIODS_COVERED,
    clear_fraction=CLEAR_FRACTION,
    seed=42,
):
    """
    simulates how many consecutive nights are needed to achieve the phase coverage
    criterion for a single asteroid period, given random weather.

    args:
        period: rotation period in hours
        bins: number of phase bins to track
        max_nights: simulation cap (returns this if criterion never met)
        observing_hours: usable hours per night
        night_interval: hours from start of one night to the next
        min_phase_coverage: fraction of bins required to be covered
        min_periods_covered: minimum number of full rotation cycles required
        clear_fraction: probability that any given night is clear
        seed: rng seed for reproducibility (vary per mc trial)

    returns:
        tuple (total_nights, clear_nights_count)
    """
    if pd.isna(period) or period <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    covered_phases = np.zeros(bins, dtype=bool)
    total_observed_hours = 0.0
    clear_nights_count = 0

    for night in range(1, max_nights + 1):
        # weather roll
        if rng.random() >= clear_fraction:
            continue  # cloudy; time passes but no data collected

        clear_nights_count += 1
        t_start = (night - 1) * night_interval
        t_end = t_start + observing_hours
        total_observed_hours += observing_hours

        # map observation window to phase bins
        if observing_hours >= period:
            covered_phases[:] = True  # entire phase covered in one night
        else:
            phase_start = (t_start % period) / period
            phase_end = (t_end % period) / period
            bin_start = int(phase_start * bins)
            bin_end = int(phase_end * bins)

            if bin_start <= bin_end:
                covered_phases[bin_start:bin_end + 1] = True
            else:
                # wraps around 0/1 boundary
                covered_phases[bin_start:] = True
                covered_phases[:bin_end + 1] = True

        # check success criteria
        phase_frac = np.mean(covered_phases)
        time_frac = total_observed_hours / period

        if phase_frac >= min_phase_coverage and time_frac >= min_periods_covered:
            return night, clear_nights_count

    return max_nights, clear_nights_count


def run_mc_weather_simulation(periods_array, n_trials=1000,
                               clear_fraction=CLEAR_FRACTION):
    """
    runs n_trials independent monte carlo weather simulations for each period
    in periods_array, returning the mean and std of nights required across trials.

    args:
        periods_array: 1d array of rotation periods in hours
        n_trials: number of mc trials per asteroid (k=1000 recommended)
        clear_fraction: fraction of clear nights to use in each trial

    returns:
        tuple (mean_nights, std_nights) — each a 1d numpy array matching periods_array
    """
    n_asteroids = len(periods_array)
    results_matrix = np.zeros((n_asteroids, n_trials))

    print(f"running mc weather simulation: {n_asteroids} asteroids x {n_trials} trials...")

    for i in range(n_trials):
        trial_results = [
            calculate_nights_to_solve(p, seed=i, clear_fraction=clear_fraction)
            for p in periods_array
        ]
        results_matrix[:, i] = [r[0] for r in trial_results]

        if (i + 1) % 100 == 0:
            print(f"  trial {i + 1}/{n_trials} complete")

    mean_nights = np.mean(results_matrix, axis=1)
    std_nights = np.std(results_matrix, axis=1)
    print("mc simulation complete.")
    return mean_nights, std_nights


def compute_cdf_roi(nights_series, max_nights=MAX_NIGHTS):
    """
    computes the cumulative distribution function (cdf) of nights required,
    plus efficiency (cdf/night) and marginal gain (delta cdf) metrics.

    args:
        nights_series: pd.Series or 1d array of nights_required values
        max_nights: upper bound for x-axis

    returns:
        pd.DataFrame indexed by night (1..max_nights) with columns:
            cumulative_probability, efficiency, marginal_gain
    """
    x = np.arange(1, max_nights + 1)
    total = len(nights_series)
    cdf_vals = np.array([(nights_series <= n).sum() / total for n in x])

    roi_df = pd.DataFrame({
        'cumulative_probability': cdf_vals,
    }, index=x)
    roi_df.index.name = 'night'

    roi_df['efficiency']     = roi_df['cumulative_probability'] / roi_df.index
    roi_df['marginal_gain']  = roi_df['cumulative_probability'].diff()
    roi_df.loc[roi_df.index[0], 'marginal_gain'] = roi_df.iloc[0]['cumulative_probability']

    return roi_df


def plot_cdf_roi(roi_df, title="Observing Strategy Metrics: CDF vs ROI",
                 figsize=(12, 7), save_path=None,
                 cdf_lower=None, cdf_upper=None):
    """
    dual-axis plot: left = cdf (fraction of population solved),
    right = efficiency and marginal gain.

    annotates peak efficiency and 80% solved milestones automatically.

    args:
        roi_df: dataframe from compute_cdf_roi() (or run_bootstrap_cdf)
        title: plot title
        figsize: figure size tuple
        save_path: optional path to save the figure
        cdf_lower: optional lower ci band (1d array matching roi_df.index)
        cdf_upper: optional upper ci band

    returns:
        (fig, ax1, ax2) tuple
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(True, which="both", linestyle='--', color='gray', alpha=0.2)

    # left axis: cdf
    cdf_col = 'cumulative_probability' if 'cumulative_probability' in roi_df.columns \
              else 'Cumulative_Probability'
    l1 = ax1.plot(roi_df.index, roi_df[cdf_col],
                  color='#1f77b4', linewidth=3, label='Mean CDF')

    if cdf_lower is not None and cdf_upper is not None:
        ax1.fill_between(roi_df.index, cdf_lower, cdf_upper,
                         color='#1f77b4', alpha=0.2, label='±1σ Weather Variance')

    ax1.axhline(0.8, color='gray', linestyle=':', linewidth=1.5, label='80% Target')
    ax1.set_ylabel('Fraction of population with 85% Phase Coverage',
                   color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('log')
    ax1.set_xlabel('Consecutive Observing Nights (7h)', fontsize=12)
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    # right axis: roi metrics
    eff_col = 'efficiency' if 'efficiency' in roi_df.columns else 'Efficiency'
    mg_col  = 'marginal_gain' if 'marginal_gain' in roi_df.columns else 'Marginal_Gain'

    ax2 = ax1.twinx()
    l2 = ax2.plot(roi_df.index, roi_df[eff_col],
                  color='green', linestyle='--', linewidth=2,
                  label='Efficiency (CDF/Nights)')
    l3 = ax2.plot(roi_df.index, roi_df[mg_col],
                  color='orange', linestyle=':', linewidth=2,
                  label=r'Marginal Gain ($\Delta$ Prob)')
    ax2.set_ylabel('ROI Metrics', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 0.3)

    # annotate peak efficiency
    eff_vals = roi_df[eff_col].values
    peak_night = roi_df.index[np.argmax(eff_vals)]
    peak_val = eff_vals[np.argmax(eff_vals)]
    ax2.axvline(peak_night, color='green', alpha=0.3, linewidth=1)
    ax2.text(peak_night, peak_val + 0.01,
             f'Peak Eff.\n(Night {peak_night})',
             color='green', ha='left', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # annotate 80% milestone
    cdf_arr = roi_df[cdf_col].values
    idx_80 = np.where(cdf_arr >= 0.80)[0]
    if len(idx_80) > 0:
        night_80 = roi_df.index[idx_80[0]]
        ax1.axvline(night_80, color='#1f77b4', linewidth=1)
        ax1.text(night_80, 0.65,
                 f'Mean 80%\n(Night {night_80})',
                 color='#1f77b4', ha='center', va='bottom', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # combined legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    if cdf_lower is not None:
        patch = mpatches.Patch(facecolor='#1f77b4', alpha=0.2, label='±1σ Weather Variance')
        lines.insert(1, patch)
        labels.insert(1, '±1σ Weather Variance')
    lines.append(plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5))
    labels.append('80% Target')
    ax1.legend(lines, labels, loc='best', fontsize=10, frameon=True)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return fig, ax1, ax2
