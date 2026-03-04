"""
convergence.py
--------------
simulates how quickly the lomb-scargle period estimate converges toward the true
value as more nights of data are accumulated. this is distinct from the phase-coverage
model in observing_strategy.py:

  phase coverage (observing_strategy.py) — physical question:
    "have we sampled enough of the asteroid's rotation to constrain its period?"

  period convergence (this module) — computational question:
    "has the lomb-scargle estimate stabilized to within an acceptable error?"

also implements a delta-p stopping criterion: stop observing when the
night-to-night change in best period estimate falls below a threshold,
which can serve as an operational guide during a real campaign.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.timeseries import LombScargle

from lightcurve_sim import generate_lightcurve, get_random_times


# convergence threshold: estimate is "solved" when |est - true| < this value
CONVERGENCE_THRESHOLD_H = 0.1   # hours


def run_convergence_simulation(ap_df, max_nights=15, obs_per_night=50,
                                noise_scale=0.01, observing_hours=7.0,
                                period_min=2.0, period_max=24.0,
                                n_freq=5000, quantile_bounds=(0.15, 0.85),
                                random_state=42):
    """
    incremental lomb-scargle convergence simulation over max_nights.
    for each target, records the nightly period estimate and its absolute error.

    samples targets within (quantile_bounds[0], quantile_bounds[1]) of the
    period distribution to focus on the iqr of the population.

    args:
        ap_df: filtered dataframe (cindygraber-like population)
        max_nights: number of nights to simulate per target
        obs_per_night: observations per night
        noise_scale: gaussian magnitude noise sigma
        observing_hours: usable hours per night
        period_min/max: bounds for ls frequency grid
        n_freq: number of points in frequency grid
        quantile_bounds: (low, high) quantile bounds to sample targets from
        random_state: rng seed for reproducibility

    returns:
        pd.DataFrame with columns:
            name, true_period, night, estimated_period, abs_error, delta_p
    """
    # build frequency grid
    freq_grid = np.linspace(1.0 / period_max, 1.0 / period_min, n_freq)

    # select targets in the iqr
    q_low  = ap_df['Period'].quantile(quantile_bounds[0])
    q_high = ap_df['Period'].quantile(quantile_bounds[1])
    targets = ap_df[
        (ap_df['Period'] > q_low) & (ap_df['Period'] < q_high)
    ].sample(frac=1.0, random_state=random_state)

    print(f"simulating convergence for {len(targets)} targets over {max_nights} nights...")

    history = []

    for _, row in targets.iterrows():
        true_p = row['Period']
        name   = row['Name']

        # pre-generate all nights
        t_all = [
            get_random_times(n * 24.0, observing_hours, obs_per_night)
            for n in range(max_nights)
        ]

        current_t   = np.array([])
        current_mag = np.array([])
        prev_est_p  = None

        for night_idx in range(max_nights):
            night_num = night_idx + 1
            t_new = t_all[night_idx]
            mag_new = (generate_lightcurve(t_new, true_p)
                       + np.random.normal(0, noise_scale, len(t_new)))

            current_t   = np.concatenate([current_t, t_new])
            current_mag = np.concatenate([current_mag, mag_new])

            # lomb-scargle
            ls = LombScargle(current_t, current_mag)
            power = ls.power(freq_grid)
            best_freq = freq_grid[np.argmax(power)]
            est_p = 1.0 / best_freq

            delta_p = abs(est_p - prev_est_p) if prev_est_p is not None else np.nan
            prev_est_p = est_p

            history.append({
                'name': name,
                'true_period': true_p,
                'night': night_num,
                'estimated_period': est_p,
                'abs_error': abs(est_p - true_p),
                'delta_p': delta_p,
            })

    print("convergence simulation complete.")
    return pd.DataFrame(history)


def find_stability_threshold(conv_df, target_accuracy=0.95,
                              min_samples=20, n_thresholds=500):
    """
    scans delta_p thresholds (large to small) to find the largest value at which
    the ls estimate is correct (abs_error < 0.1h) >= target_accuracy of the time.

    this defines an operational stopping rule:
      "stop observing when night-to-night period change < threshold."

    args:
        conv_df: dataframe from run_convergence_simulation()
        target_accuracy: required fraction correct (default 0.95)
        min_samples: minimum sample size to trust a threshold
        n_thresholds: number of threshold values to scan

    returns:
        float threshold value if found, else None
    """
    valid = conv_df.dropna(subset=['delta_p', 'abs_error']).copy()
    thresholds = np.logspace(0, -4, n_thresholds)  # scans from 1.0 down to 0.0001

    best_threshold = None
    best_rate = 0.0

    print(f"{'threshold (h)':<15} | {'success rate':<14} | {'n_samples':<10}")
    print("-" * 45)

    for t in thresholds:
        subset = valid[valid['delta_p'] <= t]
        if len(subset) < min_samples:
            continue

        success = (subset['abs_error'] < CONVERGENCE_THRESHOLD_H).sum()
        rate = success / len(subset)

        if rate > best_rate:
            best_rate = rate
            best_rate_threshold = t

        if rate >= target_accuracy:
            best_threshold = t
            print(f"{t:<15.5f} | {rate:<14.2%} | {len(subset):<10}  <-- optimal cutoff")
            break

    if best_threshold is None:
        print(f"\ncould not reach {target_accuracy:.0%} accuracy.")
        print(f"max accuracy: {best_rate:.2%} at threshold <= {best_rate_threshold:.5f} h")
        print("recommendation: use delta_p as a pre-filter; aliasing checks still needed.")
    else:
        print(f"\noperational rule: stop observing when night-to-night "
              f"period change < {best_threshold:.4f} hours.")

    return best_threshold


def plot_convergence_spaghetti(conv_df, save_path=None):
    """
    spaghetti plot showing abs_error over nights for all targets,
    overlaid with median and iqr.

    args:
        conv_df: dataframe from run_convergence_simulation()
        save_path: optional save path

    returns:
        matplotlib axes
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10, 6))

    # individual tracks (low alpha)
    sns.lineplot(data=conv_df, x='night', y='abs_error', hue='name',
                 palette='viridis', alpha=0.12, linewidth=0.7,
                 legend=False, ax=ax)

    # median and iqr overlay
    stats = conv_df.groupby('night')['abs_error']
    median_err = stats.median()
    q1_err = stats.quantile(0.25)
    q3_err = stats.quantile(0.75)

    ax.plot(median_err.index, median_err.values,
            color='black', linewidth=2.5, linestyle='--', label='Median Error')
    ax.fill_between(median_err.index, q1_err, q3_err,
                    color='gray', alpha=0.45, label='IQR (25-75%)')
    ax.axhline(0.05, color='#d62728', linestyle=':', linewidth=2,
               label='Success Threshold (0.05h)')

    ax.set_yscale('log')
    ax.set_title('Dynamic Period Prediction Convergence', fontweight='bold')
    ax.set_xlabel('Observing Night')
    ax.set_ylabel('Absolute Error (Hours)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, which="major", ls="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return ax
