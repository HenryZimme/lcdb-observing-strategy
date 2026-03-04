"""
bootstrap_analysis.py
---------------------
bootstrapped estimate of the observing campaign cdf with 95% confidence intervals.

the bootstrap resamples the 'nights_required' distribution (from observing_strategy.py)
with replacement over n_boot=1000 epochs to produce a robust mean cdf and
confidence intervals that account for finite sample size in the lcdb filter population.

this is used for the final publication-quality figure (figure 7 in the proposal).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from sklearn.utils import resample


def run_bootstrap_cdf(nights_data, n_boot=1000, max_nights=100, seed=42):
    """
    bootstraps the cdf of nights required with 95% confidence intervals.

    args:
        nights_data: 1d int array of nights_required values (from observing_strategy)
        n_boot: number of bootstrap epochs (1000 recommended)
        max_nights: upper x-axis bound
        seed: numpy random seed

    returns:
        pd.DataFrame indexed by night (1..max_nights) with columns:
            cumulative_probability, cdf_lower (2.5%), cdf_upper (97.5%),
            efficiency, marginal_gain
    """
    n_samples = len(nights_data)
    boot_cdfs = np.zeros((n_boot, max_nights))

    print(f"running {n_boot} bootstrap epochs on {n_samples} samples...")
    np.random.seed(seed)

    for i in range(n_boot):
        sample = resample(nights_data, n_samples=n_samples,
                          replace=True, random_state=i)
        # bincount approach: fast cdf from integer night values
        counts = np.bincount(sample, minlength=max_nights + 1)
        cumsum = np.cumsum(counts)
        cdf_sample = cumsum / n_samples
        boot_cdfs[i, :] = cdf_sample[1:max_nights + 1]

    cdf_mean = np.mean(boot_cdfs, axis=0)
    cdf_low  = np.percentile(boot_cdfs, 2.5,  axis=0)
    cdf_high = np.percentile(boot_cdfs, 97.5, axis=0)

    nights_idx = np.arange(1, max_nights + 1)
    roi_df = pd.DataFrame({
        'cumulative_probability': cdf_mean,
        'cdf_lower': cdf_low,
        'cdf_upper': cdf_high,
    }, index=nights_idx)
    roi_df.index.name = 'night'

    roi_df['efficiency']    = roi_df['cumulative_probability'] / roi_df.index
    roi_df['marginal_gain'] = roi_df['cumulative_probability'].diff().fillna(
        roi_df.iloc[0]['cumulative_probability']
    )

    print("bootstrap complete.")
    print(f"  peak efficiency night : {roi_df['efficiency'].idxmax()}")
    night_80_idx = roi_df[roi_df['cumulative_probability'] >= 0.80].index
    if len(night_80_idx) > 0:
        print(f"  80% solved by night   : {night_80_idx[0]}")

    return roi_df


def plot_bootstrap_cdf(roi_df, figsize=(10, 6), title=None, save_path=None):
    """
    publication-quality dual-axis plot of bootstrapped cdf with 95% ci,
    overlaid with efficiency and marginal gain.

    args:
        roi_df: dataframe from run_bootstrap_cdf()
        figsize: figure size
        title: plot title (default generated automatically)
        save_path: optional path to save the figure

    returns:
        (fig, ax1, ax2) tuple
    """
    from plotting import set_pub_style
    set_pub_style()

    if title is None:
        n_boot_label = "bootstrapped"
        title = f"Observing Strategy Optimization ({n_boot_label} CDF)"

    fig, ax1 = plt.subplots(figsize=figsize)

    # left axis: bootstrapped cdf
    l1 = ax1.plot(roi_df.index, roi_df['cumulative_probability'],
                  color='#1f77b4', linewidth=3, label='Mean CDF')
    ax1.fill_between(roi_df.index,
                     roi_df['cdf_lower'],
                     roi_df['cdf_upper'],
                     color='#1f77b4', alpha=0.2)

    ax1.set_ylabel('Fraction of Population Solved',
                   color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='major', linestyle='--', alpha=0.3)

    # right axis: roi metrics
    ax2 = ax1.twinx()
    l2 = ax2.plot(roi_df.index, roi_df['efficiency'],
                  color='#2ca02c', linewidth=2, linestyle='--',
                  label='Efficiency (CDF/Night)')
    l3 = ax2.plot(roi_df.index, roi_df['marginal_gain'],
                  color='#ff7f0e', linewidth=2, linestyle=':',
                  label=r'Marginal Gain ($\Delta$ Prob)')
    ax2.set_ylabel('ROI Metrics (per Night)', color='#333', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#333')
    ax2.set_ylim(0, 0.25)

    # annotate milestones
    peak_night = roi_df['efficiency'].idxmax()
    night_80_idx = roi_df[roi_df['cumulative_probability'] >= 0.80].index
    night_80 = night_80_idx[0] if len(night_80_idx) > 0 else None

    ax1.axvline(peak_night, color='#2ca02c', alpha=0.4, linewidth=1)
    ax1.text(peak_night, 0.55,
             f'Peak Eff.\n(Night {peak_night})',
             color='#2ca02c', fontsize=9, ha='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    if night_80 is not None:
        ax1.axvline(night_80, color='#1f77b4', alpha=0.4, linewidth=1)
        ax1.text(night_80, 0.82,
                 f'80% Solved\n(Night {night_80})',
                 color='#1f77b4', fontsize=9, ha='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # combined legend
    patch_ci = Patch(facecolor='#1f77b4', alpha=0.2, label='95% CI')
    lines = l1 + l2 + l3
    ax1.legend(lines + [patch_ci],
               [l.get_label() for l in lines] + ['95% CI'],
               loc='center right', frameon=True, fontsize=10,
               bbox_to_anchor=(0.95, 0.5), facecolor='white', framealpha=0.95)

    # formatting
    ax1.set_xscale('log')
    ax1.set_xlabel('Consecutive Observing Nights (7h)', fontweight='bold')
    ax1.set_title(title, fontweight='bold', y=1.02)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xticks([1, 2, 5, 10, 20, 50, 100])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return fig, ax1, ax2
