"""
plotting.py
-----------
visualization functions for lcdb period distributions and asteroid scatter plots.
all functions accept a dataframe and optional matplotlib axes; they return the axes
so callers can annotate further or save.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# constants shared across plots
SPIN_BARRIER_H = 2.2
CINDYGRABER_DIAM_KM = 38.46


def set_pub_style():
    """applies publication-quality rcparams (minor planet bulletin style)."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)


def add_period_stats(data, ax=None, show_spin_barrier=True):
    """
    overlays median, q1, q3, and optional spin barrier vertical lines on a period plot.

    args:
        data: dataframe with 'Period' column
        ax: matplotlib axes (defaults to current axes)
        show_spin_barrier: if true, draws the 2.2h spin barrier line
    """
    if ax is None:
        ax = plt.gca()

    series = data['Period'].dropna()
    if series.empty:
        return

    median_p = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    ax.axvline(median_p, color='purple', linestyle='-', linewidth=2.5,
               label=f'Median ({median_p:.2f}h)')
    ax.axvline(q1, color='green', linestyle=':', linewidth=2,
               label=f'Q1 ({q1:.2f}h)')
    ax.axvline(q3, color='green', linestyle=':', linewidth=2,
               label=f'Q3 ({q3:.2f}h)')

    if show_spin_barrier:
        ax.axvline(SPIN_BARRIER_H, color='red', linestyle='--', linewidth=2.5,
                   label=f'Spin Barrier (~{SPIN_BARRIER_H}h)')


def _add_stats_box(ax, periods):
    """adds an n/median/iqr text box to the upper-left of an axes."""
    n = len(periods)
    median_p = periods.median()
    q1 = periods.quantile(0.25)
    q3 = periods.quantile(0.75)
    iqr = q3 - q1
    text = (f"N = {n:,}\n"
            f"Median = {median_p:.2f} h\n"
            f"IQR = {iqr:.2f} h\n"
            f"(Q1={q1:.2f}, Q3={q3:.2f})")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)


def plot_period_distribution(df, title, bins=50, figsize=(12, 7),
                             save_path=None, ax=None):
    """
    plots a log-x period frequency histogram with spin barrier and stats overlay.

    args:
        df: dataframe with 'Period' column
        title: plot title string
        bins: number of histogram bins
        figsize: figure size tuple
        save_path: if provided, saves the figure to this path
        ax: existing matplotlib axes (creates new figure if None)

    returns:
        matplotlib axes
    """
    periods = df['Period'].dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    set_pub_style()
    sns.histplot(periods, log_scale=True, kde=True,
                 bins=round(np.sqrt(len(periods))),
                 element="step", color='#4C72B0', alpha=0.3,
                 label='Observed Distribution', ax=ax)

    add_period_stats(df, ax=ax)

    # shade fast-rotator zone
    ax.axvspan(periods.min(), SPIN_BARRIER_H, color='red', alpha=0.1)
    ax.text(SPIN_BARRIER_H * 0.55, ax.get_ylim()[1] * 0.05,
            'Fast Rotators\n(Monoliths)',
            color='darkred', ha='center', fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Rotation Period (hours)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2)

    # clean log x-axis ticks
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.1, 1, 10, 100, 1000])

    _add_stats_box(ax, periods)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return ax


def plot_period_vs_diam(df, title="Asteroid Period vs Diameter",
                        figsize=(15, 9), show_cindygraber=True,
                        save_path=None):
    """
    scatter plot of period vs diameter, colored by albedo.

    args:
        df: dataframe with 'Period', 'Diam', 'Albedo' columns
        title: plot title
        figsize: figure size
        show_cindygraber: if true, draws a horizontal line at cindygraber's diameter
        save_path: optional save path

    returns:
        matplotlib axes
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(data=df, x='Period', y='Diam', hue='Albedo',
                    palette='flare', alpha=0.6, ax=ax)
    add_period_stats(df, ax=ax)

    if show_cindygraber:
        ax.axhline(CINDYGRABER_DIAM_KM, linestyle='-', color='purple',
                   label=f'Cindygraber diameter ({CINDYGRABER_DIAM_KM} km)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Period (hours)', fontsize=13)
    ax.set_ylabel('Diameter (km)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return ax
