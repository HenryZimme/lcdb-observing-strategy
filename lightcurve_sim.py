"""
lightcurve_sim.py
-----------------
synthetic lightcurve generation and period-recovery analysis.

the lightcurve model is a double-peaked fourier series (fundamental + 2nd harmonic),
which approximates the shape of a triaxial ellipsoid rotating in sunlight.

two period-recovery methods are implemented:
  - lomb-scargle (ls): standard power spectrum analysis
  - fourier rms scan: brute-force 2nd-order fourier fit over a period grid;
                      best period = minimum rms residual

the 4x4 diagnostic grid (plot_period_evolution) shows how each method
converges as more nights are added.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from astropy.timeseries import LombScargle


# lightcurve model parameters
AMPLITUDE_FUND = 0.15   # coefficient of fundamental harmonic
AMPLITUDE_2ND  = 0.10   # coefficient of second harmonic


def generate_lightcurve(t, period):
    """
    generates a noiseless double-peaked lightcurve.

    model: mag(t) = -A*cos(2pi*t/P) - B*cos(4pi*t/P)
    this produces two minima per rotation period (double-peaked).

    args:
        t: array of observation times in hours
        period: rotation period in hours

    returns:
        numpy array of relative magnitudes (zero mean)
    """
    mag = (
        -AMPLITUDE_FUND * np.cos(2 * np.pi * t / period)
        - AMPLITUDE_2ND * np.cos(4 * np.pi * t / period)
    )
    return mag


def get_random_times(start_h, duration_h, n_points):
    """returns sorted random observation timestamps within a window."""
    return np.sort(np.random.uniform(start_h, start_h + duration_h, n_points))


def build_synthetic_datasets(true_period, max_nights=5,
                              obs_per_night=30, noise_scale=0.05,
                              observing_hours=7.0):
    """
    builds cumulative multi-night synthetic datasets for a single asteroid.
    each dataset adds one more night of noisy observations.

    args:
        true_period: known rotation period (hours)
        max_nights: number of nights to simulate
        obs_per_night: observations per night
        noise_scale: gaussian magnitude noise (sigma)
        observing_hours: usable hours per night

    returns:
        dict mapping dataset label (e.g. 'Night 1', 'Night 1+2') to
        {'time': array, 'mag': array}
    """
    # generate all nights' timestamps and magnitudes upfront
    nights = {}
    for n in range(max_nights):
        t_start = n * 24.0
        t = get_random_times(t_start, observing_hours, obs_per_night)
        mag_clean = generate_lightcurve(t, true_period)
        mag_noisy = mag_clean + np.random.normal(0, noise_scale, len(t))
        nights[n] = {'t': t, 'mag': mag_noisy}

    # build cumulative datasets
    datasets = {}
    for k in range(1, max_nights + 1):
        label_parts = ['Night ' + str(i + 1) for i in range(k)]
        label = label_parts[0] if k == 1 else '+'.join(label_parts)
        # flatten cumulative arrays
        all_t   = np.concatenate([nights[i]['t']   for i in range(k)])
        all_mag = np.concatenate([nights[i]['mag'] for i in range(k)])
        datasets[label] = {'time': all_t, 'mag': all_mag}

    return datasets


def calculate_fourier_rms(t, mag, period):
    """
    fits a 2nd-order fourier model to the data at a given trial period
    and returns the rms of the residuals.

    model: c + a1*cos(wt) + b1*sin(wt) + a2*cos(2wt) + b2*sin(2wt)

    args:
        t: observation times (hours)
        mag: observed magnitudes
        period: trial period (hours)

    returns:
        rms residual (float)
    """
    omega = 2 * np.pi / period
    x = np.column_stack([
        np.ones_like(t),
        np.cos(omega * t),
        np.sin(omega * t),
        np.cos(2 * omega * t),
        np.sin(2 * omega * t),
    ])
    coeffs, resid_sum_sq, _, _ = np.linalg.lstsq(x, mag, rcond=None)

    if len(resid_sum_sq) > 0:
        rss = resid_sum_sq[0]
    else:
        # fallback for underdetermined systems
        model = x @ coeffs
        rss = np.sum((mag - model) ** 2)

    return np.sqrt(rss / len(mag))


def run_ls_rms_analysis(datasets, period_min=2.0, period_max=24.0,
                         n_steps=5000):
    """
    runs lomb-scargle and fourier rms period recovery on each dataset.

    args:
        datasets: dict from build_synthetic_datasets()
        period_min: minimum trial period (hours)
        period_max: maximum trial period (hours)
        n_steps: number of grid points in period search

    returns:
        dict with same keys as datasets, each containing:
            {'periods', 'ls_power', 'rms_values',
             'best_ls_period', 'best_rms_period'}
    """
    periods_grid = np.linspace(period_min, period_max, n_steps)
    freqs_grid = 1.0 / periods_grid

    results = {}
    for name, data in datasets.items():
        t = data['time']
        y = data['mag']

        # lomb-scargle
        ls = LombScargle(t, y)
        power = ls.power(freqs_grid)
        best_ls_period = periods_grid[np.argmax(power)]

        # rms scan
        rms_vals = np.array([calculate_fourier_rms(t, y, p) for p in periods_grid])
        best_rms_period = periods_grid[np.argmin(rms_vals)]

        results[name] = {
            'periods': periods_grid,
            'ls_power': power,
            'rms_values': rms_vals,
            'best_ls_period': best_ls_period,
            'best_rms_period': best_rms_period,
        }

        print(f"{name:<20}  LS: {best_ls_period:.3f}h  RMS: {best_rms_period:.3f}h")

    return results


def _plot_phased(ax, t, mag, period, true_period, label_y=True, label_x=False):
    """helper: plots a phased lightcurve (2 cycles) with true model overlay."""
    phase = (t % period) / period

    # observed data
    ax.scatter(phase,         mag, alpha=0.5, s=20, color='#3C5488', edgecolor='none')
    ax.scatter(phase + 1.0,   mag, alpha=0.5, s=20, color='#3C5488', edgecolor='none')

    # true model
    phase_model = np.linspace(0, 1, 100)
    mag_model = generate_lightcurve(phase_model * true_period, true_period)
    ax.plot(phase_model,        mag_model, color='#E64B35', linewidth=1.5, alpha=0.9)
    ax.plot(phase_model + 1.0,  mag_model, color='#E64B35', linewidth=1.5, alpha=0.9)

    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    if label_y:
        ax.set_ylabel("Magnitude")
    if label_x:
        ax.set_xlabel("Phase")


def plot_period_evolution(datasets, analysis_results, true_period,
                          figsize=(16, 14), save_path=None):
    """
    produces a (n_datasets x 4) diagnostic grid showing:
      col 0: phased lightcurve at ls best period
      col 1: phased lightcurve at rms best period
      col 2: lomb-scargle periodogram
      col 3: rms scan

    args:
        datasets: dict from build_synthetic_datasets()
        analysis_results: dict from run_ls_rms_analysis()
        true_period: known rotation period for reference lines
        figsize: figure size
        save_path: optional output path

    returns:
        matplotlib figure
    """
    dataset_names = list(datasets.keys())
    n_rows = len(dataset_names)

    fig, axes = plt.subplots(n_rows, 4, figsize=figsize, constrained_layout=True)

    for row_idx, name in enumerate(dataset_names):
        data = datasets[name]
        results = analysis_results[name]

        t = data['time']
        mag = data['mag']
        periods = results['periods']
        best_ls_p = results['best_ls_period']
        best_rms_p = results['best_rms_period']

        # col 0: phased at ls period
        _plot_phased(axes[row_idx, 0], t, mag, best_ls_p, true_period,
                     label_x=(row_idx == n_rows - 1))
        axes[row_idx, 0].set_title(f"LS Phased\nP={best_ls_p:.2f}h", fontsize=10)
        axes[row_idx, 0].set_ylabel(f"{name}\nMagnitude", fontsize=10, fontweight='bold')

        # col 1: phased at rms period
        _plot_phased(axes[row_idx, 1], t, mag, best_rms_p, true_period,
                     label_y=False, label_x=(row_idx == n_rows - 1))
        axes[row_idx, 1].set_title(f"RMS Phased\nP={best_rms_p:.2f}h", fontsize=10)

        # col 2: ls periodogram
        ax_ls = axes[row_idx, 2]
        ax_ls.plot(periods, results['ls_power'], color='black', linewidth=0.8)
        ax_ls.axvline(true_period, color='#00A087', ls='--', lw=1.5, alpha=0.8,
                      label=f'True: {true_period}h')
        ax_ls.set_xscale('log')
        ax_ls.xaxis.set_major_formatter(ScalarFormatter())
        ax_ls.set_xticks([2, 4, 6, 12, 24])
        ax_ls.set_title("Lomb-Scargle Power", fontsize=10)
        ax_ls.set_ylabel("Power", fontsize=9)
        ax_ls.grid(True, which='both', ls='--', alpha=0.3)
        if row_idx == 0:
            ax_ls.legend(fontsize=8)
        if row_idx == n_rows - 1:
            ax_ls.set_xlabel("Period (h)")

        # col 3: rms scan
        ax_rms = axes[row_idx, 3]
        ax_rms.plot(periods, results['rms_values'], color='#DC0000', linewidth=0.8)
        ax_rms.axvline(true_period, color='#00A087', ls='--', lw=1.5, alpha=0.8)
        ax_rms.plot(best_rms_p, np.min(results['rms_values']), 'ro', markersize=4)
        ax_rms.set_xscale('log')
        ax_rms.xaxis.set_major_formatter(ScalarFormatter())
        ax_rms.set_xticks([2, 4, 6, 12, 24])
        ax_rms.set_title("RMS Metric", fontsize=10)
        ax_rms.set_ylabel("RMS Error", fontsize=9)
        ax_rms.grid(True, which='both', ls='--', alpha=0.3)
        if row_idx == n_rows - 1:
            ax_rms.set_xlabel("Period (h)")

    fig.suptitle(
        f"Evolution of Period Solution (True P={true_period}h, "
        f"{len(datasets)} nights shown)",
        fontsize=15, fontweight='bold'
    )

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")

    return fig
