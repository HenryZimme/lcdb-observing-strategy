"""
filters.py
----------
defines the three filter levels used in the cindygraber (mpc 7605) observing strategy analysis:

  1. full_df       — entire lcdb
  2. diam_df       — diameter-only filter (35 <= d <= 100 km)
  3. ap_df         — cindygraber-like: diameter + albedo constraint (albedo <= 0.075)

the ap_df is the primary analysis population. albedo is used as a proxy for
dark (c/d-type) asteroid taxonomy.
"""

import numpy as np
import pandas as pd


# cindygraber physical parameters (mpc 7605)
CINDYGRABER_DIAM_KM = 38.46
CINDYGRABER_ALBEDO  = 0.039

# filter bounds
DIAM_MIN_KM   = 35.0
DIAM_MAX_KM   = 100.0
ALBEDO_MAX    = 0.075   # upper bound for dark taxonomy class
SPIN_BARRIER  = 2.2     # hours — tensile-strength spin limit for rubble piles


def apply_filters(df):
    """
    applies the three-level filter hierarchy to the raw lcdb dataframe.

    args:
        df: cleaned dataframe from data_loader.load_lcdb()

    returns:
        tuple of (full_df, diam_df, ap_df) — each is a pd.DataFrame copy
    """
    full_df = df.copy()

    # diameter-only filter
    diam_df = df[
        (df['Diam'] >= DIAM_MIN_KM) &
        (df['Diam'] <= DIAM_MAX_KM)
    ].copy()

    # cindygraber-like: diameter + dark albedo
    ap_df = df[
        (df['Diam'] >= DIAM_MIN_KM) &
        (df['Diam'] <= DIAM_MAX_KM) &
        (df['Albedo'] <= ALBEDO_MAX)
    ].copy()

    print(f"full lcdb:             {len(full_df):>5,} asteroids")
    print(f"diameter filter:       {len(diam_df):>5,} asteroids  "
          f"({DIAM_MIN_KM:.0f}-{DIAM_MAX_KM:.0f} km)")
    print(f"cindygraber-like (ap): {len(ap_df):>5,} asteroids  "
          f"(diam + albedo <= {ALBEDO_MAX})")

    # sanity check: no asteroids in ap_df should violate spin barrier
    below_barrier = ap_df[ap_df['Period'] < SPIN_BARRIER]
    if len(below_barrier) > 0:
        print(f"  warning: {len(below_barrier)} ap_df asteroids below spin barrier ({SPIN_BARRIER}h)")

    return full_df, diam_df, ap_df


def summarize_period_stats(df, label="dataset"):
    """
    prints descriptive period statistics for a dataframe.

    args:
        df: dataframe with a 'Period' column
        label: string label for print output
    """
    periods = df['Period'].dropna()
    q1 = periods.quantile(0.25)
    q3 = periods.quantile(0.75)
    iqr = q3 - q1

    print(f"\n--- period stats: {label} (n={len(periods):,}) ---")
    print(f"  median : {periods.median():.2f} h")
    print(f"  mean   : {periods.mean():.2f} h")
    print(f"  IQR    : {iqr:.2f} h  (Q1={q1:.2f}, Q3={q3:.2f})")
    print(f"  min    : {periods.min():.4f} h")
    print(f"  max    : {periods.max():.2f} h")
