"""
data_loader.py
--------------
loads and cleans the lcdb lc_summary.csv file from google drive (or a local path).
handles header auto-detection, placeholder replacement, and basic type coercion.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

# lcdb uses -9.99 as a null sentinel
LCDB_NULL = -9.99


def mount_drive():
    """mount google drive if running in colab and not already mounted."""
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("mounting google drive...")
            drive.mount('/content/drive')
        else:
            print("google drive already mounted.")
    except ImportError:
        pass  # not in colab, skip


def find_lcdb_file(default_path="/content/drive/MyDrive/lc_summary.csv", search_depth=3):
    """
    returns the path to lc_summary.csv.
    checks default_path first, then walks mydrive up to search_depth levels.

    args:
        default_path: expected location on google drive
        search_depth: max directory depth to search below mydrive root

    returns:
        str path if found, else raises FileNotFoundError
    """
    if os.path.exists(default_path):
        return default_path

    root = "/content/drive/MyDrive"
    print(f"file not at default path. searching up to depth {search_depth}...")

    for dirpath, _, files in os.walk(root):
        if "lc_summary.csv" in files:
            found = os.path.join(dirpath, "lc_summary.csv")
            print(f"found: {found}")
            return found
        depth = dirpath.replace(root, "").count(os.sep)
        if depth >= search_depth:
            break

    raise FileNotFoundError(
        "lc_summary.csv not found. place it in google drive and re-run."
    )


def detect_header_row(file_path, max_scan=60):
    """
    scans the first max_scan lines to find the row containing column headers.
    looks for 'Number', 'Name', and 'Period' in the same line.

    returns:
        int row index (0-based) of the header row
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i > max_scan:
                break
            if 'Number' in line and 'Name' in line and 'Period' in line:
                return i
    # fallback: lcdb historically has a ~15-line preamble
    return 15


def load_lcdb(file_path=None, local_path=None):
    """
    main entry point. loads lc_summary.csv and returns a cleaned dataframe.

    args:
        file_path: explicit path to lc_summary.csv (overrides auto-search)
        local_path: local filesystem path (for use outside colab)

    returns:
        pd.DataFrame with numeric types coerced and lcdb null sentinels replaced with NaN
    """
    # resolve path
    if local_path and os.path.exists(local_path):
        path = local_path
    elif file_path:
        path = file_path
    else:
        mount_drive()
        path = find_lcdb_file()

    header_row = detect_header_row(path)
    print(f"loading '{path}' (header row: {header_row})...")

    df = pd.read_csv(path, header=header_row, on_bad_lines='skip', low_memory=False)
    df.columns = df.columns.str.strip()

    # drop metadata rows (non-numeric 'Number' field)
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    df = df.dropna(subset=['Number'])

    # replace lcdb null sentinels
    df = df.replace([LCDB_NULL, str(LCDB_NULL)], np.nan)

    # coerce key numeric columns
    for col in ['Period', 'Diam', 'Albedo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"loaded {len(df):,} records, {df.shape[1]} columns.")
    return df
