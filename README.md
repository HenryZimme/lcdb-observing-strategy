# lcdb-observing-strategy

Statistical analysis of asteroid rotation periods from the [LCDB](https://minplanobs.org/mpinfo/php/lcdb.php) (Asteroid Lightcurve Database) to design an optimized observing campaign for **Cindygraber (MPC 7605)** at Phillips Academy Observatory.

This is the supporting code for the PHY530 project proposal (Zimmerman, 2026).

---

## What this does

1. Loads and cleans `lc_summary.csv` from the LCDB
2. Filters for Cindygraber-like asteroids (35–100 km diameter, albedo ≤ 0.075)
3. Models how many nights are needed to cover 85% of an asteroid's rotational phase
4. Runs Monte Carlo weather simulations (k=1000) to account for clear-night uncertainty
5. Computes CDF and ROI metrics (efficiency, marginal gain) to justify the 52-night request
6. Simulates synthetic lightcurves and benchmarks Lomb-Scargle vs Fourier RMS period recovery
7. Tracks LS estimate convergence and derives a delta-P stopping criterion
8. Produces a bootstrapped (n=1000) final CDF with 95% CI

---

## Repo structure

```
lcdb-observing-strategy/
├── notebooks/
│   └── lcdb_observing_strategy.ipynb   # full pipeline, one cell per analysis stage
├── src/
│   ├── data_loader.py          # drive mount, csv load, header detection, cleaning
│   ├── filters.py              # full / diameter / cindygraber-like filter sets
│   ├── plotting.py             # period histograms, scatter plots, pub style
│   ├── observing_strategy.py   # phase-coverage model, mc weather, cdf/roi
│   ├── lightcurve_sim.py       # synthetic lc generation, ls + rms period recovery
│   ├── convergence.py          # ls convergence sim, delta-p stopping criterion
│   └── bootstrap_analysis.py   # bootstrap cdf with 95% ci
├── outputs/                    # saved figures and csv exports (gitignored)
├── docs/
│   └── DOCS.md                 # full technical documentation
├── generate_notebook.py        # regenerates the .ipynb from source
├── .gitignore
└── README.md
```

---

## Quick start (Google Colab)

1. Upload `lc_summary.csv` to your Google Drive
2. Clone this repo or upload the `src/` folder to Colab
3. Open `notebooks/lcdb_observing_strategy.ipynb`
4. Run all cells top to bottom

The notebook installs its own dependencies (`astropy`, `scikit-learn`, `seaborn`) on first run.

## Quick start (local)

```bash
pip install astropy scikit-learn seaborn pandas numpy matplotlib
```

```python
import sys
sys.path.insert(0, 'src')

from data_loader import load_lcdb
from filters import apply_filters

df = load_lcdb(local_path='./lc_summary.csv')
full_df, diam_df, ap_df = apply_filters(df)
```

---

## Data

The LCDB `lc_summary.csv` is not included in this repo (it's ~15 MB and updated regularly).  
Download it from: https://minplanobs.org/mpinfo/php/lcdb.php

---

## Key results

| Metric | Value |
|---|---|
| Cindygraber-like population (ap_df) | 698 asteroids |
| ap_df period median | 12.99 h |
| ap_df IQR | 10.92 h |
| Nights for peak efficiency (clear skies) | Night 6 |
| Nights for 80% CDF (clear skies) | Night 9 |
| Nights for 80% CDF (20% clear, mean) | Night 37 |
| Nights for 80% CDF (20% clear, −1σ) | Night 50 |
| **Requested observing nights** | **52** |

---

## Citation

> Zimmerman, H. (2026). *LCDB Observing Strategy: Cindygraber (MPC 7605) Rotational Period Determination* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

ASCL record: https://ascl.net/XXXX.XXX (pending)

LCDB data:
> Warner, B.D., Harris, A.W., & Pravec, P. (2009). The Asteroid Lightcurve Database. *Icarus*, 202(1), 134–146.
