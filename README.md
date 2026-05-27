# ML-FTLE & Poincaré-Map FTLE Proxies

> **Companion code for the paper:**
> *"Unified Geometry-Guided ML-FTLE for Tracking Transient Chaos from Scalar Data"*
> *(preprint / under review   link to be added on acceptance)*

---

## Overview

This repository provides two complementary, data-driven methods for estimating the **Finite-Time Lyapunov Exponent (FTLE)**   a scalar measure of local chaos  directly from a single observed time-series, without access to the underlying equations of motion.

| Method | Module | Principle |
|--------|--------|-----------|
| **ML-FTLE** | `core/ml_ftle.py` | Learns FTLE from a machine-learning model trained on the time-series and its short-time predictive divergence |
| **Poincaré-FTLE** | `core/poincare_ftle.py` | Extracts a topology-based chaos proxy by tracking the evolving geometry of sliding-window Poincaré return maps, calibrated to the ML-FTLE via OLS/PLSR |

Both methods are benchmarked against **QR-FTLE**, a reference FTLE computed via QR-decomposition of numerical Jacobians using Spearman ρ and Matthews Correlation Coefficient (MCC) evaluated on a **z-normalised common scale** to remove inter-method amplitude bias.

---

## Repository Structure

```
.
├── core/
│   ├── ml_ftle.py            # ML-FTLE computation (training, prediction, smoothing)
│   └── poincare_ftle.py      # Poincaré-map FTLE proxy (Phase 1–3, OLS/PLSR calibration)
│
├── parameters/
│   ├── data_1.py             # Hyperparameters for dataset 1
│   ├── data_2.py             # Hyperparameters for dataset 2
│   └── data_3.py             # Hyperparameters for dataset 3
│
├── Fig_1_ml_ftle_composite_explainer.ipynb   # Figure 1   ML-FTLE method explainer
├── Fig_2_3_ML_FTLE.ipynb                     # Figures 2–3   ML-FTLE results across datasets
├── Fig_4_5_composite_basis_attractors.ipynb  # Figures 4–5   Poincaré attractor topology
├── Fig_6_7.ipynb                             # Figures 6–7   Proxy comparison & evaluation
├── Fig_8_Noise_Robustnes.ipynb               # Figures 8   Noise Robustness of ML-FTLE and Poincare-FTLE
└── QR_FTLE.ipynb                             # QR-FTLE reference computation
```

---

## Methods

### ML-FTLE (`core/ml_ftle.py`)

The ML-FTLE is estimated by training a short-horizon predictor on the observed scalar time-series and measuring the **local predictive divergence**  the sensitivity of the model's output to small perturbations in initial conditions. This divergence is used as a surrogate for the true leading Lyapunov exponent.

Key functions:
- `train_predictor(signal, t, P)`   fits a gradient-boosted regressor
- `compute_ml_ftle(signal, t, P)`   returns `(t_ftle, lam_ml)` as NumPy arrays
- `smooth_ftle(lam, P)`   applies uniform 1-D smoothing

### Poincaré-FTLE (`core/poincare_ftle.py`)

The Poincaré-FTLE proxy tracks how the **2-D return map geometry** of a sliding window evolves over time. Topological closeness between representative attractor snapshots is compressed into a scalar proxy via PLSR, calibrated against the ML-FTLE.

**Three-phase pipeline:**

| Phase | Function | Output |
|-------|----------|--------|
| 1   Grid construction | `run_phase12(dataset_dir, P)` | Poincaré grids for every window |
| 2   Representative selection | (inside `run_phase12`) | Greedy max-min closeness series |
| 3  PLSR proxy | `run_phase3_plsr(...)` | Latent topological component proxy |

**Topology metrics** used to compare attractor snapshots:

| Metric | Abbreviation | Measures |
|--------|-------------|---------|
| Jensen–Shannon Divergence | JSD | Distributional divergence of histograms |
| Structural Similarity Index | SSIM | Spatial pattern similarity |
| Hausdorff Distance | HDF | Worst-case point-set separation |
| Intersection over Union | IOU | Occupied-pixel overlap fraction |

### QR-FTLE (`QR_FTLE.ipynb`)

Provides the **ground-truth reference** FTLE computed via QR-decomposition of finite-difference Jacobians along a numerical trajectory. All Spearman ρ and MCC scores reported in the paper are computed against this reference, on a z-normalised common scale.

---

## Installation

Python 3.9+ is recommended.

```bash
git clone https://github.com/Manivelan-sv/Unified-Geometry-Guided-ML-FTLE-for-Tracking-Transient-Chaos-from-Scalar-Data.git
cd Unified-Geometry-Guided-ML-FTLE-for-Tracking-Transient-Chaos-from-Scalar-Data
pip install -r requirements.txt
```

**Core dependencies:**

```
numpy
scipy
pandas
matplotlib
scikit-learn
scikit-image
```

---

## Data

Place raw time-series CSV files inside `data/`. Each CSV must contain at minimum:

| Column | Description |
|--------|-------------|
| `time` | Monotonically increasing time axis |
| (signal column) | Scalar observable (position, voltage, etc.) |

The file-reader (`read_table_any_sep` in `core/poincare_ftle.py`) auto-detects comma, tab, and whitespace delimiters and handles headerless files.

> **Note:** The datasets used in the paper are not distributed with this repository. Contact the corresponding author for access, or regenerate from provided code or substitute your own CSV files.

---

## Reproducing the Figures

Run notebooks in the following order for a clean reproduction:

```
QR_FTLE.ipynb                           # 1. Compute reference QR-FTLE → data/qr_ftle.csv
Fig_1_ml_ftle_composite_explainer.ipynb # 2. Figure 1
Fig_2_3_ML_FTLE.ipynb                   # 3. Figures 2–3
Fig_4_5_composite_basis_attractors.ipynb# 4. Figures 4–5
Fig_6_7.ipynb                           # 5. Figures 6–7
```

Dataset-specific hyperparameters (window size, step, lag, number of Poincaré representatives, etc.) are set via the files in `parameters/`. Each notebook imports the appropriate parameter module at its top cell.

---

## Metric Conventions

All evaluation metrics in this work compare a proxy FTLE series against QR-FTLE on their **common time-aligned support**:

- **Spearman ρ**   rank correlation on raw aligned values (scale-invariant by construction).
- **MCC**   Matthews Correlation Coefficient computed after **z-normalising** both series to zero mean and unit variance, thresholded at 0. This removes amplitude scale bias arising from different FTLE estimation methods operating at different dynamic ranges.
- **Sign Agreement**   fraction of time points where `sign(proxy) == sign(QR-FTLE)` after z-normalisation.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{,
  title   = {Unified-Geometry-Guided-ML-FTLE-for-Tracking-Transient-Chaos-from-Scalar-Data},
  author  = {S. V. Manivelan, Andrei Velichko, and I. Manimehan},
  journal = {},
  year    = {2026},
  doi     = {}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions about the code or data, please open a GitHub Issue or contact the corresponding author at `[manivelan.saminathan@gmail.com]`.
