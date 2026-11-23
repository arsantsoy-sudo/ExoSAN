# ExoSAN
### Exoplanet Signal Analysis Network

**Author:** Arsan Tsoi  
**Scientific Supervisor:** Дархан Шадыкул Кылышбекулы

---

## About the Project

Thousands of potential exoplanets have been identified through changes in the brightness of distant stars. However, not every decrease in brightness is caused by a planet. Eclipsing binary systems, stellar variability, and instrumental effects can produce signals that resemble planetary transits.

**ExoSAN** is a research project that explores how machine learning can distinguish these signals using observations from NASA's Kepler Space Telescope.

The goal is not simply to detect possible transits, but to better understand their origin and reduce the number of false-positive classifications.

## Research Hypothesis

Our hypothesis is that **combining transit characteristics with information about stellar variability can improve the classification of astrophysical signals** compared with analyzing transit features alone.

To investigate this, ExoSAN combines two complementary approaches:

- **Box Least Squares (BLS)** identifies periodic transit-like signals and estimates their periods, depths, and durations.
- **Fourier analysis (FFT)** extracts information about periodic brightness variations that may help distinguish planetary transits from stellar activity and other astrophysical signals.

We investigate whether combining these features improves the performance of machine learning models, particularly a one-dimensional Convolutional Neural Network (1D CNN).

## Dataset

The project uses Kepler light curves and publicly available astronomical catalogs.

The experimental dataset contains five labeled categories:

1. `planet` — Planet candidates
2. `false_positive` — False-positive signals
3. `eclipsing_binary` — Eclipsing binary stars
4. `stellar_variability` — Stellar variability
5. `quiet` — Quiet stars

The dataset is organized into 6,000 objects, with 4,200 assigned to training, 900 to validation, and 900 to testing.

Catalogs and dataset splits are available in [`catalogs/`](catalogs/).

## How ExoSAN Works

The processing pipeline consists of five main stages:

**1. Light Curve Preprocessing**

Kepler FITS files are read, quality-flagged measurements are filtered, and flux values are normalized and detrended.

**2. Transit Search (BLS)**

The Box Least Squares algorithm searches for periodic decreases in stellar brightness and extracts candidate transit parameters.

**3. Fourier Analysis**

The Fast Fourier Transform is used to extract frequency-domain features from the light curves.

**4. Feature Representation**

Global and local phase-folded representations are generated to capture both the overall periodic signal and the detailed shape of potential transits.

**5. Machine Learning**

Two approaches are investigated:

- **Random Forest:** classification based on numerical features extracted from BLS and Fourier analysis.
- **1D CNN:** classification based on global and local phase-folded light curves, with an additional experimental branch for Fourier features.

The models are compared using balanced accuracy, macro-F1, and confusion matrices.

## Repository Guide

| Location | Contents |
|---|---|
| [`catalogs/`](catalogs/) | Astronomical source catalogs and dataset splits |
| [`scripts/`](scripts/) | Data processing, feature extraction, and model training |
| [`reports/`](reports/) | Experimental metrics, predictions, and confusion matrices |
| `results/` | Selected scientific visualizations |
| `requirements.txt` | Python dependencies |

The main scripts follow the processing sequence:

`01_preprocessing.py` → `02_bls.py` → `02_views.py` / `03_fourier.py` → `07_full6000.py` → `08_full_rf.py` / `09_full_cnn.py`

## Getting Started

The project was developed using Python 3.11.

Clone the repository and install its dependencies:

```bash
git clone https://github.com/arsantsoy-sudo/ExoSAN.git
cd ExoSAN
python -m pip install -r requirements.txt
```

**Note:** The complete pipeline requires Kepler FITS files that are not included in this repository. Some scripts also depend on local configuration paths that must be adapted before running the experiments independently.

## Results and Ongoing Work

The current repository contains experimental results for Random Forest, CNN-BLS, and CNN-Fusion models.

Evaluation reports are available in [`reports/`](reports/). The main research question is whether combining BLS and Fourier features provides a measurable improvement in classification performance.

Future work will focus on validating these results, analyzing misclassified objects, and improving the reproducibility of the pipeline.

## Data Sources

This research uses publicly available data from:

- NASA Kepler Mission / MAST
- NASA Exoplanet Archive
- Kepler Eclipsing Binary Catalog
- McQuillan et al. stellar rotation catalogs
