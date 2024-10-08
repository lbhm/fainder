<p align="center">
  <picture>
    <img alt="Fainder logo" src="https://github.com/user-attachments/assets/41686649-f1c1-4b60-824e-80c322c5da85" width="400">
  </picture>
</p>

# Fainder

![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Flbhm%2Ffainder%2Fmain%2Fpyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub License](https://img.shields.io/github/license/lbhm/fainder)

This repository contains the source code, experiment logs, and result analyses for our VLDB 2024
paper **"Fainder: A Fast and Accurate Index for Distribution-Aware Dataset Search"**.

The repository is structured as follows:

```bash
fainder/
├── analysis  # Jupyter notebooks with result analyses and plotting code
├── data  # dataset collections and intermediate data structures from experiments
├── experiments  # Python and Bash scripts with experiment configurations
├── fainder  # main Python package with our index implementation
└── logs  # results of our experimental evaluation
```

## Setup

### Requirements

- Ubuntu >= 22.04
  - `fainder` is tested on amd64-based Ubuntu systems but other Linux systems might work as well
- Python 3.10 - 3.12
  - We use `pip` and `virtualenv` in this guide but this is not a hard requirement

**Note:** The configuration in `pyproject.toml` defines flexible dependency specifiers to ensure
maximum compatibility. If you want to reproduce the exact software dependencies we used for our
experiments, refer to `pip.lock`.

### Installation

#### User Setup

```bash
git clone https://github.com/lbhm/fainder
cd fainder
virtualenv venv
source venv/bin/activate
pip install .
```

If you also want to execute the analysis notebooks and generate the plots we show in our paper,
replace the last line with `pip install -e ".[analysis]"`. Note that to recreate the plots
as they appear in the paper, you also need a working LaTeX installation on your computer (see the
[Matplotlib docs](https://matplotlib.org/stable/users/explain/text/usetex.html) for details). If
you just want to recreate the results and do not care about the layout, you can remove the call to
`set_style()` in each notebook.

#### Development Setup

```bash
# Follow the steps above until you have activated your virtual environment
pip install -e ".[dev]"
pre-commit install
```

## Reproducibility

### Datasets

Our experiment configurations assume the existence of the following folders that contain the
dataset collections we use (formatted either as CSV or Parquet files):

- `data/sportstables/csv`: Follow the instructions at [DHBWMosbachWI/SportsTables](https://github.com/DHBWMosbachWI/SportsTables) or contact the authors of the original paper to acquire a dump of the dataset collection.
- `data/open_data_usa/csv`: Follow the instructions at [Open Data Portal Watch](https://data.wu.ac.at/portalwatch/about) or contact us to receive a download link for this collection.
- `data/gittables/pq`: Follow the instructions at [gittables.github.io](https://gittables.github.io/) or use our download script (see `download-datasets -h`).

### General Usage

To run your own experiments, review the CLI documentation of the `fainder` executables (see
`pyproject.toml`) and take a look at our scripts in `experiments/`.

### Reproducing Experiments

Our experiments can be reproduced by running the respective scripts in `experiments/` and
subsequently analyzing them with the notebooks in `analysis/`. The experiment scripts do not
exactly follow the section structure of our paper but are roughly structured as follows:

```bash
experiments/
├── setup.sh  # Create randomized histograms of the raw data and generate benchmark queries
├── runtime_benchmark.sh  # Runtime comparison to baselines
├── scalability_benchmark.sh  # Runtime scalability analysis on GitTables
├── accuracy_benchmark.sh  # Parameter grid search and comparison to baselines
├── exact_results.sh  # Runtime breakdown of Fainder Exact
├── microbenchmarks.sh  # Detailed analysis of index parameters
└── binsort_benchmark.sh  # Supplementary experiments for the binsort baseline
```

The additional Python files in `experiments/` encapsulate partial experiment logic that we use in
the scripts mentioned above.

## Citation

```bibtex
@article{behme_fainder_2024,
    title        = {Fainder: A Fast and Accurate Index for Distribution-Aware Dataset Search},
    author       = {Behme, Lennart and Galhotra, Sainyam and Beedkar, Kaustubh and Markl, Volker},
    year         = 2024,
    journal      = {Proc. VLDB Endow.},
    publisher    = {VLDB Endowment},
    volume       = 17,
    number       = 11,
    pages        = {3269--3282},
    doi          = {10.14778/3681954.3681999},
    issn         = {2150-8097},
    url          = {https://doi.org/10.14778/3681954.3681999},
    issue_date   = {August 2024}
}
```
