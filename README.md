# Fainder

![Python](https://img.shields.io/badge/python-3.10_--_3.11-informational)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the source code, experiment logs, and result analyses for our
paper **"Fainder: A Fast and Accurate Index for Distribution-Aware Dataset Search"**.

The repository is structured as follows:

```bash
fainder/
├── analysis  # Jupyter notebooks with result analyses and plotting code
├── data  # placeholder for dataset (see below) and
├── experiments  # Python and Bash scripts with experiment configurations
├── fainder  # main Python package with our index implementation
└── logs  # results of our experimental evaluation
```

## Setup

### Requirements

- Ubuntu >= 22.04
  - `fainder` is tested on amd64-based Ubuntu systems but other Linux systems might work as well
- Python 3.10 or 3.11
  - We use `pip` and `virtualenv` in this guide but this is not a hard requirement

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
replace the last line with `pip install -e ".[analysis]"`.

#### Development Setup

```bash
# Follow the steps above until you activated your virtual environment
pip install -e ".[dev]"
pre-commit install
```

## Reproducibility

### Datasets

Our experiment configurations assume the existence of the following folders that contain the
dataset collections we use formatted as either CSV or Parquet files:

- `data/sportstables/csv`: Follow the instructions at [DHBWMosbachWI/SportsTables](https://github.com/DHBWMosbachWI/SportsTables) or contact the authors of the original paper to acquire a dump of the dataset collection.
- `data/open_data_usa/csv`: Follow the instructions at [Open Data Portal Watch](https://data.wu.ac.at/portalwatch) or contact us to receive a download link for this collection.
- `data/gittables/pq`: Follow the instructions at [gittables.github.io](https://gittables.github.io/) or use our download script (see `download-datasets -h`).

### General Usage

To run your own experiments, review the CLI documentation of the `fainder` executables (see
`pyproject.toml`) and take a look at our scripts in `experiments/`.

### Reproducing Experiments

All of our experiments can be reproduced by running the respective scripts in `experiments/` and
subsequently analyzing them with the notebooks in `analysis/`. The experiment scripts do not
exactly follow the section structure of our paper but are roughly structured as follows:

```bash
experiments/
├── setup.sh  # Create randomized histograms of the raw data and generate benchmark queries
├── runtime_benchmark.sh  # Runtime comparison to baselines
├── scalability_benchmark.sh  # Runtime scalability analysis on GitTables
├── accuracy_benchmark.sh  # Parameter grid search and comparison to baselines
├── exact_results.sh  # Runtime breakdown of Fainder Exact
└── microbenchmarks.sh  # Detailed analysis of index parameters
```

The additional Python in `experiments/` files encapsulate partial experiment logic that we use in
the abovementioned scripts.

## Citation

TBD
