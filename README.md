<!-- markdownlint-disable MD033 -->
<p align="center">
  <picture>
    <img alt="Fainder logo" src="https://github.com/user-attachments/assets/41686649-f1c1-4b60-824e-80c322c5da85" width="300">
  </picture>
</p>

# Fainder

![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Flbhm%2Ffainder%2Fmain%2Fpyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub License](https://img.shields.io/github/license/lbhm/fainder)

This repository contains the source code, experiment logs, and result analyses for our VLDB 2024
paper **"Fainder: A Fast and Accurate Index for Distribution-Aware Dataset Search"**.

**This branch contains changes to the original codebase in preparation for a demo of Fainder.**
See the `main` branch for the original codebase.

The repository is structured as follows:

```bash
fainder/
├── analysis  # Jupyter notebooks with result analyses and plotting code
├── data  # dataset collections and intermediate data structures from experiments
├── experiments  # Python and Bash scripts with experiment configurations
├── fainder  # main Python package with our index implementation
├── logs  # results of our experimental evaluation
└── tex  # LaTeX source code for the paper
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

### Reproducing Experiments

To reproduce our experiments, you can perform a regular installation or use our Docker container
for convenience. The following commands clone this repo, build the Docker container, and then
execute a script that reruns the experiments for all figures in the paper to then recompile the
paper. The reproduced paper is located at `tex/main.pdf`.

```bash
git clone https://github.com/lbhm/fainder
cd fainder
docker build -t fainder:latest .
docker run -it --rm --name fainder -u "$(id -u)":"$(id -g)" --mount type=bind,src=.,dst=/fainder fainder
```

Please note:

- You still need to download the dataset collections first and place them in the abovementioned folders.
- Reproducing all experiments takes a significant amount of time. If you wish to only reproduce some experiments, you can comment out lines in `experiments/run_all.sh`.
- If you do not rerun all experiments, the existing data in `logs/` will ensure that all figures are created properly. Every experiment you rerun will overwrite parts of the existing logs. If you want to make sure that no existing logs are used for creating figures, delete the contents of `logs/` before starting experiments.
- You can append `bash` to the `docker run` command to start an interactive shell instead of executing the pre-configured experiments.
- You can interactively analyze experiment results with the notebooks in `analysis/` or rely on the plotting script in `experiments/` that reproduces the figures from the paper.

The scripts in `experiments/` contain more experiments than we could cover in the paper. Please see
the commented out lines and additional files if you are interested in them. The individual scripts
do not exactly follow the section structure of our paper but are roughly structured as follows with
the approximate execution time in parenthesis:

```bash
experiments/
├── setup.sh  # Create randomized histograms of the raw data and generate benchmark queries (~48 hours)
├── benchmark_runtime.sh  # Runtime analysis of Fainder and baselines (~97 hours)
├── benchmark_scalability.sh  # Runtime scalability analysis on GitTables (~11 hours)
├── benchmark_construction.sh  # Index construction time analysis (~95 hours)
├── benchmark_exact.sh  # Runtime breakdown of Fainder Exact (~94 hours)
├── benchmark_accuracy.sh  # Parameter grid search and accuracy comparison to baselines (~42 hours)
├── benchmark_parameters.sh  # Detailed analysis of index parameters (~2 hours)
└── run_all.sh  # Run all of the experiments above (~389 hours/~16 days)
```

The additional Python files in `experiments/` encapsulate partial experiment logic that we use in
the scripts mentioned above.

### General Usage

To run your own experiments, review the CLI documentation of the `fainder` executables (see
`pyproject.toml`) and take a look at our scripts in `experiments/`.

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
