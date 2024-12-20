#!/bin/bash

experiments/setup.sh
experiments/benchmark_runtime.sh
experiments/benchmark_scalability.sh
experiments/benchmark_construction.sh
experiments/benchmark_exact.sh
experiments/benchmark_accuracy.sh
experiments/benchmark_parameters.sh

echo "All benchmarks executed successfully. Plotting and recompiling paper..."

python experiments/plot_results.py
latexmk -pdf -cd -silent -halt-on-error -shell-escape tex/main.tex
