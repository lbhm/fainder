#!/bin/bash

echo "Executing exact results experiment"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

for dataset in "sportstables" "open_data_usa" "gittables"; do
    cp data/"$dataset"/queries/accuracy_benchmark/test-all.zst data/"$dataset"/queries/exact_results.zst

    for i in {1..5}; do
        python experiments/compute_exact_results.py \
            -d data/"$dataset"/histograms.zst \
            -i data/"$dataset"/indices/best_config_conversion.zst \
            -q data/"$dataset"/queries/exact_results.zst \
            -e pscan \
            --no-sym-difference \
            --log-file logs/exact_results/"$dataset"-pscan-"$i".zst
        python experiments/compute_exact_results.py \
            -d data/"$dataset"/binsort.zst \
            -i data/"$dataset"/indices/best_config_conversion.zst \
            -q data/"$dataset"/queries/exact_results.zst \
            -e binsort \
            --no-ground-truth \
            --log-file logs/exact_results/"$dataset"-binsort-"$i".zst
    done
done

end_time=$(date +%s)
echo Executed exact results experiment in $((end_time - start_time))s.
