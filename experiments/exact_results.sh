#!/bin/bash

echo "Executing exact results experiment"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

cp data/sportstables/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k230-b5000-standard-a1.zst data/sportstables/indices/exact_results.zst
cp data/open_data_usa/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k250-b50000-quantile-a1.zst data/open_data_usa/indices/exact_results.zst
cp data/gittables/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k750-b100000-quantile-a1.zst data/gittables/indices/exact_results.zst

for dataset in "sportstables" "open_data_usa" "gittables"; do
    cp data/"$dataset"/queries/accuracy_benchmark/test-all.zst data/"$dataset"/queries/exact_results.zst

    for i in {1..5}; do
        python experiments/compute_exact_results.py \
            -d data/"$dataset"/histograms.zst \
            -i data/"$dataset"/indices/exact_results.zst \
            -q data/"$dataset"/queries/exact_results.zst \
            --no-sym-difference \
            --log-file logs/exact_results/"$dataset"-pscan-"$i".zst
        python experiments/compute_exact_results.py \
            -d data/"$dataset"/binsort.zst \
            -i data/"$dataset"/indices/exact_results.zst \
            -q data/"$dataset"/queries/exact_results.zst \
            -e binsort \
            --no-ground-truth \
            --log-file logs/exact_results/"$dataset"-binsort-"$i".zst
    done
done

end_time=$(date +%s)
echo Executed exact results experiment in $((end_time - start_time))s.
