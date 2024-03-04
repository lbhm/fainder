#!/bin/bash

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

cp data/sportstables/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k230-b5000-standard-a1.zst data/sportstables/indices/exact_results.zst
cp data/open_data_usa/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k250-b50000-quantile-a1.zst data/open_data_usa/indices/exact_results.zst
cp data/gittables/indices/accuracy_benchmark/k_cluster/conversion-kmeans-k750-b100000-quantile-a1.zst data/gittables/indices/exact_results.zst

for dataset in "sportstables" "open_data_usa" "gittables"; do
    for i in {1..5}; do
        python experiments/compute_exact_results.py \
            -H data/"$dataset"/histograms.zst \
            -i data/"$dataset"/indices/exact_results.zst \
            -q data/"$dataset"/queries/test-all.zst \
            --no-sym-difference \
            --log-file logs/exact_results/"$dataset"-"$i".zst
    done
done

end_time=$(date +%s)
echo Executed script in $((end_time - start_time))s.
