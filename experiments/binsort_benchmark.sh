#!/bin/bash

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

for dataset in "sportstables" "open_data_usa" "gittables"; do
    for m in 10 50 100 500 1000 5000; do
        compute-histograms -i data/"$dataset"/pq -o data/"$dataset"/histograms_m"$m".zst --bin-range "$m" "$m" --log-file logs/runtime_benchmark/binsort/"$dataset"-construction-m"$m".log
        compute-binsort -i data/"$dataset"/histograms_m"$m".zst -o data/"$dataset"/binsort_m"$m".zst

        for i in {1..5}; do
            run-queries -i data/"$dataset"/histograms_m"$m".zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over --log-file logs/runtime_benchmark/binsort/"$dataset"-iterative-m"$m"-"$i".log
            run-queries -i data/"$dataset"/binsort_m"$m".zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/binsort/"$dataset"-binsort-m"$m"-"$i".log
        done
    done
done

end_time=$(date +%s)
echo Executed script in $((end_time - start_time))s.
