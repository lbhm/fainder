#!/bin/bash

echo "Executing setup"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

# Create parquet files (if necessary) and histograms
convert-to-parquet -i data/sportstables/csv -o data/sportstables/pq
compute-histograms -i data/sportstables/pq -o data/sportstables/histograms.zst --bin-range 10 20
compute-binsort -i data/sportstables/histograms.zst -o data/sportstables/binsort.zst
filter-histograms -i data/sportstables/histograms.zst -o data/sportstables/filters/01.zst -s 0.01

convert-to-parquet -i data/open_data_usa/csv -o data/open_data_usa/pq
compute-histograms -i data/open_data_usa/pq -o data/open_data_usa/histograms.zst --bin-range 10 20
compute-binsort -i data/open_data_usa/histograms.zst -o data/open_data_usa/binsort.zst
filter-histograms -i data/open_data_usa/histograms.zst -o data/open_data_usa/filters/01.zst -s 0.01

compute-histograms -i data/gittables/pq -o data/gittables/histograms.zst --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf025.zst -f 0.25 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf050.zst -f 0.5 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf100.zst -f 1 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf200.zst -f 2 --bin-range 10 20
compute-binsort -i data/gittables/histograms.zst -o data/gittables/binsort.zst
filter-histograms -i data/gittables/histograms.zst -o data/gittables/filters/01.zst -s 0.01

# Benchmark queries
cluster-histograms -i data/sportstables/histograms.zst \
    -o data/sportstables/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 50000 \
    -t quantile \
    --alpha 1 \
    --seed 42
generate-queries -o data/sportstables/queries/accuracy_benchmark/all.zst \
    --n-percentiles 20 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-100" "1000"
python experiments/collate_benchmark_queries.py \
    -d sportstables \
    -q data/sportstables/queries/accuracy_benchmark/all.zst \
    -c data/sportstables/clusterings/benchmark_queries.zst \
    -m logs/query_metrics/sportstables-metrics.zst

cluster-histograms -i data/open_data_usa/histograms.zst \
    -o data/open_data_usa/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 50000 \
    -t quantile \
    --alpha 1 \
    --seed 42
generate-queries -o data/open_data_usa/accuracy_benchmark/all.zst \
    --n-percentiles 20 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-1000" "1000"
python experiments/collate_benchmark_queries.py \
    -d open_data_usa \
    -q data/open_data_usa/accuracy_benchmark/all.zst \
    -c data/open_data_usa/clusterings/benchmark_queries.zst \
    -m logs/query_metrics/open_data_usa-metrics.zst

cluster-histograms -i data/gittables/histograms.zst \
    -o data/gittables/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 100000 \
    -t quantile \
    --alpha 1 \
    --seed 42
generate-queries -o data/gittables/accuracy_benchmark/all.zst \
    --n-percentiles 10 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-10000" "10000"
python experiments/collate_benchmark_queries.py \
    -d gittables \
    -q data/gittables/accuracy_benchmark/all.zst \
    -c data/gittables/clusterings/benchmark_queries.zst \
    -m logs/query_metrics/gittables-metrics.zst

end_time=$(date +%s)
echo Executed setup in $((end_time - start_time))s.
