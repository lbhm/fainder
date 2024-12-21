#!/bin/bash

echo "Executing setup"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)
log_level=INFO
nproc=$(nproc)

# Create parquet files (if necessary) and histograms
convert-to-parquet -i data/sportstables/csv -o data/sportstables/pq
compute-histograms -i data/sportstables/pq -o data/sportstables/histograms.zst --bin-range 10 20
compute-binsort -i data/sportstables/histograms.zst -o data/sportstables/binsort.zst
compute-distributions -i data/sportstables/pq -o data/sportstables/normal_dists.zst -k normal
filter-histograms -i data/sportstables/histograms.zst -o data/sportstables/filters/01.zst -s 0.01

convert-to-parquet -i data/open_data_usa/csv -o data/open_data_usa/pq
compute-histograms -i data/open_data_usa/pq -o data/open_data_usa/histograms.zst --bin-range 10 20
compute-binsort -i data/open_data_usa/histograms.zst -o data/open_data_usa/binsort.zst
compute-distributions -i data/open_data_usa/pq -o data/open_data_usa/normal_dists.zst -k normal
filter-histograms -i data/open_data_usa/histograms.zst -o data/open_data_usa/filters/01.zst -s 0.01

compute-histograms -i data/gittables/pq -o data/gittables/histograms.zst --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf025.zst -f 0.25 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf050.zst -f 0.5 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf100.zst -f 1 --bin-range 10 20
compute-histograms -i data/gittables/pq -o data/gittables/histograms_sf200.zst -f 2 --bin-range 10 20
compute-binsort -i data/gittables/histograms.zst -o data/gittables/binsort.zst
compute-distributions -i data/gittables/pq -o data/gittables/normal_dists.zst -k normal
filter-histograms -i data/gittables/histograms.zst -o data/gittables/filters/01.zst -s 0.01

### Benchmark queries ###
cluster-histograms -i data/sportstables/histograms.zst \
    -o data/sportstables/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 50000 \
    -t quantile \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"
generate-queries -o data/sportstables/queries/accuracy_benchmark/all.zst \
    --n-percentiles 20 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-100" "1000"
python experiments/collate_benchmark_queries.py \
    -d sportstables \
    -q data/sportstables/queries/accuracy_benchmark/all.zst \
    -c data/sportstables/clusterings/benchmark_queries.zst \
    -w "$nproc" \
    --log-level "$log_level"

cluster-histograms -i data/open_data_usa/histograms.zst \
    -o data/open_data_usa/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 50000 \
    -t quantile \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"
generate-queries -o data/open_data_usa/queries/accuracy_benchmark/all.zst \
    --n-percentiles 20 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-1000" "1000"
python experiments/collate_benchmark_queries.py \
    -d open_data_usa \
    -q data/open_data_usa/queries/accuracy_benchmark/all.zst \
    -c data/open_data_usa/clusterings/benchmark_queries.zst \
    -w "$nproc" \
    --log-level "$log_level"

cluster-histograms -i data/gittables/histograms.zst \
    -o data/gittables/clusterings/benchmark_queries.zst \
    -a kmeans \
    -c 140 140 \
    -b 100000 \
    -t quantile \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"
generate-queries -o data/gittables/queries/accuracy_benchmark/all.zst \
    --n-percentiles 10 \
    --n-reference-values 100 \
    --seed 42 \
    --reference-value-range "-10000" "10000"
python experiments/collate_benchmark_queries.py \
    -d gittables \
    -q data/gittables/queries/accuracy_benchmark/all.zst \
    -c data/gittables/clusterings/benchmark_queries.zst \
    -w "$nproc" \
    --log-level "$log_level"

### Index creation ###
# Based on the results of our grid search (see the accuracy benchmark), we selected the best index
# configuration for each dataset collection and run the accuracy comparisons with them. For
# convenience, we create the best index configuration for each dataset collection in this setup
# so that you do not have to execute the entire grid search. For more details on the grid search,
# please see the discussion in our paper.

# Sportstables
cluster-histograms \
    -i data/sportstables/histograms.zst \
    -o data/sportstables/clusterings/best_config.zst \
    -a kmeans \
    -c 230 230 \
    -b 5000 \
    -t standard \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"

# Open Data
cluster-histograms \
    -i data/open_data_usa/histograms.zst \
    -o data/open_data_usa/clusterings/best_config.zst \
    -a kmeans \
    -c 250 250 \
    -b 50000 \
    -t quantile \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"

# GitTables
cluster-histograms \
    -i data/gittables/histograms.zst \
    -o data/gittables/clusterings/best_config.zst \
    -a kmeans \
    -c 750 750 \
    -b 100000 \
    -t quantile \
    --alpha 1 \
    --seed 42 \
    --log-level "$log_level"

for dataset in "sportstables" "open_data_usa" "gittables"; do
    create-index \
        -i data/"$dataset"/clusterings/best_config.zst \
        -m rebinning \
        -p float32 \
        -o data/"$dataset"/indices \
        --index-file best_config_rebinning.zst \
        --log-level "$log_level"
    create-index \
        -i data/"$dataset"/clusterings/best_config.zst \
        -m conversion \
        -p float32 \
        -o data/"$dataset"/indices \
        --index-file best_config_conversion.zst \
        --log-level "$log_level"
done

end_time=$(date +%s)
echo Executed setup in $((end_time - start_time))s.
