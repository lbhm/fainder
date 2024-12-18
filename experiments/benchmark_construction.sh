#!/bin/bash
# shellcheck disable=SC2043

echo "Executing index construction benchmark"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

for dataset in "gittables"; do
    k=100
    for i in {1..5}; do
        ### k-cluster runtime experiment ###
        # NOTE: GitTables crashes for bad clustering parameters due to excessive memory usage so we only run the experiment for k>=50
        for k in {50..250..10} {300..1000..50}; do
            cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -a kmeans -c "$k" "$k" -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-k"$k"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-k"$k"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-k"$k"-"$i".log
        done

        ### Bin budget runtime experiment ###
        # NOTE: Again, we need to adjust the parameters for GitTables to avoid crashes
        for budget in 1000 5000 10000 50000 100000; do
            cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -a kmeans -c "$k" "$k" -b "$budget" -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-b"$budget"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-b"$budget"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-b"$budget"-"$i".log
        done
    done
done

# for dataset in "sportstables" "open_data_usa"; do
#     k=10
#     for i in {1..5}; do
#         ### k-cluster runtime experiment ###
#         for k in 1 2 5 {10..250..10}; do
#             cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -a kmeans -c "$k" "$k" -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-k"$k"-"$i".log
#             create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-k"$k"-"$i".log
#             create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-k"$k"-"$i".log
#         done

#         ### Bin budget runtime experiment ###
#         for budget in 1000 5000 10000 50000 100000 500000 1000000; do
#             cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -a kmeans -c "$k" "$k" -b "$budget" -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-b"$budget"-"$i".log
#             create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-b"$budget"-"$i".log
#             create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-b"$budget"-"$i".log
#         done
#     done
# done

end_time=$(date +%s)
echo Executed index construction benchmark in $((end_time - start_time))s.
