#!/bin/bash
# shellcheck disable=SC2043

echo "Executing accuracy benchmark"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)
log_level=INFO

### Grid search preprocessing ###
# Sportstables
# dataset="sportstables"
# i=1
# for algo in "kmeans" "agglomerative"; do
#     for budget in 5000 10000 50000; do
#         for alpha in 0 1 5; do
#             for transform in "none" "standard" "robust" "quantile"; do
#                 printf -v j "%03d" $i
#                 cluster-histograms -i data/"$dataset"/histograms.zst \
#                     -o "data/$dataset/clusterings/accuracy_benchmark/grid_search/$j-$algo-k%-b$budget-$transform-a$alpha.zst" \
#                     -a "$algo" \
#                     -c 2 150 \
#                     -b "$budget" \
#                     -t "$transform" \
#                     --alpha "$alpha" \
#                     --seed 42
#                 ((i++))
#             done
#         done
#     done

#     for k in {10..250..10} {300..1000..50}; do
#         for transform in "standard" "quantile"; do
#             cluster-histograms -i data/"$dataset"/histograms.zst \
#                 -o "data/$dataset/clusterings/accuracy_benchmark/k_cluster/$algo-k$k-b5000-$transform-a1.zst" \
#                 -a "$algo" \
#                 -c "$k" "$k" \
#                 -b 5000 \
#                 -t "$transform" \
#                 --alpha 1 \
#                 --seed 42
#         done
#     done
# done

# Open Data
# dataset="open_data_usa"
# i=1
# for algo in "kmeans" "agglomerative"; do
#     for budget in 50000 100000; do
#         for alpha in 0 1 5; do
#             for transform in "none" "standard" "robust" "quantile"; do
#                 printf -v j "%03d" $i
#                 cluster-histograms -i data/"$dataset"/histograms.zst \
#                     -o "data/$dataset/clusterings/accuracy_benchmark/grid_search/$j-$algo-k%-b$budget-$transform-a$alpha.zst" \
#                     -a "$algo" \
#                     -c 2 150 \
#                     -b "$budget" \
#                     -t "$transform" \
#                     --alpha "$alpha" \
#                     --seed 42
#                 ((i++))
#             done
#         done
#     done

#     for k in {10..250..10} {300..1000..50}; do
#         for transform in "standard" "quantile"; do
#             cluster-histograms -i data/"$dataset"/histograms.zst \
#                 -o "data/$dataset/clusterings/accuracy_benchmark/k_cluster/$algo-k$k-b50000-$transform-a1.zst" \
#                 -a "$algo" \
#                 -c "$k" "$k" \
#                 -b 50000 \
#                 -t "$transform" \
#                 --alpha 1 \
#                 --seed 42
#         done
#     done
# done

# GitTables
# dataset="gittables"
# i=1
# for algo in "kmeans"; do # Agglomerative clustering is too slow for GitTables
#     for budget in 50000 100000; do
#         for alpha in 0 1 5; do
#             for transform in "quantile"; do
#                 # NOTE: All transforms except for quantile produce unbalanced clusterings and thus lead to crashes during query execution for GitTables
#                 printf -v j "%03d" $i
#                 cluster-histograms -i data/"$dataset"/histograms.zst \
#                     -o "data/$dataset/clusterings/accuracy_benchmark/grid_search/$j-$algo-k%-b$budget-$transform-a$alpha.zst" \
#                     -a "$algo" \
#                     -c 100 150 \
#                     -b "$budget" \
#                     -t "$transform" \
#                     --alpha "$alpha" \
#                     --seed 42
#                 ((i++))
#             done
#         done
#     done

#     for k in {100..250..10} {300..1000..50}; do
#         # Mid and high selectivity crash for k < 100 because the result set together with the index becomes too large
#         cluster-histograms -i data/"$dataset"/histograms.zst \
#             -o "data/$dataset/clusterings/accuracy_benchmark/k_cluster/$algo-k$k-b100000-quantile-a1.zst" \
#             -a "$algo" \
#             -c "$k" "$k" \
#             -b 100000 \
#             -t quantile \
#             --alpha 1 \
#             --seed 42
#     done
# done

### Grid search execution ###
# for dataset in "sportstables" "open_data_usa" "gittables"; do
#     for experiment in "grid_search" "k_cluster"; do
#         for path in data/"$dataset"/clusterings/accuracy_benchmark/"$experiment"/*; do
#             file=${path##*/}
#             create-index -i "$path" \
#                 -m rebinning \
#                 -p float32 \
#                 -o data/"$dataset"/indices/accuracy_benchmark/"$experiment" \
#                 --index-file "rebinning-${file%%.*}.zst"
#             create-index -i "$path" \
#                 -m conversion \
#                 -p float32 \
#                 -o data/"$dataset"/indices/accuracy_benchmark/"$experiment" \
#                 --index-file "conversion-${file%%.*}.zst"
#         done
#         for queries in "low_selectivity" "mid_selectivity" "high_selectivity"; do
#             for path in data/"$dataset"/indices/accuracy_benchmark/"$experiment"/*; do
#                 file=${path##*/}
#                 python experiments/compute_fainder_results.py \
#                     -i "$path" \
#                     -q data/"$dataset"/queries/accuracy_benchmark/val-"$queries".zst \
#                     -t data/"$dataset"/results/accuracy_benchmark/ground_truth-val-"$queries".zst \
#                     --log-file logs/accuracy_benchmark/"$experiment"/"$dataset"-"${file%%.*}"-"$queries".zst
#             done
#         done
#     done
# done

### Baseline comparison ###
for dataset in "sportstables" "open_data_usa" "gittables"; do
    compute-distributions -i data/"$dataset"/pq -o data/"$dataset"/normal_dists.zst -k normal

    for queries in "low_selectivity" "mid_selectivity" "high_selectivity"; do
        python experiments/compute_pscan_results.py \
            -H data/"$dataset"/histograms.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-pscan-"$queries".zst \
            --log-level "$log_level"
        python experiments/compute_binsort_results.py \
            -i data/"$dataset"/binsort.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            -t data/"$dataset"/results/accuracy_benchmark/ground_truth-test-"$queries".zst \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-binsort-"$queries".zst \
            --log-level "$log_level"
        python experiments/compute_ndist_results.py \
            -d data/"$dataset"/normal_dists.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            -t data/"$dataset"/results/accuracy_benchmark/ground_truth-test-"$queries".zst \
            -w "$(nproc)" \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-ndist-"$queries".zst \
            --log-level "$log_level"
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/best_config_rebinning.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            -t data/"$dataset"/results/accuracy_benchmark/ground_truth-test-"$queries".zst \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-rebinning-"$queries".zst \
            --log-runtime \
            --log-level "$log_level"
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/best_config_conversion.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            -t data/"$dataset"/results/accuracy_benchmark/ground_truth-test-"$queries".zst \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-conversion-"$queries".zst \
            --log-runtime \
            --log-level "$log_level"
        python experiments/compute_exact_results.py \
            -d data/"$dataset"/binsort.zst \
            -i data/"$dataset"/indices/best_config_conversion.zst \
            -q data/"$dataset"/queries/accuracy_benchmark/test-"$queries".zst \
            -e binsort \
            --no-ground-truth \
            --log-file logs/accuracy_benchmark/baseline_comp/"$dataset"-exact-"$queries".zst \
            --log-level "$log_level"
    done
done

### LLM Query Workload ###
# for dataset in "${!datasets[@]}"; do
#     cp data/llm_workload/llm_queries.zst data/"$dataset"/queries/llm/all.zst
#     python experiments/collate_benchmark_queries.py \
#         -d "$dataset" \
#         -q data/"$dataset"/queries/llm/all.zst \
#         -W llm \
#         -c "data/$dataset/clusterings/accuracy_benchmark/k_cluster/${datasets[$dataset]}.zst"
#
#     python experiments/compute_ndist_results.py \
#         -d data/"$dataset"/normal_dists.zst \
#         -q data/"$dataset"/queries/llm/all.zst \
#         -t data/"$dataset"/results/llm/ground_truth-all.zst \
#         -w "$(nproc)" \
#         --log-file logs/accuracy_benchmark/llm/"$dataset"-ndist.zst
#     python experiments/compute_fainder_results.py \
#         -i data/"$dataset"/indices/accuracy_benchmark/k_cluster/rebinning-"${datasets[$dataset]}".zst \
#         -q data/"$dataset"/queries/llm/all.zst \
#         -t data/"$dataset"/results/llm/ground_truth-all.zst \
#         --log-file logs/accuracy_benchmark/llm/"$dataset"-rebinning.zst \
#         --log-runtime
#     python experiments/compute_fainder_results.py \
#         -i data/"$dataset"/indices/accuracy_benchmark/k_cluster/conversion-"${datasets[$dataset]}".zst \
#         -q data/"$dataset"/queries/llm/all.zst \
#         -t data/"$dataset"/results/llm/ground_truth-all.zst \
#         --log-file logs/accuracy_benchmark/llm/"$dataset"-conversion.zst \
#         --log-runtime
# done

end_time=$(date +%s)
echo Executed accuracy benchmark in $((end_time - start_time))s.
