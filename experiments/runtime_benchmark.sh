#!/bin/bash

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

for dataset in "sportstables" "open_data_usa" "gittables"; do
    ### Setup ###
    cp data/"$dataset"/queries/accuracy_benchmark/test-all.zst data/"$dataset"/queries/runtime_benchmark.zst
    cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/default.zst -a kmeans -c 2 150 -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering.log
    create-index -i data/"$dataset"/clusterings/runtime_benchmark/default.zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file rebinning.zst --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning.log
    create-index -i data/"$dataset"/clusterings/runtime_benchmark/default.zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file conversion.zst --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion.log

    query=("0.1" "lt" "50")
    nproc=$(nproc)
    for i in {1..5}; do
        ### General runtime experiments ###
        # Single query
        run-query -i data/"$dataset"/histograms.zst -t histograms -q "${query[@]}" -e over --log-file logs/runtime_benchmark/execution/"$dataset"-query-iterative-single-"$i".log
        run-query -i data/"$dataset"/binsort.zst -t binsort -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-binsort-single-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-rebinning-single-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-conversion-single-"$i".log

        # Single query (without IPC)
        run-query -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q "${query[@]}" -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-query-rebinning-single_suppressed-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q "${query[@]}" -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-query-conversion-single_suppressed-"$i".log

        # Query collection
        run-queries -i data/"$dataset"/histograms.zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over --log-file logs/runtime_benchmark/execution/"$dataset"-collection-iterative-single-"$i".log
        run-queries -i data/"$dataset"/histograms.zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over -w "$nproc" --log-file logs/runtime_benchmark/execution/"$dataset"-collection-iterative-parallel-"$i".log
        run-queries -i data/"$dataset"/binsort.zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-collection-binsort-single-"$i".log
        run-queries -i data/"$dataset"/binsort.zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w "$nproc" --log-file logs/runtime_benchmark/execution/"$dataset"-collection-binsort-parallel-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w "$nproc" --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-parallel-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w "$nproc" --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-parallel-"$i".log

        # Query collection (without IPC)
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w"$nproc" --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-parallel_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w"$nproc" --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-parallel_suppressed-"$i".log

        ### Low selectivity runtime experiments ###
        run-queries -i data/"$dataset"/histograms.zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-iterative-single-"$i".log
        run-queries -i data/"$dataset"/binsort.zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-binsort-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-rebinning-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-conversion-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst  --suppress-results --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-rebinning-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst  --suppress-results --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-conversion-single_suppressed-"$i".log

        ### k-cluster runtime experiment ###
        # NOTE: GitTables crashes for bad clustering parameters due to excessive memory usage so we only run the experiment for k>=50
        if [[ "$dataset" == "gittables" ]]; then
            cluster_range=({50..250..10} {300..1000..50})
        else
            cluster_range=(1 2 5 {10..250..10})
        fi
        for k in "${cluster_range[@]}"; do
            cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -a kmeans -c "$k" "$k" -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-k"$k"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-k"$k"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b50000.zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-k$k.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-k"$k"-"$i".log

            # NOTE: We only execute the queries for rebinning here, since conversion is equally fast
            run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning-k"$k".zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/k_cluster/"$dataset"-rebinning-k"$k"-single-"$i".log
            run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning-k"$k".zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/k_cluster/"$dataset"-rebinning-k"$k"-single_suppressed-"$i".log
        done

        ### Bin budget runtime experiment ###
        # NOTE: Again, we need to adjust the parameters for GitTables to avoid crashes
        if [[ "$dataset" == "gittables" ]]; then
            budget_range=(1000 5000 10000 50000 100000)
            k=100
        else
            budget_range=(1000 5000 10000 50000 100000 500000 1000000)
            k=10
        fi
        for budget in "${budget_range[@]}"; do
            cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -a kmeans -c "$k" "$k" -b "$budget" -t quantile --alpha 1 --seed 42 --log-file logs/runtime_benchmark/indexing/"$dataset"-clustering-b"$budget"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m rebinning -o data/"$dataset"/indices/runtime_benchmark --index-file "rebinning-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-rebinning-b"$budget"-"$i".log
            create-index -i data/"$dataset"/clusterings/runtime_benchmark/k"$k"-b"$budget".zst -m conversion -o data/"$dataset"/indices/runtime_benchmark --index-file "conversion-b$budget.zst" --log-file logs/runtime_benchmark/indexing/"$dataset"-conversion-b"$budget"-"$i".log

            # NOTE: We only execute the queries for rebinning here, since conversion is equally fast
            run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning-b"$budget".zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/bin_budget/"$dataset"-rebinning-b"$budget"-single-"$i".log
            run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning-b"$budget".zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/bin_budget/"$dataset"-rebinning-b"$budget"-single_suppressed-"$i".log
        done

        ### Runtime breakdown experiment ###
        # NOTE: Tracing slows down the execution by about an order of magnitude
        run-query -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index_trace -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/index_trace/"$dataset"-rebinning-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index_trace -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/index_trace/"$dataset"-conversion-"$i".log
    done
done

end_time=$(date +%s)
echo Executed script in $((end_time - start_time))s.
