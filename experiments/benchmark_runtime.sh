#!/bin/bash

echo "Executing runtime benchmark"

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

    # Binsort setup
    for m in 10 50 100 500 1000 5000; do
        compute-histograms -i data/"$dataset"/pq -o data/"$dataset"/histograms_m"$m".zst --bin-range "$m" "$m" --log-file logs/runtime_benchmark/binsort/"$dataset"-construction-m"$m".log
        compute-binsort -i data/"$dataset"/histograms_m"$m".zst -o data/"$dataset"/binsort_m"$m".zst
    done

    query=("0.1" "lt" "50")
    nproc=$(nproc)
    for i in {1..5}; do
        ### General runtime experiments ###
        # Single query
        run-query -i data/"$dataset"/histograms.zst -t histograms -q "${query[@]}" -e over --log-file logs/runtime_benchmark/execution/"$dataset"-query-iterative-single-"$i".log
        run-query -i data/"$dataset"/binsort.zst -t binsort -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-binsort-single-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-rebinning-single-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/execution/"$dataset"-query-conversion-single-"$i".log

        # Single query (without processing results)
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

        # Query collection (without processing results)
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w"$nproc" --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-rebinning-parallel_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -w"$nproc" --suppress-results --log-file logs/runtime_benchmark/execution/"$dataset"-collection-conversion-parallel_suppressed-"$i".log

        ### Low selectivity runtime experiments ###
        run-queries -i data/"$dataset"/histograms.zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-iterative-single-"$i".log
        run-queries -i data/"$dataset"/binsort.zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-binsort-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-rebinning-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-conversion-single-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --suppress-results --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-rebinning-single_suppressed-"$i".log
        run-queries -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index -q data/"$dataset"/queries/runtime_benchmark.zst -m recall -f data/"$dataset"/filters/01.zst --suppress-results --log-file logs/runtime_benchmark/low_selectivity/"$dataset"-collection-conversion-single_suppressed-"$i".log

        ### Binsort experiments ###
        for m in 10 50 100 500 1000 5000; do
            run-queries -i data/"$dataset"/histograms_m"$m".zst -t histograms -q data/"$dataset"/queries/runtime_benchmark.zst -e over --log-file logs/runtime_benchmark/binsort/"$dataset"-iterative-m"$m"-"$i".log
            run-queries -i data/"$dataset"/binsort_m"$m".zst -t binsort -q data/"$dataset"/queries/runtime_benchmark.zst -m recall --log-file logs/runtime_benchmark/binsort/"$dataset"-binsort-m"$m"-"$i".log
        done

        ### Runtime breakdown experiment ###
        # NOTE: Tracing slows down the execution by about an order of magnitude
        run-query -i data/"$dataset"/indices/runtime_benchmark/rebinning.zst -t index_trace -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/index_trace/"$dataset"-rebinning-"$i".log
        run-query -i data/"$dataset"/indices/runtime_benchmark/conversion.zst -t index_trace -q "${query[@]}" -m recall --log-file logs/runtime_benchmark/index_trace/"$dataset"-conversion-"$i".log
    done
done

end_time=$(date +%s)
echo Executed runtime benchmark in $((end_time - start_time))s.
