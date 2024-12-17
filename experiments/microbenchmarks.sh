#!/bin/bash

echo "Executing microbenchmarks"

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

for dataset in "open_data_usa" "gittables"; do
    cp data/"$dataset"/queries/accuracy_benchmark/test-all.zst data/"$dataset"/queries/microbenchmarks.zst
    cp data/"$dataset"/results/accuracy_benchmark/ground_truth-test-all.zst data/"$dataset"/results/microbenchmarks.zst

    if [ "$dataset" == "open_data_usa" ]; then
        k_values=(1 {10..50..10} {100..1000..50})
    else
        k_values=({100..1000..50})
    fi


    for k in "${k_values[@]}"; do
        # Create index and measure its size
        cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/microbenchmarks/k"$k"-b50000.zst -a kmeans -c "$k" "$k" -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/microbenchmarks/indexing/"$dataset"-clustering-k"$k".log
        create-index -i data/"$dataset"/clusterings/microbenchmarks/k"$k"-b50000.zst -m rebinning -o data/"$dataset"/indices/microbenchmarks --index-file "rebinning-k$k.zst" --log-file logs/microbenchmarks/indexing/"$dataset"-rebinning-k"$k".log
        create-index -i data/"$dataset"/clusterings/microbenchmarks/k"$k"-b50000.zst -m conversion -o data/"$dataset"/indices/microbenchmarks --index-file "conversion-k$k.zst" --log-file logs/microbenchmarks/indexing/"$dataset"-conversion-k"$k".log

        for i in {1..5}; do
            # Measure runtime
            # NOTE: We only execute the queries for rebinning here, since conversion is equally fast
            run-queries -i data/"$dataset"/indices/microbenchmarks/rebinning-k"$k".zst -t index -q data/"$dataset"/queries/microbenchmarks.zst -m recall --log-file logs/microbenchmarks/runtime/"$dataset"-rebinning-k"$k"-single-"$i".log
            run-queries -i data/"$dataset"/indices/microbenchmarks/rebinning-k"$k".zst -t index -q data/"$dataset"/queries/microbenchmarks.zst -m recall --suppress-results --log-file logs/microbenchmarks/runtime/"$dataset"-rebinning-k"$k"-single_suppressed-"$i".log
        done

        # Measure accuracy
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/microbenchmarks/rebinning-k"$k".zst \
            -q data/"$dataset"/queries/microbenchmarks.zst \
            -t data/"$dataset"/results/microbenchmarks.zst \
            --log-file logs/microbenchmarks/accuracy/"$dataset"-rebinning-k"$k".zst
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/microbenchmarks/conversion-k"$k".zst \
            -q data/"$dataset"/queries/microbenchmarks.zst \
            -t data/"$dataset"/results/microbenchmarks.zst \
            --log-file logs/microbenchmarks/accuracy/"$dataset"-conversion-k"$k".zst
    done

    if [ "$dataset" == "gittables" ]; then
        continue
    fi

    for b in 100 500 1000 5000 10000 50000 100000 500000 1000000; do
        # Create index and measure its size
        cluster-histograms -i data/"$dataset"/histograms.zst -o data/"$dataset"/clusterings/microbenchmarks/k100-b"$b".zst -a kmeans -c 100 100 -b "$b" -t quantile --alpha 1 --seed 42 --log-file logs/microbenchmarks/indexing/"$dataset"-clustering-b"$b".log
        create-index -i data/"$dataset"/clusterings/microbenchmarks/k100-b"$b".zst -m rebinning -o data/"$dataset"/indices/microbenchmarks --index-file "rebinning-b$b.zst" --log-file logs/microbenchmarks/indexing/"$dataset"-rebinning-b"$b".log
        create-index -i data/"$dataset"/clusterings/microbenchmarks/k100-b"$b".zst -m conversion -o data/"$dataset"/indices/microbenchmarks --index-file "conversion-b$b.zst" --log-file logs/microbenchmarks/indexing/"$dataset"-conversion-b"$b".log

        for i in {1..5}; do
            # Measure runtime
            # NOTE: We only execute the queries for rebinning here, since conversion is equally fast
            run-queries -i data/"$dataset"/indices/microbenchmarks/rebinning-b"$b".zst -t index -q data/"$dataset"/queries/microbenchmarks.zst -m recall --log-file logs/microbenchmarks/runtime/"$dataset"-rebinning-b"$b"-single-"$i".log
            run-queries -i data/"$dataset"/indices/microbenchmarks/rebinning-b"$b".zst -t index -q data/"$dataset"/queries/microbenchmarks.zst -m recall --suppress-results --log-file logs/microbenchmarks/runtime/"$dataset"-rebinning-b"$b"-single_suppressed-"$i".log
        done

        # Measure accuracy
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/microbenchmarks/rebinning-b"$b".zst \
            -q data/"$dataset"/queries/microbenchmarks.zst \
            -t data/"$dataset"/results/microbenchmarks.zst \
            --log-file logs/microbenchmarks/accuracy/"$dataset"-rebinning-b"$b".zst
        python experiments/compute_fainder_results.py \
            -i data/"$dataset"/indices/microbenchmarks/conversion-b"$b".zst \
            -q data/"$dataset"/queries/microbenchmarks.zst \
            -t data/"$dataset"/results/microbenchmarks.zst \
            --log-file logs/microbenchmarks/accuracy/"$dataset"-conversion-b"$b".zst
    done
done

end_time=$(date +%s)
echo Executed microbenchmarks in $((end_time - start_time))s.
