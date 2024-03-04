#!/bin/bash

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"
start_time=$(date +%s)

### Setup ###
dataset="gittables"
query=("0.1" "lt" "50")
# nproc=$(nproc)

for f in 025 050 100 200; do
    cluster-histograms -i data/"$dataset"/histograms_sf"$f".zst -o data/"$dataset"/clusterings/scalability_benchmark/sf"$f".zst -a kmeans -c 125 150 -b 50000 -t quantile --alpha 1 --seed 42 --log-file logs/scalability_benchmark/indexing/"$dataset"-clustering-sf"$f".log
    create-index -i data/"$dataset"/clusterings/scalability_benchmark/sf"$f".zst -m rebinning -o data/"$dataset"/indices/scalability_benchmark --index-file rebinning-sf"$f".zst --log-file logs/scalability_benchmark/indexing/"$dataset"-rebinning-sf"$f".log
    create-index -i data/"$dataset"/clusterings/scalability_benchmark/sf"$f".zst -m conversion -o data/"$dataset"/indices/scalability_benchmark --index-file conversion-sf"$f".zst --log-file logs/scalability_benchmark/indexing/"$dataset"-conversion-sf"$f".log

    for i in {1..5}; do
        ### General runtime experiments ###
        # Single query
        run-query -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q "${query[@]}" -m recall --log-file logs/scalability_benchmark/execution/"$dataset"-query-rebinning-single-sf"$f"-"$i".log
        run-query -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q "${query[@]}" -m recall --log-file logs/scalability_benchmark/execution/"$dataset"-query-conversion-single-sf"$f"-"$i".log

        # Single query (without IPC)
        run-query -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q "${query[@]}" -m recall --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-query-rebinning-single_suppressed-sf"$f"-"$i".log
        run-query -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q "${query[@]}" -m recall --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-query-conversion-single_suppressed-sf"$f"-"$i".log

        # Query collection
        run-queries -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall --log-file logs/scalability_benchmark/execution/"$dataset"-collection-rebinning-single-sf"$f"-"$i".log
        # run-queries -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall -w "$nproc" --log-file logs/scalability_benchmark/execution/"$dataset"-collection-rebinning-parallel-sf"$f"-"$i".log
        run-queries -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall --log-file logs/scalability_benchmark/execution/"$dataset"-collection-conversion-single-sf"$f"-"$i".log
        # run-queries -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall -w "$nproc" --log-file logs/scalability_benchmark/execution/"$dataset"-collection-conversion-parallel-sf"$f"-"$i".log

        # Query collection (without IPC)
        run-queries -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-collection-rebinning-single_suppressed-sf"$f"-"$i".log
        # run-queries -i data/"$dataset"/indices/scalability_benchmark/rebinning-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall -w"$nproc" --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-collection-rebinning-parallel_suppressed-sf"$f"-"$i".log
        run-queries -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-collection-conversion-single_suppressed-sf"$f"-"$i".log
        # run-queries -i data/"$dataset"/indices/scalability_benchmark/conversion-sf"$f".zst -t index -q data/"$dataset"/queries/test-all.zst -m recall -w"$nproc" --suppress-results --log-file logs/scalability_benchmark/execution/"$dataset"-collection-conversion-parallel_suppressed-sf"$f"-"$i".log
    done
done

end_time=$(date +%s)
echo Executed script in $((end_time - start_time))s.
