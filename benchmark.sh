#!/bin/sh

readonly BENCHMARK_FILENAME="exports/benchmarks/default.csv"
rm -f $BENCHMARK_FILENAME

for k in $(seq 1 6); 
do 
    N=$((20 * k))
    python3 solve_sdp.py 6 $N --repeats 30
done 