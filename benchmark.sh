#!/bin/sh

readonly BENCHMARK_FILENAME="exports/benchmarks/old_constraints.csv"
rm -f $BENCHMARK_FILENAME

for k in $(seq 1 6); 
do 
    N=$((20 * k))
    python3 sdp-solver-symmetrized.py 6 $N --repeats 30
done 