#!/bin/sh

readonly BENCHMARK_FILENAME="exports/benchmarks/new_constraints.txt"
rm -f $BENCHMARK_FILENAME

for k in $(seq 1 6); 
do 
    N=$((20 * k))
    python3 sdp-solver-symmetrized.py 6 $N --use-new-constraints --repeats 30
done 