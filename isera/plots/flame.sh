#!/bin/bash

if [ "$#" -eq 1 ]; then
    perf record --call-graph dwarf ../target/release/isera -f ~/Documents/graph/benchIsera/a/$1
fi

perf script | ~/.cargo/bin/inferno-collapse-perf > stacks.folded
cat stacks.folded | ~/.cargo/bin/inferno-flamegraph > $1.svg
rm perf.data
rm stacks.folded
