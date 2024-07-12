#!/bin/bash
#
cargo build --release
if [ "$#" -eq 1 ]; then
    perf record --call-graph dwarf target/release/isera -f $1
fi

perf script | ~/.cargo/bin/inferno-collapse-perf > stacks.folded
cat stacks.folded | ~/.cargo/bin/inferno-flamegraph > flamegraph.svg
rm perf.data
rm stacks.folded
firefox flamegraph.svg
