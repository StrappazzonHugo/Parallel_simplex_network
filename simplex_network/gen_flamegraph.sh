#!/bin/bash
#
cargo build --release
perf record --call-graph dwarf target/release/main #~/Documents/graph/gridgen/grid_long_20e.min
perf script | ~/.cargo/bin/inferno-collapse-perf > stacks.folded
cat stacks.folded | ~/.cargo/bin/inferno-flamegraph > flamegraph.svg
rm perf.data
rm stacks.folded
firefox flamegraph.svg
