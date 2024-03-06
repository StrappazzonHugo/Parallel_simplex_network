#!/bin/bash
#
perf record --call-graph dwarf target/release/main
perf script | ~/.cargo/bin/inferno-collapse-perf > stacks.folded
cat stacks.folded | ~/.cargo/bin/inferno-flamegraph > flamegraph.svg
rm perf.data
rm stacks.folded
