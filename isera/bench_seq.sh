#!/bin/bash

cargo build --release
FILES="$1/*.min"
for f in $FILES
do
  ./target/release/isera $f 1 1 >> res_seq.csv 
done
