#!/bin/bash

cargo build --release

FILES="$1/*.min"
for f in $FILES
do
  ./target/release/isera $f -k 1 -n 1 >> res.csv 
  ./target/release/isera $f -k 1 -n 2 >> res.csv 
  ./target/release/isera $f -k 2 -n 2 >> res.csv 
  ./target/release/isera $f -k 2 -n 4 >> res.csv 
  ./target/release/isera $f -k 4 -n 4 >> res.csv 
  ./target/release/isera $f -k 4 -n 8 >> res.csv 
  ./target/release/isera $f -k 8 -n 8 >> res.csv 
done





