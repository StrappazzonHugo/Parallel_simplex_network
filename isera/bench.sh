#!/bin/bash

cargo build --release

FILES="$1/*.min"
for f in $FILES
do
  ./target/release/isera $f 1 1 >> res.csv 
  ./target/release/isera $f 2 1 >> res.csv 
  ./target/release/isera $f 2 2 >> res.csv 
  ./target/release/isera $f 4 2 >> res.csv 
  ./target/release/isera $f 4 4 >> res.csv 
  ./target/release/isera $f 8 4 >> res.csv 
  ./target/release/isera $f 8 8 >> res.csv 
done





