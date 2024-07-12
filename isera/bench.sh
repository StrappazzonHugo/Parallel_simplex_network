#!/bin/bash

cargo build --release

FILES=$1
./target/release/isera -f $FILES -k 1 -n 1 >> res.csv 
./target/release/isera -f $FILES -k 1 -n 2 >> res.csv 
./target/release/isera -f $FILES -k 2 -n 2 >> res.csv 
./target/release/isera -f $FILES -k 2 -n 4 >> res.csv 
./target/release/isera -f $FILES -k 4 -n 4 >> res.csv 
./target/release/isera -f $FILES -k 4 -n 8 >> res.csv 
./target/release/isera -f $FILES -k 8 -n 8 >> res.csv 





