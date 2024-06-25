#!/bin/bash

cargo build --release
./target/release/isera $1 2 1 >> res.csv 
./target/release/isera $1 2 2 >> res.csv 
./target/release/isera $1 2 4 >> res.csv 
./target/release/isera $1 2 8 >> res.csv 
