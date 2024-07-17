#!/bin/bash
mkdir instances;
cd instances;
for i in `cat instancelink.txt`; do
   wget $i;
done





