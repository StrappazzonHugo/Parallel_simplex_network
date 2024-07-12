#!/bin/bash
mkdir instances;
for i in `cat instancelink.txt`; do
   wget $i;
done





