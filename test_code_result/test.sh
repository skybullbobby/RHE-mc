#!/bin/bash


../cmake-build-debug/RHEmc \
    -g adr_new.txt \
    -c sample_new.cov \
    -p phenos_new.plink \
    -k 100 -jn 100 \
    -o newout.txt \
    -annot annot_new.txt
