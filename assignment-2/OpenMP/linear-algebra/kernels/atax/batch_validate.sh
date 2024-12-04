#!/bin/bash

EXE_PATH="$(pwd)/atax_acc"

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=MINI_DATASET OPTIMIZATION=OPTIMIZATION_$1 CHECK_RESULTS=1 clean run

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=SMALL_DATASET OPTIMIZATION=OPTIMIZATION_$1 CHECK_RESULTS=1 clean run

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=STANDARD_DATASET OPTIMIZATION=OPTIMIZATION_$1 CHECK_RESULTS=1 clean run

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=LARGE_DATASET OPTIMIZATION=OPTIMIZATION_$1 CHECK_RESULTS=1 clean run
