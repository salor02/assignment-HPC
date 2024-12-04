#!/bin/bash

EXE_PATH="$(pwd)/atax_acc"

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=$1_DATASET OPTIMIZATION=OPTIMIZATION_$2 CHECK_RESULTS=1 clean run
