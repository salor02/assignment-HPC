#!/bin/bash

EXE_PATH="$(pwd)/atax_acc"

rm -f ./atax_acc
make EXT_CFLAGS="-DPOLYBENCH_TIME -D$1_DATASET -D$2" clean all run
bash ../../../utilities/time_benchmark.sh $EXE_PATH
