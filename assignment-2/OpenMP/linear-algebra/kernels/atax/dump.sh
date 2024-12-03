#!/bin/bash

make EXERCISE=atax.cu DATASET_TYPE=MINI_DATASET OPTIMIZATION=OPTIMIZATION_$1 DUMP_ARRAYS=1 clean run  
./atax_acc 2> ./dump/mini/"$1"
make EXERCISE=atax.cu DATASET_TYPE=SMALL_DATASET OPTIMIZATION=OPTIMIZATION_$1 DUMP_ARRAYS=1 clean run 
./atax_acc 2> ./dump/small/"$1"
make EXERCISE=atax.cu DATASET_TYPE=STANDARD_DATASET OPTIMIZATION=OPTIMIZATION_$1 DUMP_ARRAYS=1  clean run
./atax_acc 2> ./dump/standard/"$1"
make EXERCISE=atax.cu DATASET_TYPE=LARGE_DATASET OPTIMIZATION=OPTIMIZATION_$1 DUMP_ARRAYS=1 clean  run  ./atax_acc 2> ./dump/large/"$1"


