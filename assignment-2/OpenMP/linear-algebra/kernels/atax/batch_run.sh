EXE_PATH="$(pwd)/atax_acc"

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=MINI_DATASET OPTIMIZATION=OPTIMIZATION_$1 clean run
bash ../../../utilities/time_benchmark.sh $EXE_PATH

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=SMALL_DATASET OPTIMIZATION=OPTIMIZATION_$1 clean run
bash ../../../utilities/time_benchmark.sh $EXE_PATH

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=STANDARD_DATASET OPTIMIZATION=OPTIMIZATION_$1 clean run
bash ../../../utilities/time_benchmark.sh $EXE_PATH

rm -f ./atax_acc
make EXERCISE=atax.cu DATASET_TYPE=LARGE_DATASET OPTIMIZATION=OPTIMIZATION_$1 clean run
bash ../../../utilities/time_benchmark.sh $EXE_PATH
