#!/bin/bash

# File temporaneo per salvare l'output
RAW_OUTPUT="profiling_raw_output.txt"

# Array delle ottimizzazioni e dei dataset
optimizations=("SEQUENTIAL" "OPTIMIZATION_1" "OPTIMIZATION_2")
datasets=("MINI" "SMALL" "STANDARD" "LARGE")

# Pulisce il file temporaneo
> "$RAW_OUTPUT"

# Loop per ogni ottimizzazione
for optimization in "${optimizations[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Profiling $optimization con dataset $dataset..." >&2
        echo "OPTIMIZATION: $optimization" >> "$RAW_OUTPUT"
        echo "DATASET: $dataset" >> "$RAW_OUTPUT"
        
        # Esegui make e redirigi l'output
        make EXERCISE=atax.cu DATASET_TYPE=${dataset}_DATASET OPTIMIZATION=${optimization} clean all run profile >> "$RAW_OUTPUT" 2>&1
        echo "______________________________________" >> "$RAW_OUTPUT"
    done
done

echo "Profiling completato. Dati salvati in $RAW_OUTPUT."