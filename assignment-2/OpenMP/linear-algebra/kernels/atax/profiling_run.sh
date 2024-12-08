#!/bin/bash

# File di output per i risultati
OUTPUT_FILE="profiling_results.txt"
LOG_FILE="profiling_log.txt"

# Array delle ottimizzazioni e dei dataset
optimizations=("SEQUENTIAL" "OPTIMIZATION_1" "OPTIMIZATION_2" "OPTIMIZATION_3")
datasets=("MINI" "SMALL" "STANDARD" "LARGE")

# Inizializza o svuota i file di output e log
> "$OUTPUT_FILE"
> "$LOG_FILE"

# Funzione per eseguire il profiling
run_profiling() {
    local dataset=$1
    local optimization=$2

    echo "Esecuzione profiling per $optimization con dataset $dataset..." >&2

    # Pulisce, compila e lancia il benchmark con profiling
    profiling_output=$(make EXERCISE=atax.cu DATASET_TYPE=${dataset}_DATASET OPTIMIZATION=${optimization} clean all run profile 2>>"$LOG_FILE")

    # Estrarre il runtime (tempo totale del benchmark)
    runtime=$(echo "$profiling_output" | awk '/^[0-9]+\.[0-9]+$/ {print $1; exit}')

    # Estrarre il tempo GPU più significativo (tempo massimo di attività GPU)
    gpu_time=$(echo "$profiling_output" | awk '/ms/ {gsub("ms", "", $1); print $1 / 1000}' | sort -nr | head -n 1)

    # Se non c'è un tempo GPU valido, imposta N/A
    if [ -z "$gpu_time" ]; then
        gpu_time="N/A"
    fi

    # Se non c'è un runtime valido, imposta N/A
    if [ -z "$runtime" ]; then
        runtime="N/A"
    fi

    # Ritorna i risultati
    echo "$runtime" "$gpu_time"
}

# Loop per ogni ottimizzazione
for optimization in "${optimizations[@]}"; do
    echo "$optimization" >> "$OUTPUT_FILE"

    # Loop per ogni dataset
    for dataset in "${datasets[@]}"; do
        # Esegui il profiling e cattura i risultati
        results=$(run_profiling "$dataset" "$optimization")
        runtime=$(echo "$results" | awk '{print $1}')
        gpu_time=$(echo "$results" | awk '{print $2}')

        # Scrive il risultato formattato nel file di output
        printf "%-10s: %-10s GPU: %-10s\n" "$dataset" "$runtime" "$gpu_time" >> "$OUTPUT_FILE"
    done

    # Separatore tra le ottimizzazioni
    echo "______________________________________" >> "$OUTPUT_FILE"
done

echo "Profiling completato. Risultati salvati in $OUTPUT_FILE."
