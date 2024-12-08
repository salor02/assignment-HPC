#!/bin/bash

# Percorso dell'eseguibile
EXE_PATH="$(pwd)/atax_acc"

# File di output
OUTPUT_FILE="execution_times.txt"
LOG_FILE="execution_log.txt"

# Numero massimo di tentativi in caso di varianza troppo alta
MAX_RETRIES=5

# Array delle ottimizzazioni da eseguire
optimizations=("nulla" "1" "2" "3")

# Array dei dataset da eseguire
datasets=("MINI" "SMALL" "STANDARD" "LARGE")

# Inizializza o svuota i file di output e log
> "$OUTPUT_FILE"
> "$LOG_FILE"

# Funzione per eseguire il benchmark con controllo della varianza
run_benchmark() {
    local dataset_type=$1
    local optimization_label=$2
    local attempt=1
    local success=0
    local normalized_time=""

    while [ $attempt -le "$MAX_RETRIES" ]; do
        echo "Esecuzione del benchmark per $dataset_type con $optimization_label (Tentativo $attempt)..." >&2

        # Rimuovi l'eseguibile precedente
        rm -f ./atax_acc >> "$LOG_FILE" 2>&1

        # Compila con i parametri specificati, redirigendo i dettagli tecnici nel file di log
        if [ "$optimization_label" == "SEQUENTIAL" ]; then
            make -s EXERCISE=atax.cu DATASET_TYPE=${dataset_type}_DATASET clean run >> "$LOG_FILE" 2>&1
        else
            make -s EXERCISE=atax.cu DATASET_TYPE=${dataset_type}_DATASET OPTIMIZATION=${optimization_label} clean all >> "$LOG_FILE" 2>&1
        fi

        # Verifica se la compilazione è andata a buon fine
        if [ ! -f "$EXE_PATH" ]; then
            echo "❌ Errore nella compilazione per $dataset_type con $optimization_label. Dettagli nel file di log." >&2
            return 1
        fi

        # Esegui il benchmark e cattura l'output
        benchmark_output=$(bash ../../../utilities/time_benchmark.sh "$EXE_PATH" 2>>"$LOG_FILE")

        # Controlla se l'output contiene un warning di varianza alta
        if echo "$benchmark_output" | grep -q "WARNING"; then
            echo "⚠️  Varianza troppo alta per $dataset_type con $optimization_label. Riprovo..." >&2
            attempt=$((attempt + 1))
            sleep 1
        else
            # Estrai il tempo normalizzato
            normalized_time=$(echo "$benchmark_output" | awk '/Normalized time:/ {print $4}')
            if [ -z "$normalized_time" ]; then
                echo "❌ Impossibile estrarre il tempo normalizzato per $dataset_type con $optimization_label." >&2
                return 1
            fi
            echo "✅ Benchmark completato per $dataset_type con $optimization_label: $normalized_time secondi." >&2
            success=1
            break
        fi
    done

    # Verifica se il benchmark è riuscito
    if [ $success -ne 1 ]; then
        echo "❌ Benchmark fallito per $dataset_type con $optimization_label dopo $MAX_RETRIES tentativi." >&2
        return 1
    fi

    # Ritorna solo il tempo normalizzato
    echo "$normalized_time"
}

# Loop per ogni ottimizzazione e dataset
for opt in "${optimizations[@]}"; do
    if [ "$opt" == "nulla" ]; then
        optimization_label="SEQUENTIAL"
    else
        optimization_label="OPTIMIZATION_$opt"
    fi

    echo -e "\n=========================================="
    echo "=== Esecuzione dell'ottimizzazione: $optimization_label ==="
    echo "=========================================="

    echo "$optimization_label" >> "$OUTPUT_FILE"

    for dataset in "${datasets[@]}"; do
        # Esegui il benchmark e cattura solo il tempo normalizzato
        time=$(run_benchmark "$dataset" "$optimization_label")
        if [ $? -ne 0 ]; then
            echo "Errore durante il benchmark di $dataset con $optimization_label. Controlla $LOG_FILE." >&2
            exit 1
        fi

        # Scrivi solo il risultato normalizzato nel file di output
        echo "$dataset: $time" >> "$OUTPUT_FILE"
    done

    echo "__________________________________________" >> "$OUTPUT_FILE"
done

echo "Benchmark completato. Risultati salvati in $OUTPUT_FILE."
