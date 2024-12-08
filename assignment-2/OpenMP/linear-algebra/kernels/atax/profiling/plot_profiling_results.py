#!/usr/bin/env python3

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# File di input e output
PROFILING_RESULTS = "profiling_results.txt"
PLOT_DIR = "gpu_plots"

# Crea la cartella per i grafici se non esiste
os.makedirs(PLOT_DIR, exist_ok=True)

# Struttura dati: data[ottimizzazione][operazione] = lista di tempi
data = defaultdict(lambda: defaultdict(list))

# Variabili di stato
current_optimization = None
current_dataset = None
capture_gpu_timings = False

# Pattern regex per identificare le sezioni e le operazioni GPU
optimization_pattern = re.compile(r'^(OPTIMIZATION_\d+)$')  # Esclude SEQUENTIAL
dataset_pattern = re.compile(r'^([A-Z]+)\s*:\s*([\d.]+)$')
gpu_timings_start_pattern = re.compile(r'^GPU Timings:$')
gpu_timing_line_pattern = re.compile(r'^\s*([\d.]+[a-zA-Z]+)\s+([\d.]+[a-zA-Z]+)\s+(.*?):\s+([\d.]+)\s+s$')

# Legge il file di profiling
with open(PROFILING_RESULTS, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Verifica se la linea indica una ottimizzazione (esclude SEQUENTIAL)
        opt_match = optimization_pattern.match(line)
        if opt_match:
            current_optimization = opt_match.group(1)
            continue
        # Verifica se la linea indica un dataset
        dataset_match = dataset_pattern.match(line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            runtime = float(dataset_match.group(2))
            # Reset della flag di cattura GPU Timings
            capture_gpu_timings = False
            continue
        # Verifica se inizia la sezione GPU Timings
        if gpu_timings_start_pattern.match(line):
            capture_gpu_timings = True
            continue
        # Cattura le linee di GPU Timings
        if capture_gpu_timings and current_optimization and current_dataset:
            # Verifica se la linea è un separatore
            if line.startswith("______________________________________"):
                capture_gpu_timings = False
                continue
            # Verifica se la linea corrisponde al pattern delle GPU Timings
            gpu_match = gpu_timing_line_pattern.match(line)
            if gpu_match:
                time1 = gpu_match.group(1)  # Es: 80.002us
                time2 = gpu_match.group(2)  # Es: 1.4899ms
                operation = gpu_match.group(3).strip()  # Es: cudaMemcpy
                time_sec = float(gpu_match.group(4))  # Es: 0.003036

                # Pulizia del nome dell'operazione
                operation = operation.replace('[', '').replace(']', '')
                operation = re.sub(r'\([^)]*\)', '', operation).strip()

                # Aggiungi il tempo all'operazione corrente
                data[current_optimization][operation].append(time_sec)
            else:
                # Linea non corrispondente al pattern, ignora o logga se necessario
                pass

# Calcola i tempi medi per operazione e ottimizzazione
average_data = defaultdict(dict)

for optimization, operations in data.items():
    for operation, times in operations.items():
        avg_time = sum(times) / len(times) if times else 0.0
        average_data[optimization][operation] = avg_time

# Genera i grafici per ciascuna ottimizzazione
for optimization, operations in average_data.items():
    if not operations:
        continue  # Salta se non ci sono operazioni
    operations_sorted = sorted(operations.keys())
    avg_times = [operations[op] for op in operations_sorted]
    
    plt.figure(figsize=(12, 8))
    plt.bar(operations_sorted, avg_times, color='skyblue')
    plt.xlabel('Operazione GPU', fontsize=14)
    plt.ylabel('Tempo Medio (s)', fontsize=14)
    plt.title(f'Tempi Medi per Operazione GPU - {optimization}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Pulizia del nome dell'ottimizzazione per il nome del file
    safe_optimization = re.sub(r'[^\w\s-]', '', optimization).replace(' ', '_')
    plt.savefig(os.path.join(PLOT_DIR, f'{safe_optimization}_gpu_timings.png'))
    plt.close()

print(f"Grafici generati nella cartella '{PLOT_DIR}'.")