#!/usr/bin/env python3

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys

# File di input e directory di output
PROFILING_RESULTS = "profiling_results.txt"
PLOT_DIR = "aggregated_gpu_plots_per_dataset"

# Crea la cartella per i grafici se non esiste
os.makedirs(PLOT_DIR, exist_ok=True)

# Struttura dati: data[ottimizzazione][categoria][dataset] = tempo totale
data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# Variabili di stato
current_optimization = None
current_dataset = None
capture_gpu_timings = False

# Pattern regex per identificare le sezioni e le operazioni GPU
optimization_pattern = re.compile(r'^(OPTIMIZATION_\d+)$')  # Include SEQUENTIAL
dataset_pattern = re.compile(r'^([A-Z]+)\s*:\s*([\d.]+)\s*s$')  # Es: MINI      : 1.611967 s
gpu_timings_start_pattern = re.compile(r'^GPU Timings:$')

# Funzione di categorizzazione
def categorize_operation(operation_name):
    operation_lower = operation_name.lower()
    # Riconosce 'compute_y', 'compute_y_shared', 'compute_tmp', 'compute_tmp_shared', 'cudalaunchkernel'
    if "compute" in operation_lower or "launchkernel" in operation_lower:
        return "Kernel"
    elif "memcpy" in operation_lower:
        return "Memory Operations"
    elif "synchronize" in operation_lower:
        return "Synchronization"
    else:
        return None  # Esclude le operazioni non desiderate

# Legge il file di profiling
with open(PROFILING_RESULTS, "r") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        # Verifica se la linea indica una ottimizzazione (incluso SEQUENTIAL)
        opt_match = optimization_pattern.match(line)
        if opt_match:
            current_optimization = opt_match.group(1)
            continue
        # Verifica se la linea indica un dataset
        dataset_match = dataset_pattern.match(line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
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
            # Split the line at the last ':'
            if ':' not in line:
                # Linea non conforme, skip
                print(f"Line {line_num}: Skipping non-conforme line: {line}", file=sys.stderr)
                continue
            operation_part, time_part = line.rsplit(':', 1)
            operation_part = operation_part.strip()
            time_part = time_part.strip()

            # Estrai il nome dell'operazione
            if operation_part.startswith('[') and operation_part.endswith(']'):
                operation = operation_part[1:-1]
            else:
                # Take the last token as the operation name
                tokens = operation_part.split()
                operation = tokens[-1] if tokens else "Unknown"

            # Estrai il tempo in secondi
            # Use regex to extract the first number
            time_match = re.match(r'^([\d.]+)', time_part)
            if time_match:
                time_sec_str = time_match.group(1)
                try:
                    time_sec = float(time_sec_str)
                except ValueError:
                    print(f"Line {line_num}: Unable to parse time: {time_part}", file=sys.stderr)
                    continue
            else:
                print(f"Line {line_num}: Unable to parse time: {time_part}", file=sys.stderr)
                continue

            # Determina la categoria dell'operazione
            category = categorize_operation(operation)
            if category:
                data[current_optimization][category][current_dataset] += time_sec
                # Debugging: Stampa le operazioni categorizzate
                # print(f"Line {line_num}: Categorized operation '{operation}' as '{category}' with time {time_sec} s")
            else:
                # Ignora l'operazione
                pass

# Definisci le categorie e i dataset
categories = ["Kernel", "Memory Operations", "Synchronization"]
datasets = ["MINI", "SMALL", "STANDARD", "LARGE"]

# Assicurati che tutti i dataset abbiano tutte le categorie
for optimization in data.keys():
    for category in categories:
        for dataset in datasets:
            if dataset not in data[optimization][category]:
                data[optimization][category][dataset] = 0.0  # Assegna 0 se mancano dati

# Genera i grafici aggregati
for optimization, categories_data in data.items():
    if not categories_data:
        continue  # Salta se non ci sono categorie

    x_labels = categories
    x = range(len(x_labels))  # Posizioni delle categorie
    width = 0.2  # Larghezza delle barre

    plt.figure(figsize=(14, 10))
    for i, dataset in enumerate(datasets):
        times = [categories_data[cat][dataset] for cat in x_labels]
        plt.bar([pos + i * width for pos in x], times, width=width, label=f"{dataset}")

    plt.xticks(
        [pos + (len(datasets) / 2 - 0.5) * width for pos in x],
        x_labels,
        fontsize=12,
        rotation=45
    )
    plt.xlabel('Categoria Operazione', fontsize=14)
    plt.yscale('log')  # Imposta la scala logaritmica sull'asse Y
    plt.ylabel('Tempo Totale (s) [Scala Logaritmica]', fontsize=14)
    plt.title(f'Tempi Aggregati per Dataset e Categoria - {optimization}', fontsize=16)
    plt.legend(title="Dataset", fontsize=12)
    plt.tight_layout()

    # Pulizia del nome dell'ottimizzazione per il nome del file
    safe_optimization = re.sub(r'[^\w\s-]', '', optimization).replace(' ', '_')
    plt.savefig(os.path.join(PLOT_DIR, f'{safe_optimization}_aggregated_gpu_timings_per_dataset.png'))
    plt.close()

print(f"Grafici aggregati per dataset generati nella cartella '{PLOT_DIR}'.")