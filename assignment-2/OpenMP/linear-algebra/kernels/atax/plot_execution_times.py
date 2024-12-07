#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os

input_file = 'execution_times.txt'

data = {}  # data[optimization][dataset] = time_in_seconds
optimizations = []
datasets = ['MINI', 'SMALL', 'STANDARD', 'LARGE']

with open(input_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

current_optimization = None
for line in lines:
    if line.startswith("OPTIMIZATION_") or line.startswith("SEQUENTIAL"):
        # Nuova ottimizzazione
        current_optimization = line
        optimizations.append(current_optimization)
    elif line.startswith("MINI:") or line.startswith("SMALL:") or line.startswith("STANDARD:") or line.startswith("LARGE:"):
        # Riga con dataset e tempo
        # Esempio: "MINI: 0.09434933"
        parts = line.split(":")
        dataset = parts[0].strip()
        time_str = parts[1].strip()
        time_val = float(time_str)
        if current_optimization not in data:
            data[current_optimization] = {}
        data[current_optimization][dataset] = time_val
    elif line.startswith("_"):
        # Linea di underscore - separatore, ignoriamo
        continue

# Controllo che tutti i dataset siano presenti per ogni ottimizzazione
for opt in optimizations:
    for dset in datasets:
        if dset not in data[opt]:
            print(f"Warning: {dset} not found in {opt}")
            data[opt][dset] = None

# Creiamo la cartella per i plot
os.makedirs('plots', exist_ok=True)

# Plot dei tempi di esecuzione
plt.figure(figsize=(10, 6))
for opt in optimizations:
    times = [data[opt][d] for d in datasets]
    plt.plot(datasets, times, marker='o', label=opt)

plt.xlabel('Dataset')
plt.ylabel('Execution Time (s)')
plt.title('Runtimes by Optimization and Dataset')
plt.legend(title='Optimization')
plt.grid(True)
plt.yscale('log')  # scala logaritmica sui tempi
plt.tight_layout()
plt.savefig('plots/execution_times_plot.png')
plt.close()

# Calcolo dello speedup rispetto alla versione SEQUENTIAL
# Speedup = tempo_sequenziale / tempo_ottimizzato
if 'SEQUENTIAL' in data:
    sequential_times = data['SEQUENTIAL']
    plt.figure(figsize=(10, 6))
    for opt in optimizations:
        if opt == 'SEQUENTIAL':
            continue
        speedups = []
        for d in datasets:
            seq_t = sequential_times[d]
            opt_t = data[opt][d]
            if seq_t is not None and opt_t is not None and opt_t != 0:
                speedups.append(seq_t / opt_t)
            else:
                speedups.append(None)
        plt.plot(datasets, speedups, marker='o', label=opt)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Speedup = 1')
    plt.text(datasets[0], 1.1, 'No Speedup', fontsize=10, color='black')
    plt.xlabel('Dataset')
    plt.ylabel('Speedup (Sequential / Optimization)')
    plt.title('Speedup by Dataset and Optimization')
    plt.legend(title='Optimizations')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/speedup_plot.png')
    plt.close()
else:
    print("No SEQUENTIAL data found, skipping speedup plot.")
