#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np

# Definisci l'ordine dei dataset e delle ottimizzazioni
datasets_order = ['MINI', 'SMALL', 'STANDARD', 'LARGE', 'EXTRALARGE']
optimizations_order = ['SEQUENTIAL', 'PARALLEL', 'REDUCTION', 'COLLAPSE', 'TARGET']

# Inizializza le strutture dati
data = {}  # data[optimization][dataset] = execution_time

# Leggi il file execution_times.txt
with open('./execution_times.txt', 'r') as f:
    lines = f.readlines()

# Salta l'intestazione
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 3:
        continue  # Salta le linee non valide
    dataset, optimization, execution_time = parts
    if execution_time == 'ERROR':
        execution_time = None
    else:
        try:
            execution_time = float(execution_time)
            if execution_time == 0:
                execution_time = None  # Considera zero come dato non valido
        except ValueError:
            execution_time = None
    if optimization not in data:
        data[optimization] = {}
    data[optimization][dataset] = execution_time

# Genera il grafico generale
plt.figure(figsize=(10, 6))

for optimization in optimizations_order:
    times = []
    for dataset in datasets_order:
        time = data.get(optimization, {}).get(dataset, None)
        times.append(time)
    plt.plot(datasets_order, times, marker='o', label=optimization)

plt.xlabel('Dataset')
plt.ylabel('Tempo di Esecuzione (s)')
plt.title('Tempi di Esecuzione per Ottimizzazione e Dataset')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Usa una scala logaritmica per l'asse Y
plt.tight_layout()
plt.savefig('execution_times_plot.png', dpi=300)
plt.close()

# Crea la directory per i grafici dei dataset se non esiste
os.makedirs('dataset_plots', exist_ok=True)

# Definisci una mappa di colori per le ottimizzazioni
optimization_colors = {
    'SEQ': 'blue',
    'PARALLEL_FOR': 'green',
    'REDUCTION': 'orange',
    'COLLAPSE': 'red',
    'TASK': 'purple',
    'TARGET': 'brown'
}

# Genera un bar chart per ogni dataset
for dataset in datasets_order:
    times = []
    colors = []
    optimizations = []
    for optimization in optimizations_order:
        time = data.get(optimization, {}).get(dataset, None)
        if time is not None:
            times.append(time)
            optimizations.append(optimization)
            # Ottieni il colore per l'ottimizzazione corrente
            color = optimization_colors.get(optimization, 'gray')
            colors.append(color)
        else:
            print(f"Attenzione: Nessun dato per {optimization} su {dataset}")

    if not times:
        print(f"Nessun dato disponibile per il dataset {dataset}.")
        continue

    plt.figure(figsize=(8, 6))
    plt.bar(optimizations, times, color=colors)
    plt.xlabel('Ottimizzazione')
    plt.ylabel('Tempo di Esecuzione (s)')
    plt.title(f'Tempi di Esecuzione per {dataset}')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.yscale('log')  # Usa una scala logaritmica per l'asse Y

    # Aggiungi i valori sopra le barre
    for i, time in enumerate(times):
        plt.text(i, time, f'{time:.2e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Salva il grafico nella cartella dataset_plots
    plt.savefig(f'dataset_plots/execution_times_{dataset}.png', dpi=300)
    plt.close()
    
    # Crea la directory per i grafici dello speedup se non esiste
os.makedirs('speedup_plots', exist_ok=True)

# Genera lo speedup per ogni dataset rispetto a SEQ
for dataset in datasets_order:
    sequential_time = data.get('SEQUENTIAL', {}).get(dataset, None)
    if sequential_time is None:
        print(f"Attenzione: Tempo sequenziale non disponibile per il dataset {dataset}.")
        continue

    speedups = []
    optimizations = []
    colors = []

    for optimization in optimizations_order:
        if optimization == 'SEQUENTIAL':
            continue  # Salta l'ottimizzazione sequenziale
        opt_time = data.get(optimization, {}).get(dataset, None)
        if opt_time is not None and sequential_time is not None:
            speedup = sequential_time / opt_time
            speedups.append(speedup)
            optimizations.append(optimization)
            # Colori per le ottimizzazioni
            colors.append(optimization_colors.get(optimization, 'gray'))
        else:
            print(f"Attenzione: Nessun dato per {optimization} su {dataset}.")

    if not speedups:
        print(f"Nessun dato disponibile per il calcolo dello speedup del dataset {dataset}.")
        continue

    plt.figure(figsize=(8, 6))
    plt.bar(optimizations, speedups, color=colors)
    plt.xlabel('Ottimizzazione')
    plt.ylabel('Speedup (Seq / Ottimizzazione)')
    plt.title(f'Speedup per {dataset}')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # Aggiungi i valori sopra le barre
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup, f'{speedup:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Salva il grafico nella cartella speedup_plots
    plt.savefig(f'speedup_plots/speedup_{dataset}.png', dpi=300)
    plt.close()
