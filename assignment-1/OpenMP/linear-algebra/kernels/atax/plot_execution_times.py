#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
# Definisci l'ordine dei dataset e delle ottimizzazioni
datasets_order = ['MINI', 'SMALL', 'STANDARD', 'LARGE', 'EXTRALARGE']
optimizations_order = ['SEQUENTIAL', 'PARALLEL', 'REDUCTION', 'COLLAPSE', 'TARGET']

# Inizializza le strutture dati
data = {}  # data[optimization][dataset] = execution_time
optimizations = set()
datasets = set()

# Leggi il file execution_times.txt
with open('./execution_times.txt', 'r') as f:
    lines = f.readlines()

# Salta l'intestazione
header = lines[0].strip()
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 3:
        continue  # Salta le linee non valide
    dataset, optimization, execution_time = parts
    datasets.add(dataset)
    optimizations.add(optimization)
    if execution_time == 'ERROR':
        execution_time = None
    else:
        execution_time = float(execution_time)
        if execution_time == 0:
            execution_time = None  # Considera zero come dato non valido
    if optimization not in data:
        data[optimization] = {}
    data[optimization][dataset] = execution_time

# Genera il grafico
plt.figure(figsize=(10, 6))

for optimization in optimizations_order:
    times = []
    for dataset in datasets_order:
        time = data.get(optimization, {}).get(dataset, None)
        times.append(time)
    plt.plot(datasets_order, times, marker='o', label=optimization)

plt.xlabel('Dataset')
plt.ylabel('Execution Time (s)')
plt.title('Runtimes for Optimization and Datasets')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Usa una scala logaritmica per l'asse Y
plt.tight_layout()
plt.savefig('execution_times_plot.png')

# Crea la directory per i grafici dei dataset se non esiste
os.makedirs('dataset_plots', exist_ok=True)

# Definisci una mappa di colori per le ottimizzazioni
optimization_colors = {
    'SEQUENTIAL': 'blue',
    'PARALLEL': 'orange',
    'REDUCTION': 'green',
    'COLLAPSE': 'red',
    'TARGET': 'purple'
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
    plt.xlabel('Optimization')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Times for {dataset}')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.yscale('log')  # Usa una scala logaritmica per l'asse Y

    # Aggiungi i valori sopra le barre
    for i, time in enumerate(times):
        plt.text(i, time, f'{time:.2e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Salva il grafico nella cartella dataset_plots
    plt.savefig(f'dataset_plots/execution_times_{dataset}.png')
    plt.close()
