#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Definisci l'ordine dei dataset e delle ottimizzazioni
datasets_order = ['MINI', 'SMALL', 'STANDARD', 'LARGE', 'EXTRALARGE']
optimizations_order = ['SEQUENTIAL', 'PARALLEL', 'REDUCTION', 'COLLAPSE', 'TASK', 'TARGET']

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
plt.ylabel('Tempo di Esecuzione (s)')
plt.title('Tempi di Esecuzione per Ottimizzazione e Dataset')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Usa una scala logaritmica per l'asse Y
plt.tight_layout()
plt.savefig('execution_times_plot.png')

