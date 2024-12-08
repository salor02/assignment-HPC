#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import re

# Directory di output per i grafici
PLOT_DIR = "aggregated_gpu_plots_per_dataset_static"

# Crea la cartella per i grafici se non esiste
os.makedirs(PLOT_DIR, exist_ok=True)

# Dati Aggregati: data[ottimizzazione][categoria][dataset] = tempo totale in secondi
data = {
    'SEQUENTIAL': {
        'MINI': {
            'Kernel': 0.0,
            'Memory Operations': 0.015151,
            'Synchronization': 0.0
        },
        'SMALL': {
            'Kernel': 0.0,
            'Memory Operations': 0.007421,
            'Synchronization': 0.0
        },
        'STANDARD': {
            'Kernel': 0.0,
            'Memory Operations': 0.270003,
            'Synchronization': 0.0
        },
        'LARGE': {
            'Kernel': 0.0,
            'Memory Operations': 1.055336,
            'Synchronization': 0.0
        }
    },
    'OPTIMIZATION_1': {
        'MINI': {
            'Kernel': 0.000253,
            'Memory Operations': 0.012660,
            'Synchronization': 0.003525
        },
        'SMALL': {
            'Kernel': 0.000599,
            'Memory Operations': 0.017371,
            'Synchronization': 0.005879
        },
        'STANDARD': {
            'Kernel': 0.018794,
            'Memory Operations': 0.281025,
            'Synchronization': 0.342520
        },
        'LARGE': {
            'Kernel': 0.625360,
            'Memory Operations': 0.568758,
            'Synchronization': 0.793120
        }
    },
    'OPTIMIZATION_2': {
        'MINI': {
            'Kernel': 0.000105,
            'Memory Operations': 0.000565,
            'Synchronization': 0.000292
        },
        'SMALL': {
            'Kernel': 0.003331,
            'Memory Operations': 0.004867,
            'Synchronization': 0.006966
        },
        'STANDARD': {
            'Kernel': 0.024592,
            'Memory Operations': 0.273656,
            'Synchronization': 0.183660
        },
        'LARGE': {
            'Kernel': 0.625389,
            'Memory Operations': 0.568758,
            'Synchronization': 0.793120
        }
    }
}

# Definizione delle categorie e dei dataset
categories = ["Kernel", "Memory Operations", "Synchronization"]
datasets = ["MINI", "SMALL", "STANDARD", "LARGE"]

# Colori per i dataset
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']  # Puoi personalizzare i colori

# Funzione per pulire il nome dell'ottimizzazione per il nome del file
def safe_filename(name):
    return re.sub(r'[^\w\s-]', '', name).replace(' ', '_')

# Generazione dei grafici
for optimization, categories_data in data.items():
    # Preparazione dei dati per il grafico
    kernel_times = [categories_data[dataset]['Kernel'] for dataset in datasets]
    memory_times = [categories_data[dataset]['Memory Operations'] for dataset in datasets]
    sync_times = [categories_data[dataset]['Synchronization'] for dataset in datasets]
    
    # Posizioni delle barre
    x = range(len(categories))
    width = 0.2  # Larghezza delle barre
    
    plt.figure(figsize=(10, 6))
    
    # Plot delle barre per ciascun dataset
    for i, dataset in enumerate(datasets):
        # Calcolo delle posizioni per le barre
        pos = [p + i * width for p in x]
        # Estrazione dei tempi per le categorie
        times = [
            categories_data[dataset]['Kernel'],
            categories_data[dataset]['Memory Operations'],
            categories_data[dataset]['Synchronization']
        ]
        plt.bar(pos, times, width, label=dataset, color=colors[i])
    
    # Impostazione delle etichette e del titolo
    plt.xlabel('Categoria Operazione', fontsize=12)
    plt.ylabel('Tempo Totale (s)', fontsize=12)
    plt.title(f'Tempi Aggregati per Dataset e Categoria - {optimization}', fontsize=14)
    plt.xticks([p + 1.5 * width for p in x], categories, fontsize=12)
    plt.yscale('log')  # Scala logaritmica per gestire differenze significative
    plt.legend(title="Dataset")
    plt.tight_layout()
    
    # Salvataggio del grafico
    filename = f"{safe_filename(optimization)}_aggregated_gpu_timings_per_dataset.png"
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

print(f"Grafici aggregati per dataset generati nella cartella '{PLOT_DIR}'.")