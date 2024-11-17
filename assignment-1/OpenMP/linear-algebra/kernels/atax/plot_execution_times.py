#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
# Define the order of datasets and optimizations
datasets_order = ['MINI', 'SMALL', 'STANDARD', 'LARGE', 'EXTRALARGE']
optimizations_order = ['SEQUENTIAL', 'PARALLEL', 'REDUCTION', 'COLLAPSE', 'TARGET']

# Initialize data structures
data = {}  # data[optimization][dataset] = execution_time
optimizations = set()
datasets = set()

# Read the execution_times.txt file
with open('./execution_times.txt', 'r') as f:
    lines = f.readlines()

# Skip the header
header = lines[0].strip()
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 3:
        continue  # Skip invalid lines
    dataset, optimization, execution_time = parts
    datasets.add(dataset)
    optimizations.add(optimization)
    if execution_time == 'ERROR':
        execution_time = None
    else:
        execution_time = float(execution_time)
        if execution_time == 0:
            execution_time = None  # Treat zero as invalid data
    if optimization not in data:
        data[optimization] = {}
    data[optimization][dataset] = execution_time

# Generate the overall runtime plot
plt.figure(figsize=(10, 6))

for optimization in optimizations_order:
    times = []
    for dataset in datasets_order:
        time = data.get(optimization, {}).get(dataset, None)
        times.append(time)
    plt.plot(datasets_order, times, marker='o', label=optimization)

plt.xlabel('Dataset')
plt.ylabel('Execution Time (s)')
plt.title('Runtimes by Optimization and Dataset')
plt.legend(title='Optimizations')
plt.grid(True)
plt.yscale('log')  # Use a logarithmic scale for Y-axis
plt.tight_layout()
plt.savefig('execution_times_plot.png')

# Create the directory for dataset-specific plots if it doesn't exist
os.makedirs('dataset_plots', exist_ok=True)

# Define a color map for optimizations
optimization_colors = {
    'SEQUENTIAL': 'blue',
    'PARALLEL': 'orange',
    'REDUCTION': 'green',
    'COLLAPSE': 'red',
    'TARGET': 'purple'
}

# Generate a bar chart for each dataset
for dataset in datasets_order:
    times = []
    colors = []
    optimizations = []
    for optimization in optimizations_order:
        time = data.get(optimization, {}).get(dataset, None)
        if time is not None:
            times.append(time)
            optimizations.append(optimization)
            # Get the color for the current optimization
            color = optimization_colors.get(optimization, 'gray')
            colors.append(color)
        else:
            print(f"Warning: No data for {optimization} on {dataset}")

    if not times:
        print(f"No data available for dataset {dataset}.")
        continue

    plt.figure(figsize=(8, 6))
    plt.bar(optimizations, times, color=colors)
    plt.xlabel('Optimization')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Times for {dataset}')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.yscale('log')  # Use a logarithmic scale for Y-axis

    # Add values above the bars
    for i, time in enumerate(times):
        plt.text(i, time, f'{time:.2e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Save the plot in the dataset_plots folder
    plt.savefig(f'dataset_plots/execution_times_{dataset}.png')
    plt.close()
    
# Generate the speedup plot for all optimizations and datasets
plt.figure(figsize=(10, 6))

for optimization in optimizations_order:
    if optimization == 'SEQUENTIAL':
        continue  # Skip sequential optimization
    speedups = []
    for dataset in datasets_order:
        sequential_time = data.get('SEQUENTIAL', {}).get(dataset, None)
        opt_time = data.get(optimization, {}).get(dataset, None)
        if sequential_time is not None and opt_time is not None:
            speedups.append(sequential_time / opt_time)
        else:
            speedups.append(None)  # Missing value

    # Add the line for the optimization
    plt.plot(
        datasets_order,
        speedups,
        marker='o',
        label=optimization,
        color=optimization_colors.get(optimization, 'gray')
    )

# Graph configuration
plt.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Speedup = 1')  # Horizontal line at 1
plt.text(datasets_order[0], 1.1, 'No Speedup', fontsize=10, color='black')  # Label above the line

plt.xlabel('Dataset')
plt.ylabel('Speedup (Sequential / Optimization)')
plt.title('Speedup by Dataset and Optimization')
plt.legend(title='Optimizations')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot
os.makedirs('speedup_plots', exist_ok=True)
plt.savefig('speedup_plots/speedup_all_datasets.png', dpi=300)
