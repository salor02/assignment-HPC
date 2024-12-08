import re
from collections import defaultdict

# File di input (output grezzo dal profiling)
RAW_OUTPUT = "profiling_raw_output.txt"
# File di output (risultati finali)
OUTPUT_FILE = "profiling_results.txt"

# Struttura per memorizzare i risultati
results = defaultdict(lambda: defaultdict(lambda: {"runtime": None, "gpu_timings": defaultdict(list)}))

# Pattern per estrarre runtime totale e GPU activities
runtime_pattern = re.compile(r"^(\d+\.\d+)$")
gpu_pattern = re.compile(r"^\s*\d+\.\d+%\s+(\d+\.\d+)ms\s+\d+\s+.+?\s+(.*)$")

# Legge il file di output grezzo
with open(RAW_OUTPUT, "r") as f:
    current_optimization = None
    current_dataset = None
    for line in f:
        line = line.strip()
        if line.startswith("OPTIMIZATION:"):
            current_optimization = line.split(":")[1].strip()
        elif line.startswith("DATASET:"):
            current_dataset = line.split(":")[1].strip()
        elif runtime_pattern.match(line):
            # Runtime totale
            results[current_optimization][current_dataset]["runtime"] = float(runtime_pattern.match(line).group(1))
        elif gpu_pattern.match(line):
            # Tempi GPU classificati
            match = gpu_pattern.match(line)
            time_ms = float(match.group(1))
            operation = match.group(2).strip()
            results[current_optimization][current_dataset]["gpu_timings"][operation].append(time_ms / 1000)  # Converti in secondi

# Scrive i risultati finali
with open(OUTPUT_FILE, "w") as f:
    for optimization, datasets in results.items():
        f.write(f"{optimization}\n")
        for dataset, data in datasets.items():
            runtime = data["runtime"] if data["runtime"] is not None else "N/A"
            f.write(f"{dataset:10}: {runtime:.6f}\n")
            f.write("GPU Timings:\n")
            for operation, times in data["gpu_timings"].items():
                avg_time = sum(times) / len(times) if times else 0
                f.write(f"  {operation:30}: {avg_time:.6f} s\n")
        f.write("______________________________________\n")

print(f"Risultati finali salvati in {OUTPUT_FILE}.")