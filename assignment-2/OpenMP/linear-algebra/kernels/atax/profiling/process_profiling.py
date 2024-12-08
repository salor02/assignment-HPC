import re
from collections import defaultdict

# File di input (output grezzo dal profiling)
RAW_OUTPUT = "profiling_raw_output.txt"
# File di output (risultati finali)
OUTPUT_FILE = "profiling_results.txt"

# Struttura per memorizzare i risultati
results = defaultdict(lambda: defaultdict(lambda: {"runtime": None, "gpu_timings": defaultdict(list)}))

# Pattern per estrarre runtime totale
runtime_pattern = re.compile(r"^\d+\.\d+$")

# Pattern per estrarre GPU activities e API calls
gpu_pattern = re.compile(
    r"^\s*\d+\.\d+%\s+([\d.]+)(ns|us|ms|s)\s+\d+\s+.*?(?:\[(.+?)\]|(.+?))$"
)

# Funzione per convertire il tempo in secondi
def convert_to_seconds(value, unit):
    if unit == 'ns':
        return float(value) * 1e-9
    elif unit == 'us':
        return float(value) * 1e-6
    elif unit == 'ms':
        return float(value) * 1e-3
    elif unit == 's':
        return float(value)
    else:
        return 0.0  # Unità sconosciuta

# Legge il file di output grezzo
with open(RAW_OUTPUT, "r") as f:
    current_optimization = None
    current_dataset = None
    for line in f:
        line = line.strip()
        if line.startswith("OPTIMIZATION:"):
            current_optimization = line.split(":", 1)[1].strip()
        elif line.startswith("DATASET:"):
            current_dataset = line.split(":", 1)[1].strip()
        elif runtime_pattern.match(line):
            # Runtime totale
            results[current_optimization][current_dataset]["runtime"] = float(line)
        else:
            match = gpu_pattern.match(line)
            if match:
                time_value, unit, op_bracket, op_no_bracket = match.groups()
                operation = op_bracket if op_bracket else op_no_bracket
                time_sec = convert_to_seconds(time_value, unit)
                results[current_optimization][current_dataset]["gpu_timings"][operation].append(time_sec)

# Scrive i risultati finali
with open(OUTPUT_FILE, "w") as f:
    for optimization, datasets in results.items():
        f.write(f"{optimization}\n")
        for dataset, data in datasets.items():
            runtime = data["runtime"] if data["runtime"] is not None else "N/A"
            if isinstance(runtime, float):
                f.write(f"{dataset:10}: {runtime:.6f} s\n")
            else:
                f.write(f"{dataset:10}: {runtime}\n")
            f.write("GPU Timings:\n")
            if data["gpu_timings"]:
                for operation, times in data["gpu_timings"].items():
                    avg_time = sum(times) / len(times) if times else 0
                    f.write(f"  {operation:50}: {avg_time:.6f} s\n")
            else:
                f.write("  N/A\n")
        f.write("______________________________________\n")

print(f"Risultati finali salvati in {OUTPUT_FILE}.")