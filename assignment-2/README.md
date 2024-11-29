## Istruzioni per la Compilazione e l'Esecuzione

### Compilazione e Esecuzione Base
Per compilare il codice ed eseguirlo:
```bash
make EXT_CFLAGS="-DPOLYBENCH_TIME" clean all run
```

### Compilazione con Dump degli Array
Per fare il dump degli array, usare:
```bash
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```

L'esecuzione successiva produrrà il file di dump con il seguente comando:
```bash
./atax_acc 2> DUMP_{DIMENSIONE_DATASET}_{OTTIMIZZAZIONE}
```

#### Esempio di Dump
Per esempio, per un dataset piccolo con ottimizzazione di riduzione:
```bash
./atax_acc 2> DUMP_SMALL_FOR_REDUCTION
```

### Confronto dei Risultati
Per confrontare i risultati di due dump, usare il comando `diff`:
```bash
diff DUMP_SMALL_SEQ DUMP_SMALL_FOR_REDUCTION
```

### Valdiazione automatica dei risultati
Per validare i risultati di un tipo di ottimizzazione:
```bash
./validate.sh {TIPO_OTTIMIZZAZIONE}
```

## Cambiare la Dimensione del Dataset

Per cambiare la dimensione del dataset, utilizzare:
```bash
make EXT_CFLAGS="-DPOLYBENCH_TIME -D{DIMENSIONE_DATASET}" clean all run
```

#### Esempio
Ad esempio, per usare un dataset ridotto:
```bash
make EXT_CFLAGS="-DPOLYBENCH_TIME -DMINI_DATASET" clean all run
```

## Task da Completare

1. Generare il dump per le varie dimensioni di dataset:
   - `DUMP_SMALL_SEQ`
   - `DUMP_STANDARD_SEQ`
   - `DUMP_LARGE_SEQ`
   - `DUMP_EXTRALARGE_SEQ`
  
2. Decidere quali ottimizzazioni usare e generare i rispettivi dump
3. Controllare con i dump che non ci siano errori logici
4. Raccogliere i tempi di esecuzioni delle varie ottimizzazioni con le varie dimensioni dei dataset
   Quindi per ogni ottimizzazione i tempi di esecuzione con ogni dataset
5. Profilare l'esecuzione sequenziale e le esecuzioni parallelizzate 
   

## Nota Importante

Assicurarsi di **rimuovere l'opzione `-O2` dal Makefile** e sostituirla con `-O0`.  
Il Makefile si trova in:
```
/assignment-HPC/assignment-1/OpenMP/utilities/common.mk
```

# Compilazione semplificata e Test
Per compilare e rilevare il tempo, usare lo script `run.sh` come segue:
```
./run.sh [GRANDEZZA_DATASET] [OTTIMIZZAZIONE]
```

Per esempio:
```
./run.sh SMALL PARALLEL
```
