## Istruzioni per la Compilazione e l'Esecuzione

### Compilazione e Esecuzione Base
Per compilare il codice ed eseguirlo:
```bash
 make EXERCISE=atax.cu DATASET_TYPE=DIMENSIONE_DATASET OPTIMIZATION=OPTIMIZATION_N clean run
```
Ad esempio: 
```bash
 make EXERCISE=atax.cu DATASET_TYPE=STANDARD_DATASET OPTIMIZATION=OPTIMIZATION_2 clean run
```
Per profilare l'esecuzione:
```bash
 make EXERCISE=atax.cu DATASET_TYPE=DIMENSIONE_DATASET OPTIMIZATION=OPTIMIZATION_N clean profile
```

### Compilazione con Dump degli Array
Per fare il dump degli array, usare:
```bash
make EXERCISE=atax.cu DATASET_TYPE=DIMENSIONE_DATASET DUMP_ARRAYS=1 OPTIMIZATION=OPTIMIZATION_N clean run
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


## Task da Completare

1. Modificare gli script per:
   - dump.sh
   - validate.sh
   - run.sh   

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
