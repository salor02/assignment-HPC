#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "atax.h"

#include <omp.h>
/* Funzione di inizializzazione degli array. */
static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),  // Matrice A di dimensione nx x ny
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))           // Vettore x di dimensione ny
{
  int i, j;

  // Inizializza il vettore x con valori che vanno da 0 a (ny-1) moltiplicati per PI
  for (i = 0; i < ny; i++)
    x[i] = i * M_PI;  // Ogni elemento di x è il suo indice moltiplicato per PI

  // Inizializza la matrice A con valori calcolati tramite la formula (i * (j + 1)) / nx
  // i: indice di riga, j: indice di colonna
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;  // Imposta A[i][j] come un valore normalizzato
}

/* Funzione per stampare l'array y (output del calcolo). */
static void print_array(int nx,
                        DATA_TYPE POLYBENCH_1D(y, NX, nx))  // Vettore y di dimensione nx
{
  int i;

  // Stampa i valori dell'array y, uno alla volta
  for (i = 0; i < nx; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);  // Stampa un valore di y usando un formato specifico
    if (i % 20 == 0)  // Aggiunge una nuova riga ogni 20 valori per migliorare la leggibilità
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. */
static void kernel_atax(int nx, int ny,
                        DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),  // Matrice A di dimensione nx x ny
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),          // Vettore x di dimensione ny
                        DATA_TYPE POLYBENCH_1D(y, NY, ny),          // Vettore y di dimensione ny (output)
                        DATA_TYPE POLYBENCH_1D(tmp, NX, nx))        // Vettore temporaneo tmp di dimensione nx
{
  int i, j;
  //___________SOLUZIONE SEQUENZIALE___________
  int i, j;

  #if defined SEQUENTIAL

  // Inizializza l'array y a zero
  for (i = 0; i < _PB_NY; i++)
    y[i] = 0;

  // Calcola il prodotto matrice-vettore: A * x
  // tmp[i] conterrà il risultato della moltiplicazione della riga i della matrice A per il vettore x
  for (i = 0; i < _PB_NX; i++)  // Ciclo sulle righe della matrice A
  {
    tmp[i] = 0;  // Inizializza tmp[i] a zero
    for (j = 0; j < _PB_NY; j++)  // Ciclo sulle colonne della matrice A (lunghezza di x)
      tmp[i] = tmp[i] + A[i][j] * x[j];  // Somma il prodotto A[i][j] * x[j] nel vettore tmp[i]

    // Ora aggiorna il vettore y con il risultato della moltiplicazione riga di A * tmp
    for (j = 0; j < _PB_NY; j++)
      y[j] = y[j] + A[i][j] * tmp[i];  // Somma il prodotto A[i][j] * tmp[i] nel vettore y
  }

  // _________SOLUZIONE PARALLEL FOR___________
  #elif defined PARALLEL
  // Inizializza l'array y a zero
  #pragma omp parallel for
  for (i = 0; i < _PB_NY; i++)
    y[i] = 0;

  // Calcolo parallelo di tmp[i]
  #pragma omp parallel for 
  for (i = 0; i < _PB_NX; i++)
  {
    tmp[i] = 0;
    for (j = 0; j < _PB_NY; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
  }

  // Aggiornamento parallelo di y[j] DA NOTARE IL CAMBIO DI VARIABILI D'ITERAZIONE
  #pragma omp parallel for
  for (j = 0; j < _PB_NY; j++)
  {
    for (i = 0; i < _PB_NX; i++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }

  // _________SOLUZIONE PARALLEL FOR + REDUCTION___________
  #elif defined REDUCTION
  // Inizializza l'array y a zero
  #pragma omp parallel for
  for (i = 0; i < _PB_NY; i++)
      y[i] = 0;
      
  #pragma omp parallel for
  for (i = 0; i < _PB_NX; i++)
  {
      double sum = 0;  // Variabile temporanea per accumulare il risultato parziale di tmp[i]
      
      #pragma omp parallel for reduction(+:sum)
      for (j = 0; j < _PB_NY; j++)  // Ciclo sulle colonne della matrice A (lunghezza di x)
      {
          sum += A[i][j] * x[j];
      }
      
      tmp[i] = sum;  // Salva il risultato finale nella posizione corretta
  }

  #pragma omp parallel for
  for (j = 0; j < _PB_NY; j++)  // Ciclo sulle colonne della matrice A
  {
      double sum = 0;  // Variabile temporanea per l'accumulo parziale

      #pragma omp parallel for reduction(+:sum)
      for (i = 0; i < _PB_NX; i++)  // Ciclo sulle righe della matrice A
      {
          sum += A[i][j] * tmp[i];
      }

      y[j] = sum;  // Assegna il risultato finale a y[j]
  }

  // // _________SOLUZIONE PARALLEL FOR + COLLAPSE___________
  #elif defined COLLAPSE
  // Inizializzazione del vettore y
  #pragma omp parallel for
      for (i = 0; i < ny; i++)
          y[i] = 0; 

  // Inizializza tmp a zero prima del ciclo parallelo
  #pragma omp parallel for
      for (i = 0; i < _PB_NX; i++)
          tmp[i] = 0;

  #pragma omp parallel for collapse(2)
  for (i = 0; i < _PB_NX; i++) {
      for (j = 0; j < _PB_NY; j++) {
          tmp[i] += A[i][j] * x[j];
      }
  }

  #pragma omp parallel for collapse(2)
  for (j = 0; j < _PB_NY; j++) {
      for (i = 0; i < _PB_NX; i++) {
          y[j] += A[i][j] * tmp[i];
      }
  }

  // _________SOLUZIONE TASK___________
  #elif defined TASK
  #pragma omp parallel
    {
        // Inizializza y fuori dal loop delle task
        #pragma omp for
        for (i = 0; i < ny; i++)
            y[i] = 0;
 
        // Task per ognii iterazione del ciclo esterno i
        #pragma omp for
        for (i = 0; i < nx; i++)
        {
            #pragma omp task firstprivate(i) shared(A, x, tmp, y)
            {
                tmp[i] = 0;
                for (j = 0; j < ny; j++)
                    tmp[i] += A[i][j] * x[j];
 
                #pragma omp task shared(A, tmp, y) firstprivate(i)
                {
                    for (j = 0; j < ny; j++)
                    {
                        #pragma omp atomic
                        y[j] += A[i][j] * tmp[i];
                    }
                }
            }
        }
    }

  // _________SOLUZIONE TARGET___________
  #elif defined TARGET
  // Inizializzazione del vettore y
  #pragma omp parallel for
          for (i = 0; i < ny; i++)
              y[i] = 0; 
  #pragma omp parallel for
          for (i = 0; i < _PB_NX; i++)
              tmp [i] = 0; 
  // serve ritornare solo y, tmp è solo una variabile temporanea
  #pragma omp target data map(to: A[0:nx][0:ny], x[0:ny], tmp[0:nx]) map(tofrom: y[0:ny])
      { 
          // Prima fase: calcolo di tmp[i] = A[i][j] * x[j]
          #pragma omp target teams distribute parallel for collapse(2)
          for (i = 0; i < nx; i++)
          {
              for (j = 0; j < ny; j++)
                  tmp[i] += A[i][j] * x[j];
          }
  
          // Seconda fase: aggiornamento di y[j] += A[i][j] * tmp[i]
          #pragma omp target teams distribute parallel for collapse(2)
          for (i = 0; i < nx; i++)
          {
              for (j = 0; j < ny; j++)
              {
                  #pragma omp atomic
                  y[j] += A[i][j] * tmp[i]; 
              }
          }
      } 
  #endif

}

int main(int argc, char **argv)
{
  /* Recupera la dimensione del problema (in questo caso, NX e NY sono predefiniti). */
  int nx = NX;  // Numero di righe della matrice A
  int ny = NY;  // Numero di colonne della matrice A

  /* Dichiarazione e allocazione degli array necessari. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);  // Matrice A
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);           // Vettore x
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);           // Vettore y (output)
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);         // Vettore temporaneo tmp

  /* Inizializza gli array con i dati appropriati. */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Avvia il timer per misurare il tempo di esecuzione del calcolo. */
  polybench_start_instruments;

  /* Esegui il kernel ATAx (calcolo principale). */
  kernel_atax(nx, ny,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(tmp));

  /* Ferma il timer e stampa i risultati delle misurazioni. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Previene l'eliminazione del codice morto (DCE) e stampa i risultati finali di y. */
  polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));

  /* Dealloca la memoria per gli array per evitare perdite di memoria. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}
