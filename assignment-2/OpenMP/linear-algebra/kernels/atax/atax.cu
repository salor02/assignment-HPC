#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "atax.h"

#include <omp.h>
/* Funzione di inizializzazione degli array. */
static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), // Matrice A di dimensione nx x ny
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))         // Vettore x di dimensione ny
{
  int i, j;

  // Inizializza il vettore x con valori che vanno da 0 a (ny-1) moltiplicati per PI
  for (i = 0; i < ny; i++)
    x[i] = i * M_PI; // Ogni elemento di x è il suo indice moltiplicato per PI

  // Inizializza la matrice A con valori calcolati tramite la formula (i * (j +  1)) / nx
  // i: indice di riga, j: indice di colonna
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx; // Imposta A[i][j] come un valore normalizzato
}

/* Funzione per stampare l'array y (output del calcolo). */
static void print_array(int nx,
                        DATA_TYPE POLYBENCH_1D(y, NX, nx)) // Vettore y di dimensione nx
{
  int i;

  // Stampa i valori dell'array y, uno alla volta
  for (i = 0; i < nx; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]); // Stampa un valore di y usando un formato specifico
    if (i % 20 == 0)                             // Aggiunge una nuova riga ogni 20 valori per migliorare la leggibilità
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}


//+++++++++++++++++++++++++++++ KERNELS CUDA +++++++++++++++++++++++++++++
//_______________________________________________________ SOLUZIONE BASE
// Primo kernel: Calcola tmp[i] = A[i][:] * x
__global__ void compute_tmp(const float *A, const float *x, float *tmp) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Ogni thread processa una riga
    if (row < NX) {
        float sum = 0.0;
        for (int j = 0; j < NY; j++) {
            sum += A[row * NY + j] * x[j];
        }
        tmp[row] = sum;
    }
}

// Secondo kernel: Calcola y[j] += A[i][j] * tmp[i]
__global__ void compute_y(const float *A, const float *tmp, float *y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Ogni thread processa una colonna
    if (col < NY) {
        float sum = 0.0;
        for (int i = 0; i < NX; i++) {
            sum += A[i * NY + col] * tmp[i];
        }
        y[col] = sum; // Accumula i risultati in parallelo
    }
}

//_______________________________________________________ SOLUZIONE CON SHARED MEMORY
__global__ void compute_tmp_shared(const float *A, const float *x, float *tmp) {
    __shared__ float x_shared[BLOCK_SIZE]; // Memoria condivisa per x

    int row = blockIdx.x * blockDim.x + threadIdx.x; // Riga processata dal thread
    int tx = threadIdx.x;                           // Indice thread nel blocco

    float sum = 0.0;

    // Carica x nella shared memory, in blocchi
    for (int tile = 0; tile < (NY + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        if (tile * BLOCK_SIZE + tx < NY) {
            x_shared[tx] = x[tile * BLOCK_SIZE + tx];
        } else {
            x_shared[tx] = 0.0; // Fuori dai limiti di x
        }
        __syncthreads(); // Assicura che tutti i thread abbiano caricato x

        // Usa x_shared per calcolare tmp[row]
        for (int j = 0; j < BLOCK_SIZE && tile * BLOCK_SIZE + j < NY; ++j) {
            sum += A[row * NY + tile * BLOCK_SIZE + j] * x_shared[j];
        }

        __syncthreads(); // Prima di ricaricare la shared memory
    }

    if (row < NX) {
        tmp[row] = sum;
    }
}

__global__ void compute_y_shared(const float *A, const float *tmp, float *y) {
    __shared__ float tmp_shared[BLOCK_SIZE]; // Shared memory per tmp

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonna processata dal thread
    int tx = threadIdx.x;                           // Indice thread nel blocco

    float sum = 0.0;

    // Calcolo per blocchi di tmp
    for (int tile = 0; tile < (NX + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Carica un tile di tmp in memoria condivisa
        if (tile * BLOCK_SIZE + tx < NX) {
            tmp_shared[tx] = tmp[tile * BLOCK_SIZE + tx];
        } else {
            tmp_shared[tx] = 0.0; // Padding se fuori dai limiti
        }
        __syncthreads(); // Assicura che tutti i thread abbiano caricato il tile

        // Usa il tile di tmp_shared per calcolare y[col]
        if (col < NY) {
            for (int i = 0; i < BLOCK_SIZE && tile * BLOCK_SIZE + i < NX; ++i) {
                sum += A[(tile * BLOCK_SIZE + i) * NY + col] * tmp_shared[i];
            }
        }

        __syncthreads(); // Prima di ricaricare la shared memory
    }

    // Scrittura del risultato
    if (col < NY) {
        y[col] = sum;
    }
}

//_______________________________________________________ SOLUZIONE CON SHARED MEMORY E A TRASPOSTA

// La commento perché è stata già definita sopra
// __global__ void compute_tmp_shared(const float *A, const float *x, float *tmp) {
//     __shared__ float x_shared[BLOCK_SIZE]; // Memoria condivisa per x

//     int row = blockIdx.x * blockDim.x + threadIdx.x; // Riga processata dal thread
//     int tx = threadIdx.x;                           // Indice thread nel blocco

//     float sum = 0.0;

//     // Carica x nella shared memory, in blocchi
//     for (int tile = 0; tile < (NY + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
//         if (tile * BLOCK_SIZE + tx < NY) {
//             x_shared[tx] = x[tile * BLOCK_SIZE + tx];
//         }
//         else {
//             x_shared[tx] = 0.0; // Fuori dai limiti di x
//         }
//         __syncthreads(); // Assicura che tutti i thread abbiano caricato x

//         // Usa x_shared per calcolare tmp[row]
//         for (int j = 0; j < BLOCK_SIZE && tile * BLOCK_SIZE + j < NY; ++j) {
//             sum += A[row * NY + tile * BLOCK_SIZE + j] * x_shared[j];
//         }

//         __syncthreads(); // Prima di ricaricare la shared memory
//     }

//     if (row < NX) {
//         tmp[row] = sum;
//     }
// }

__global__ void transpose_matrix(const float *A, float *A_T) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < NX && col < NY) {
        A_T[col * NX + row] = A[row * NY + col]; // Trasporre gli elementi
    }
}

__global__ void compute_y_transposed(const float *A_T, const float *tmp, float *y) {
    __shared__ float tmp_shared[BLOCK_SIZE]; // Memoria condivisa per tmp

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonna processata dal thread
    int tx = threadIdx.x;                           // Indice thread nel blocco

    float sum = 0.0;

    // Calcolo per blocchi di tmp
    for (int tile = 0; tile < (NX + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Carica un tile di tmp in memoria condivisa
        if (tile * BLOCK_SIZE + tx < NX) {
            tmp_shared[tx] = tmp[tile * BLOCK_SIZE + tx];
        } else {
            tmp_shared[tx] = 0.0; // Padding se fuori dai limiti
        }
        __syncthreads(); // Assicura che tutti i thread abbiano caricato il tile

        // Usa il tile di tmp_shared per calcolare y[col]
        if (col < NY) {
            for (int i = 0; i < BLOCK_SIZE && tile * BLOCK_SIZE + i < NX; ++i) {
                sum += A_T[col * NX + (tile * BLOCK_SIZE + i)] * tmp_shared[i];
            }
        }

        __syncthreads(); // Prima di ricaricare la shared memory
    }

    // Scrittura del risultato
    if (col < NY) {
        y[col] = sum;
    }
}


/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. 
    VERSIONE CUDA
 */

void kernel_atax_cuda(  DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), // Matrice A di dimensione nx x ny
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),         // Vettore x di dimensione ny
                        DATA_TYPE POLYBENCH_1D(y, NY, ny))         // Vettore y di dimensione ny (output)
  {
    // Allocazione memoria sulla GPU
    float *d_A, *d_x, *d_tmp, *d_y;
    cudaMalloc(&d_A, NX * NY * sizeof(float));
    cudaMalloc(&d_x, NY * sizeof(float));
    cudaMalloc(&d_tmp, NX * sizeof(float));
    cudaMalloc(&d_y, NY * sizeof(float));

    // Copia dati dalla CPU alla GPU
    cudaMemcpy(d_A, A, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, NY * sizeof(float)); // Inizializza y a zero, senza bisogno di copiarlo

    // Lancia il primo kernel
    int grid_size_tmp = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_tmp<<<grid_size_tmp, BLOCK_SIZE>>>(d_A, d_x, d_tmp);

    // Lancia il secondo kernel
    int grid_size_y = (NY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_y<<<grid_size_y, BLOCK_SIZE>>>(d_A, d_tmp, d_y);

    // Copia il risultato finale dalla GPU alla CPU
    cudaMemcpy(y, d_y, NY * sizeof(float), cudaMemcpyDeviceToHost);

    // Libera memoria
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_tmp);
    cudaFree(d_y);
}

/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. 
    VERSIONE SEQUENZIALE
 */
static void kernel_atax_seq(int nx, int ny,
                        DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), // Matrice A di dimensione nx x ny
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),         // Vettore x di dimensione ny
                        DATA_TYPE POLYBENCH_1D(y, NY, ny),         // Vettore y di dimensione ny (output)
                        DATA_TYPE POLYBENCH_1D(tmp, NX, nx))       // Vettore temporaneo tmp di dimensione nx
{
  int i, j;
  // Inizializza l'array y a zero
  for (i = 0; i < _PB_NY; i++)
    y[i] = 0;

  // Calcola il prodotto matrice-vettore: A * x
  // tmp[i] conterrà il risultato della moltiplicazione della riga i della matrice A per il vettore x
  for (i = 0; i < _PB_NX; i++) // Ciclo sulle righe della matrice A
  {
    tmp[i] = 0;                         // Inizializza tmp[i] a zero
    for (j = 0; j < _PB_NY; j++)        // Ciclo sulle colonne della matrice A (lunghezza di x)
      tmp[i] = tmp[i] + A[i][j] * x[j]; // Somma il prodotto A[i][j] * x[j] nel vettore tmp[i]

    // Ora aggiorna il vettore y con il risultato della moltiplicazione riga di A * tmp
    for (j = 0; j < _PB_NY; j++)
      y[j] = y[j] + A[i][j] * tmp[i]; // Somma il prodotto A[i][j] * tmp[i] nel vettore y
  }
}

int main(int argc, char **argv)
{
  /* Recupera la dimensione del problema (in questo caso, NX e NY sono predefiniti). */
  int nx = NX; // Numero di righe della matrice A
  int ny = NY; // Numero di colonne della matrice A

  /* Dichiarazione e allocazione degli array necessari. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny); // Matrice A
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);         // Vettore x
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);         // Vettore y (output)
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);       // Vettore temporaneo tmp

  /* Inizializza gli array con i dati appropriati. */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Avvia il timer per misurare il tempo di esecuzione del calcolo. */
  polybench_start_instruments;

  /* Chiamata alla funzione principale di calcolo sequenziale.  
  kernel_atax_seq(nx, ny,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(tmp));
    */

   // Chiamata alla funzione principale di calcolo parallelo
    kernel_atax_cuda( POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y) // Questo va inserito per riottenere il risultato
              // Invece tmp non è necessario, perché può crearlo direttamente la funzione
         );

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