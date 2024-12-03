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
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

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



__global__ void compute_tmp(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < _PB_NX)
	{
        float sum = 0.0;
        for (int j = 0; j < NY; j++) {
            sum += A[row * NY + j] * x[j];
        }
        tmp[row] = sum;
	}
}

__global__ void compute_y(int nx, int ny, DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (col < _PB_NY)
	{
        float sum = 0.0;
        for (int i = 0; i < NX; i++) {
            sum += A[i * NY + col] * tmp[i];
        }
        y[col] = sum; // Accumula i risultati in parallelo
	}
}

/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. 
    VERSIONE CUDA
 */

void ataxGpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,NY,ny))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	/* Start timer. */
  	polybench_start_instruments;

	atax_kernel1<<< grid1, block >>>(nx, ny, A_gpu,x_gpu,tmp_gpu);
	cudaThreadSynchronize();
	atax_kernel2<<< grid2, block >>>(nx, ny, A_gpu,y_gpu,tmp_gpu);
	cudaThreadSynchronize();
	
	/* Stop and print timer. */
  	polybench_stop_instruments;
    //Stampa di y:
	
	cudaMemcpy(y, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
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
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NX,nx);

  /* Inizializza gli array con i dati appropriati. */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Avvia il timer per misurare il tempo di esecuzione del calcolo. */
  polybench_start_instruments;

	ataxGpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp), 
		POLYBENCH_ARRAY(y_outputFromGpu));

//   kernel_atax_seq(nx, ny,
//               POLYBENCH_ARRAY(A),
//               POLYBENCH_ARRAY(x),
//               POLYBENCH_ARRAY(y),
//               POLYBENCH_ARRAY(tmp));

  /* Ferma il timer e stampa i risultati delle misurazioni. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Previene l'eliminazione del codice morto (DCE) e stampa i risultati finali di y. */
  polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));

  /* Dealloca la memoria per gli array per evitare perdite di memoria. */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(x);
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);
	POLYBENCH_FREE_ARRAY(tmp);


  return 0;
}