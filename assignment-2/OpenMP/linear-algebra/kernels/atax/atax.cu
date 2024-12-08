#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "atax.h"

#include <omp.h>
#define BLOCK_SIZE_X 512
#define BLOCK_SIZE_Y 1
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5
#define EPSILON 1e-6 // per gestire divisori piccoli

float percentDiff(float a, float b) {
    if ((fabs(a) < EPSILON) && (fabs(b) < EPSILON)) {
        return 0.0f; // Entrambi gli elementi sono vicini a zero
    }
    return fabs((a - b) / ((a + b) / 2.0f)) * 100.0f;
}


void compareResults(int ny, DATA_TYPE POLYBENCH_1D(z_CPU,NY,ny), DATA_TYPE POLYBENCH_1D(z_GPU,NY,ny))
{
	int i, fail;
	fail = 0;

	for (i=0; i<ny; i++)
	{
		if (percentDiff(z_CPU[i], z_GPU[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
if (fail > 0) {
    printf("Mannaggia alla miseria ci sono %d discrepanze tra i risultati della CPU e della GPU che superano la soglia di errore di %.2f%%.\n", fail, PERCENT_DIFF_ERROR_THRESHOLD);
} else {
    printf("Non ci sono errori :-)\n", PERCENT_DIFF_ERROR_THRESHOLD);
}
}

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

//_______________________________________________________ SOLUZIONE CON SHARED MEMORY
__global__ void compute_tmp_shared(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp) {
    __shared__ DATA_TYPE x_shared[BLOCK_SIZE_X]; // Shared memory per un tile di x

    int row = blockIdx.x * blockDim.x + threadIdx.x; // Riga processata dal thread
    int tx = threadIdx.x;                           // Indice thread nel blocco

    float sum = 0.0;

    // Itera sui tile di x
    for (int tile = 0; tile < (ny + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; ++tile) {
        // Carica un tile di x nella memoria condivisa
        int col = tile * BLOCK_SIZE_X + tx;
        if (col < ny) {
            x_shared[tx] = x[col];
        } else {
            x_shared[tx] = 0.0; // Padding per evitare accessi fuori dai limiti
        }
        __syncthreads(); // Assicura che tutti i thread abbiano caricato il tile

        // Usa il tile di x_shared per calcolare tmp[row]
        if (row < nx) {
            for (int j = 0; j < BLOCK_SIZE_X && tile * BLOCK_SIZE_X + j < ny; ++j) {
                sum += A[row * ny + tile * BLOCK_SIZE_X + j] * x_shared[j];
            }
        }

        __syncthreads(); // Prima di caricare il prossimo tile
    }

    // Scrittura del risultato
    if (row < nx) {
        tmp[row] = sum;
    }
}


__global__ void compute_y_shared(int nx, int ny, DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp) {
    __shared__ float tmp_shared[BLOCK_SIZE_X]; // Shared memory per un tile di tmp

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonna processata dal thread
    int tx = threadIdx.x;                           // Indice thread nel blocco

    float sum = 0.0;

    // Itera sui tile di tmp
    for (int tile = 0; tile < (nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; ++tile) {
        // Carica un tile di tmp nella memoria condivisa
        int row = tile * BLOCK_SIZE_X + tx;
        if (row < nx) {
            tmp_shared[tx] = tmp[row];
        } else {
            tmp_shared[tx] = 0.0; // Padding per evitare accessi fuori dai limiti
        }
        __syncthreads(); // Assicura che tutti i thread abbiano caricato il tile

        // Usa il tile di tmp_shared per calcolare y[col]
        if (col < ny) {
            for (int i = 0; i < BLOCK_SIZE_X && tile * BLOCK_SIZE_X + i < nx; ++i) {
                sum += A[(tile * BLOCK_SIZE_X + i) * ny + col] * tmp_shared[i];
            }
        }

        __syncthreads(); // Prima di caricare il prossimo tile
    }

    // Scrittura del risultato
    if (col < ny) {
        y[col] = sum;
    }
}
/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. 
    VERSIONE CUDA
 */

void kernel_atax(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
	DATA_TYPE *d_A;
	DATA_TYPE *d_x;
	DATA_TYPE *d_y;
	DATA_TYPE *d_tmp;

	cudaMalloc((void **)&d_A, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&d_x, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&d_y, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&d_tmp, sizeof(DATA_TYPE) * NX);
	
	cudaMemcpy(d_A, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
	
	//In teoria potremmo fare così: 
	// cudaMemset (d_y, 0, sizeof(DATA_TYPE) * NY);
	// cudaMemset (d_tmp, 0, sizeof(DATA_TYPE) * NX);

	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

    #if defined OPTIMIZATION_1
    compute_tmp<<< grid1, block >>>(nx, ny, d_A,d_x,d_tmp);
    cudaThreadSynchronize();
    compute_y<<< grid2, block >>>(nx, ny, d_A,d_y,d_tmp);
    cudaThreadSynchronize();

    #elif defined OPTIMIZATION_2
    compute_tmp_shared<<< grid1, block >>>(nx, ny, d_A,d_x,d_tmp);
    cudaThreadSynchronize();
    compute_y_shared<<< grid2, block >>>(nx, ny, d_A,d_y,d_tmp);
    cudaThreadSynchronize();

    #endif

	
	
	cudaMemcpy(y, d_y, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_tmp);
}


/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. 
    VERSIONE SEQUENZIALE
 */
static void sequential_atax(int nx, int ny,
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
	POLYBENCH_1D_ARRAY_DECL(y_CPU,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NX,nx);

  /* Inizializza gli array con i dati appropriati. */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Avvia il timer per misurare il tempo di esecuzione del calcolo. */
  polybench_start_instruments;

	kernel_atax(nx, ny, 
	POLYBENCH_ARRAY(A), 
	POLYBENCH_ARRAY(x),
	POLYBENCH_ARRAY(y),
	POLYBENCH_ARRAY(tmp)
	);

  polybench_stop_instruments;
  polybench_print_instruments;

  #ifdef CHECK_RESULTS 
  sequential_atax(nx, ny,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y_CPU),
              POLYBENCH_ARRAY(tmp));

	compareResults(ny, POLYBENCH_ARRAY(y_CPU), POLYBENCH_ARRAY(y));
    #endif


  /* Ferma il timer e stampa i risultati delle misurazioni. */


  /* Previene l'eliminazione del codice morto (DCE) e stampa i risultati finali di y. */
  polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));

  /* Dealloca la memoria per gli array per evitare perdite di memoria. */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(x);
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(tmp);


  return 0;
}
