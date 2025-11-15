#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "heat_simulation.h"
#include "cuda_utils.cuh"

// ====================== VARIABLES GLOBALES (HOST) ======================
// Estas dos son usadas por main.c (HOST).
float *grid = NULL; // HOST: buffer que pinta OpenGL en main.c en cada tick
int grid_size = 0;  // HOST: lado N de la grilla

// Parámetros del modelo (HOST)
static float diffusion_rate = 0.25f; // HOST: coeficiente D.

// ====================== CONTROL DE EQUILIBRIO ======================
// Criterio de convergencia: |Ḡ^(t+1) − Ḡ^t| <= EPSILON
static const float EPSILON = 1e-4f;
static int equilibrium_reached = 0;
static float prev_avg = 0.0f;
static int has_prev_avg = 0;

// ====================== BUFFERS EN DEVICE (GPU) ======================
static float *d_grid = NULL; // DEVICE global memory: estado T(t)
static float *d_new = NULL;  // DEVICE global memory: estado T(t+1)

// ====================== KERNEL DE REDUCCIÓN (DEVICE) ======================
// Reducción paralela para sumar un arreglo de floats siguiendo el esquema
// "Reduction #4: First Add During Load" de Mark Harris.
__global__ void reduce_first_add(const float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = 0.0f;
    if (i < n)
    {
        mySum = g_idata[i];
        if (i + blockDim.x < n)
        {
            mySum += g_idata[i + blockDim.x];
        }
    }

    sdata[tid] = mySum;
    __syncthreads();

    // Reducción secuencial en shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Guardamos el resultado parcial de este bloque en memoria global
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ====================== HELPER EN HOST ======================
// Calcula el promedio de N*N valores almacenados en device usando la reducción anterior.
static float compute_grid_average_device(const float *d_vals, int N)
{
    const size_t n = (size_t)N * (size_t)N;
    if (n == 0)
    {
        return 0.0f;
    }

    const int threads = 256;
    const int blocks = (int)((n + threads * 2 - 1) / (threads * 2));

    float *d_partial = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_partial, blocks * sizeof(float)));

    // Lanzamos la reducción en GPU
    reduce_first_add<<<blocks, threads, threads * sizeof(float)>>>(d_vals, d_partial, (unsigned int)n);
    HANDLE_ERROR(cudaDeviceSynchronize());

    float *h_partial = (float *)malloc(blocks * sizeof(float));
    if (h_partial == NULL)
    {
        fprintf(stderr, "Error: malloc h_partial\n");
        cudaFree(d_partial);
        exit(1);
    }

    HANDLE_ERROR(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        sum += h_partial[i];
    }

    free(h_partial);
    cudaFree(d_partial);

    return sum / (float)n;
}

// ====================== KERNEL (DEVICE) ======================
// Hilo = Celda (x,y). Sin shared memory. Sólo registros + global memory.
__global__ void diffuse5_kernel(const float *in, float *out, int N, float D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= N || y >= N)
        return;

    int idx = y * N + x;

    // Frontera fija
    if (x == 0 || y == 0 || x == N - 1 || y == N - 1)
    {
        out[idx] = in[idx];
        return;
    }

    float c = in[idx];
    float up = in[(y - 1) * N + x];
    float dn = in[(y + 1) * N + x];
    float lf = in[y * N + (x - 1)];
    float rg = in[y * N + (x + 1)];

    float sum = up + dn + lf + rg;
    out[idx] = c + D * (sum - 4.0f * c);
}

// HOST
// Misma funcion que en heat_simulation.c
void mantener_fuentes_de_calor(float *_grid)
{
    int cx = grid_size / 2;
    int cy = grid_size / 2;

    _grid[cy * grid_size + cx] = 100.0f;

    int offset = 20;
    _grid[(cy + offset) * grid_size + (cx + offset)] = 100.0f;
    _grid[(cy + offset) * grid_size + (cx - offset)] = 100.0f;
    _grid[(cy - offset) * grid_size + (cx + offset)] = 100.0f;
    _grid[(cy - offset) * grid_size + (cx - offset)] = 100.0f;
}

void initialize_grid(int N)
{
    // ===== HOST: reservo y limpio el buffer que pinta OpenGL =====
    grid_size = N;
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    // Reset de flags de equilibrio para una nueva simulación
    equilibrium_reached = 0;
    has_prev_avg = 0;
    prev_avg = 0.0f;

    grid = (float *)malloc(bytes);
    if (grid == NULL)
    {
        fprintf(stderr, "Error: malloc grid\n");
        exit(1);
    }

    for (int i = 0; i < N * N; i++)
        grid[i] = 0.0f;

    // Pinto las fuentes en HOST para verlas desde el primer frame
    mantener_fuentes_de_calor(grid); // HOST write

    // ===== DEVICE: reservo buffers en memoria global de GPU =====
    HANDLE_ERROR(cudaMalloc((void **)&d_grid, bytes));
    HANDLE_ERROR(cudaMalloc((void **)&d_new, bytes));

    // Subo el estado inicial (con fuentes) a la GPU
    HANDLE_ERROR(cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice));
}

void update_simulation()
{
    const int N = grid_size;
    if (N <= 0)
        return;

    // Si ya alcanzamos el equilibrio, no avanzamos más la simulación.
    if (equilibrium_reached)
        return;

    // 1) Configuración de la grilla de hilos
    dim3 block(32, 8); // 256 hilos por bloque
    dim3 gridDim((N + block.x - 1) / block.x,
                 (N + block.y - 1) / block.y);

    // 2) Paso de difusión en GPU: d_grid (T(t)) -> d_new (T(t+1))
    diffuse5_kernel<<<gridDim, block>>>(d_grid, d_new, N, diffusion_rate);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // 3) Calculamos el promedio de la grilla en GPU a partir de d_new
    float current_avg = compute_grid_average_device(d_new, N);
    if (has_prev_avg)
    {
        float diff = fabsf(current_avg - prev_avg);
        if (diff <= EPSILON)
        {
            equilibrium_reached = 1;
            printf("Equilibrio alcanzado: avg = %f, diff = %f\n", current_avg, diff);
        }
    }
    prev_avg = current_avg;
    has_prev_avg = 1;

    // 4) Traigo T(t+1) a HOST para dibujar
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    HANDLE_ERROR(cudaMemcpy(grid, d_new, bytes, cudaMemcpyDeviceToHost));

    // 5) Repongo fuentes en HOST
    mantener_fuentes_de_calor(grid);

    // 6) Subo T(t+1) con fuentes a d_grid (será la nueva entrada)
    HANDLE_ERROR(cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice));
}

void destroy__grid()
{
    // HOST: libero memoria en CPU y GPU

    free(grid);
    grid = NULL;

    grid_size = 0;

    cudaFree(d_grid);
    d_grid = NULL;

    cudaFree(d_new);
    d_new = NULL;

    // Reset de estado de equilibrio
    equilibrium_reached = 0;
    has_prev_avg = 0;
    prev_avg = 0.0f;
}
