#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat_simulation.h"

// ====================== VARIABLES GLOBALES (HOST) ======================
// Estas dos son usadas por main.c (HOST).
float *grid = NULL; // HOST: buffer que pinta OpenGL en main.c en cada tick
int grid_size = 0;  // HOST: lado N de la grilla

// Parámetros del modelo (HOST)
static float diffusion_rate = 0.25f; // HOST: coeficiente D.

// ====================== BUFFERS EN DEVICE (GPU) ======================
static float *d_grid = NULL; // DEVICE global memory: estado T(t)
static float *d_new = NULL;  // DEVICE global memory: estado T(t+1)

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

    grid = (float *)malloc(bytes); // HOST RAM
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
    // d_grid = T(t), d_new = T(t+1)
    if (cudaMalloc((void **)&d_grid, bytes) != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaMalloc d_grid\n");
        exit(1);
    }
    if (cudaMalloc((void **)&d_new, bytes) != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaMalloc d_new\n");
        exit(1);
    }

    // Subo el estado inicial (con fuentes) a la GPU
    cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice); // H2D
    // d_new queda sin inicializar: el kernel lo va a escribir completo cada paso
}

void update_simulation()
{
    const int N = grid_size;
    if (N <= 0)
        return;

    // 1) Configuración de la grilla de hilos
    dim3 block(16, 16); // 256 hilos por bloque (8 warps)
    dim3 gridDim((N + block.x - 1) / block.x,
                 (N + block.y - 1) / block.y); // Total de threads con N=400: 256 * 28 * 28 = 200704 sabiendo que 400 x 400 = 160000 -> alcanza

    // 2) Paso de difusión en GPU: d_grid (T(t)) -> d_new (T(t+1))
    diffuse5_kernel<<<gridDim, block>>>(d_grid, d_new, N, diffusion_rate);
    cudaDeviceSynchronize(); // Esperamos a que termine el kernel

    // 3) Traigo T(t+1) a HOST para dibujar
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    cudaMemcpy(grid, d_new, bytes, cudaMemcpyDeviceToHost); // D2H

    // 4) Repongo fuentes en HOST
    mantener_fuentes_de_calor(grid);

    // 5) Subo T(t+1) con fuentes a d_grid (será la nueva entrada)
    cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice); // H2D
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
}
