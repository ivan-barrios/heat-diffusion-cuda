#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "heat_simulation.h"
#include "cuda_utils.cuh"

float *grid = NULL; // buffer en CPU de grilla.
int grid_size = 0;  // lado N de la grilla (N x N celdas)

// buffer en host para la reducción (cálculo del promedio).
static float *h_partial = NULL;
static int h_partial_capacity = 0;

// diffusion_rate es el coeficiente de difusión D del modelo.
static float diffusion_rate = 0.25f;

// ====================== CONTROL DE EQUILIBRIO ======================
// uso el criterio |prom(G(t+1)) - prom(G(t))| <= EPSILON, donde prom(G(t)) es el promedio de la grilla.
static const float EPSILON = 0.000471f;
static int equilibrium_reached = 0;
static float prev_avg = 0.0f;
static int has_prev_avg = 0;

// ====================== BUFFERS EN DEVICE (GPU) ======================
// En la GPU guardo el estado actual y el siguiente.
static float *d_grid = NULL; // estado T(t)
static float *d_new = NULL;  // estado T(t+1)

// buffer en GPU para las sumas parciales de la reduccion.
static float *d_partial = NULL;  // sumas parciales de la reduccion
static int d_partial_blocks = 0; // cantidad de bloques usados en la reduccion

// ====================== KERNEL DE REDUCCIÓN (DEVICE) ======================
// hago una reduccion paralela para sumar un arreglo de floats.
// siguiendo el esquema "Reduction #4: First Add During Load" de Mark Harris o tambien esta en la diapositiva 21 del pdf "Ejemplo de aplicación Reducción Optimización" de Adrian Pousa.
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

    // reduzco en shared memory dentro del bloque.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // escribo la suma parcial del bloque en memoria global.
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ====================== HELPER EN HOST ======================
// calculo el promedio de la grilla N x N que está en la GPU usando el kernel de reduccion.
static float compute_grid_average_device(const float *d_vals, int N)
{
    const size_t n = (size_t)N * (size_t)N;

    const int threads = 256; // hilos por bloque para la reducción
    const int blocks = (int)((n + threads * 2 - 1) / (threads * 2));

    // me aseguro de que el buffer d_partial exista y tenga tamaño suficiente.
    if (d_partial == NULL || d_partial_blocks < blocks)
    {
        fprintf(stderr, "Error: d_partial no inicializado o tamaño insuficiente (bloques requeridos = %d, disponibles = %d)\n",
                blocks, d_partial_blocks);
        exit(1);
    }

    // lanzo la reduccion en GPU.
    reduce_first_add<<<blocks, threads, threads * sizeof(float)>>>(d_vals, d_partial, (unsigned int)n);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // reservo o agrando el buffer reutilizable en host si hace falta.
    if (h_partial_capacity < blocks)
    {
        float *new_buf = (float *)realloc(h_partial, blocks * sizeof(float));
        if (new_buf == NULL)
        {
            fprintf(stderr, "Error: realloc h_partial (bloques requeridos = %d, capacidad previa = %d)\n",
                    blocks, h_partial_capacity);
            free(h_partial);
            h_partial = NULL;
            h_partial_capacity = 0;
            exit(1);
        }
        h_partial = new_buf;
        h_partial_capacity = blocks;
    }

    // bajo las sumas parciales a host.
    HANDLE_ERROR(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        sum += h_partial[i];
    }

    return sum / (float)n;
}

// ====================== KERNEL DE DIFUSIÓN (DEVICE) ======================
// Cada hilo actualiza una celda (x,y) usando 5 puntos.
// Uso shared memory 2D con un halo de 1 celda alrededor del bloque para poder leer los vecinos que no pertenecen al bloque.
__global__ void diffuse5_kernel(const float *in, float *out, int N, float D)
{
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx; // columna global
    int gy = blockIdx.y * blockDim.y + ty; // fila global

    int tile_w = blockDim.x + 2; // ancho del tile incluyendo el halo
    int lx = tx + 1;             // indice local X dentro del tile
    int ly = ty + 1;             // indice local Y dentro del tile

    // indice global lineal (sólo válido si gx, gy están en grilla).
    int g_idx = gy * N + gx;

    // Cargo la celda central del hilo.
    if (gx < N && gy < N)
    {
        tile[ly * tile_w + lx] = in[g_idx];
    }

    // Cargo el halo en X (vecinos izquierdo y derecho).
    // vecino izquierdo.
    if (tx == 0 && gx > 0 && gy < N)
    {
        tile[ly * tile_w + 0] = in[gy * N + (gx - 1)];
    }
    // vecino derecho.
    if (tx == blockDim.x - 1 && gx + 1 < N && gy < N)
    {
        tile[ly * tile_w + (tile_w - 1)] = in[gy * N + (gx + 1)];
    }

    // Cargo el halo en Y (vecinos superior e inferior).
    // vecino superior.
    if (ty == 0 && gy > 0 && gx < N)
    {
        tile[0 * tile_w + lx] = in[(gy - 1) * N + gx];
    }
    // vecino inferior.
    if (ty == blockDim.y - 1 && gy + 1 < N && gx < N)
    {
        tile[(blockDim.y + 1) * tile_w + lx] = in[(gy + 1) * N + gx];
    }

    __syncthreads();

    // Si el hilo quedó fuera de la grilla, no hago nada más.
    if (gx >= N || gy >= N)
        return;

    // En la frontera copio el valor original
    if (gx == 0 || gy == 0 || gx == N - 1 || gy == N - 1)
    {
        out[g_idx] = in[g_idx];
        return;
    }

    // Para celdas interiores uso sólo shared memory para leer vecinos.
    float c = tile[ly * tile_w + lx];
    float up = tile[(ly - 1) * tile_w + lx];
    float dn = tile[(ly + 1) * tile_w + lx];
    float lf = tile[ly * tile_w + (lx - 1)];
    float rg = tile[ly * tile_w + (lx + 1)];

    float sum = up + dn + lf + rg;
    out[g_idx] = c + D * (sum - 4.0f * c);
}

// se mantienen las fuentes de calor fijas en la grilla
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
    // reservo y limpio la grilla
    grid_size = N;
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    // limpio el estado de equilibrio para arrancar una simulación
    equilibrium_reached = 0;
    has_prev_avg = 0;
    prev_avg = 0.0f;

    grid = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++)
        grid[i] = 0.0f;

    // pongo las fuentes en grid para verlas desde el primer frame.
    mantener_fuentes_de_calor(grid);

    // reservo los buffers en memoria global de la GPU.
    HANDLE_ERROR(cudaMalloc((void **)&d_grid, bytes));
    HANDLE_ERROR(cudaMalloc((void **)&d_new, bytes));

    // reservo el buffer global para las sumas parciales de la reduccion (promedio).
    const size_t n = (size_t)N * (size_t)N;
    const int threads = 256;
    d_partial_blocks = (int)((n + threads * 2 - 1) / (threads * 2));
    HANDLE_ERROR(cudaMalloc((void **)&d_partial, d_partial_blocks * sizeof(float)));

    // Subo el estado inicial (ya con fuentes) a la GPU.
    HANDLE_ERROR(cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice));
}

void update_simulation()
{
    const int N = grid_size;
    if (N <= 0)
        return;

    // Si ya alcanzamos el equilibrio, no avanzamos mas la simulacion.
    if (equilibrium_reached)
        return;

    // Defino el tamaño de los bloques y la grilla de hilos.
    dim3 block(32, 8); // 256 hilos por bloque
    dim3 gridDim((N + block.x - 1) / block.x,
                 (N + block.y - 1) / block.y);

    // Calculo cuánta shared memory necesito para el tile 2D
    size_t shared_bytes = (block.x + 2) * (block.y + 2) * sizeof(float);

    // paso de difusión en GPU: d_grid (T(t)) -> d_new (T(t+1)).
    diffuse5_kernel<<<gridDim, block, shared_bytes>>>(d_grid, d_new, N, diffusion_rate);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Calculo el promedio de la grilla en GPU a partir de d_new.
    float current_avg = compute_grid_average_device(d_new, N);
    if (has_prev_avg)
    {
        float diff = fabsf(current_avg - prev_avg);
        printf("diff = %f\n", diff);
        if (diff <= EPSILON)
        {
            equilibrium_reached = 1;
            printf("Equilibrio alcanzado: avg = %f, diff = %f\n", current_avg, diff);
        }
    }
    prev_avg = current_avg;
    has_prev_avg = 1;

    // Traigo T(t+1) a host para dibujar.
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    HANDLE_ERROR(cudaMemcpy(grid, d_new, bytes, cudaMemcpyDeviceToHost));

    mantener_fuentes_de_calor(grid);

    // subo T(t+1) (ya con fuentes) a d_grid, que va a ser la entrada del próximo paso.
    HANDLE_ERROR(cudaMemcpy(d_grid, grid, bytes, cudaMemcpyHostToDevice));
}

void destroy__grid()
{
    // libero toda la memoria en CPU y GPU.

    free(grid);
    grid = NULL;

    grid_size = 0;

    cudaFree(d_grid);
    d_grid = NULL;

    cudaFree(d_new);
    d_new = NULL;

    cudaFree(d_partial);
    d_partial = NULL;
    d_partial_blocks = 0;

    free(h_partial);
    h_partial = NULL;
    h_partial_capacity = 0;

    equilibrium_reached = 0;
    has_prev_avg = 0;
    prev_avg = 0.0f;
}
