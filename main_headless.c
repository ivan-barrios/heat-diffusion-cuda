// main_headless.c
#include <stdio.h>
#include <stdlib.h>
#include "heat_simulation.h"

// Uso: ./heat_sim N pasos [hilos_por_bloque] prefijo_salida
// Ej:  ./heat_sim 400 1000 256 frames/frame_   (256 = 32x8 hilos por bloque)

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Uso: %s N pasos [hilos_por_bloque] prefijo_salida\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);     // tamaño de la grilla
    int steps = atoi(argv[2]); // cantidad de iteraciones

    int threads_per_block = 32 * 8; // valor por defecto: 256 hilos por bloque (32x8)
    const char *prefix = NULL;

    if (argc >= 5)
    {
        // Se especificó hilos_por_bloque explícitamente
        threads_per_block = atoi(argv[3]);
        prefix = argv[4];
    }
    else
    {
        fprintf(stderr, "Error: mal uso.\n");
        return 1;
    }

    if (N <= 0)
    {
        fprintf(stderr, "Error: N debe ser mayor que 0.\n");
        return 1;
    }

    if (threads_per_block < 32 || (threads_per_block % 32) != 0)
    {
        fprintf(stderr, "Error: hilos_por_bloque debe ser múltiplo de 32 y al menos 32.\n");
        return 1;
    }

    if (threads_per_block > 1024)
    {
        fprintf(stderr, "Error: hilos_por_bloque no puede exceder 1024.\n");
        return 1;
    }

    initialize_grid_with_block(N, threads_per_block);

    // Guardamos, por ejemplo, un frame cada 10 pasos
    int dump_every = 10;

    for (int t = 0; t < steps; ++t)
    {
        update_simulation();

        if (t % dump_every == 0)
        {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s%05d.bin", prefix, t);

            FILE *f = fopen(fname, "wb");
            if (!f)
            {
                perror("fopen");
                break;
            }
            // grid es un float* de N*N elementos
            size_t total = (size_t)N * (size_t)N;
            fwrite(grid, sizeof(float), total, f);
            fclose(f);
        }
    }

    destroy__grid();
    return 0;
}
