// main_headless.c
#include <stdio.h>
#include <stdlib.h>
#include "heat_simulation.h"

// Uso: ./heat_sim N pasos prefijo_salida
// Ej:  ./heat_sim 400 1000 frames/frame_

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Uso: %s N pasos prefijo_salida\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);     // tamaño de la grilla
    int steps = atoi(argv[2]); // cantidad de iteraciones
    const char *prefix = argv[3];

    initialize_grid(N);

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

            printf("Guardado frame t=%d en %s\n", t, fname);
        }
    }

    destroy__grid();
    return 0;
}
