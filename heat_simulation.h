#ifndef HEAT_SIMULATION_H
#define HEAT_SIMULATION_H

extern float *grid;
extern int grid_size;

void initialize_grid_with_block(int N, int threads_per_block);
void update_simulation();
void mantener_fuentes_de_calor(float *_grid);
void destroy__grid();

#endif
