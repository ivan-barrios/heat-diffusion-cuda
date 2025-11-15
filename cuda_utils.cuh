//
// Utilidades simples para CUDA: manejo de errores e información de dispositivos.
//

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Información básica de las GPUs disponibles (para debug/manual).
static void getDevicesInfo()
{
    int devices;
    HANDLE_ERROR(cudaGetDeviceCount(&devices));

    for (int i = 0; i < devices; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("<---------------- GPU %d ---------------->\n", i);
        printf("Name: %s\n", prop.name);
        printf("Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total memory: %llu Bytes\n", (unsigned long long)prop.totalGlobalMem);
        printf("Shared memory per block: %llu Bytes\n", (unsigned long long)prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Max. threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max. block dimensions: %dx%dx%d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max. grid dimensions: %dx%dx%d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

#endif // CUDA_UTILS_CUH
