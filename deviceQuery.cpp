#include <memory>
#include <iostream>
#include <unistd.h>

#include <cuda_runtime.h>

int *pArgc = NULL;
char **pArgv = NULL;

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int dev = 0, driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    char msg[256];
    sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    printf("%s", msg);

    printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);

    printf("\n");

    size_t free, total;
    
    while(1){
        cudaMemGetInfo(&free, &total);
        printf("free: %.4f MBytes, total: %.4f MBytes\n", free/1048576.0f, total/1048576.0f);

        fflush(stdout);
        sleep(1);
    }
    

    printf("\n");
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
