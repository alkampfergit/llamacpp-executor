// Pin a precise amount of VRAM on a chosen device, then idle until killed.
// Used to vary free VRAM as an independent variable while every llama.cpp
// runtime flag is held fixed.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char ** argv) {
    int    device = (argc > 1) ? atoi(argv[1]) : 0;
    size_t mib    = (argc > 2) ? (size_t) atoll(argv[2]) : 256;

    if (cudaSetDevice(device) != cudaSuccess) { printf("setDevice failed\n"); return 1; }

    void * p = nullptr;
    cudaError_t e = cudaMalloc(&p, mib * 1024 * 1024);
    if (e != cudaSuccess) { printf("cudaMalloc %zu MiB failed: %s\n", mib, cudaGetErrorString(e)); return 2; }
    cudaMemset(p, 0, mib * 1024 * 1024);
    cudaDeviceSynchronize();

    printf("BALLAST OK device=%d %zu MiB\n", device, mib);
    fflush(stdout);
    for (;;) {
#ifdef _WIN32
        Sleep(1000);
#endif
    }
}
