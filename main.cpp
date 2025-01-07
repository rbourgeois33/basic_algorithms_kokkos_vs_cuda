#include <Kokkos_Core.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define GB (1e9)
#define MB (1024 * 1024)
#define KB 1024
#define BLOCK_SIZE 512
#define NREPEAT_KERNEL 10
#define NREPEAT_MEMCPY 10

#include "stencil_cuda.h"

int main(int argc, char *argv[])
{

    std::cout << "Kokkos and CUDA implementation comparisons\n";

    // Get device properties and size problems accordingly
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int GlobalGMem_MB = prop.totalGlobalMem / MB;
    std::cout << "Device " << 0 << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << GlobalGMem_MB << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max sharedMemPerBlock: " << prop.sharedMemPerBlock/KB <<"KB" <<std::endl;
    // We use arrays of memsize of a 100th of the global GMem
    const int MemSizeArraysMB = GlobalGMem_MB / 100;

    // Initialize Kokkos runtime
    Kokkos::initialize(argc, argv);

    { // Kokkos tests
    }

    // Finalize Kokkos runtime
    Kokkos::finalize();

    { // CUDA tests
        const int radius = 7;
        stencil_cuda<float, radius>(MemSizeArraysMB);
        stencil_cuda_shared_memory<float, radius>(MemSizeArraysMB, prop.sharedMemPerBlock);
    }

    return 0;
}
