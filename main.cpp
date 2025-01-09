#include <Kokkos_Core.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <typeinfo>


#define GB (1024 * 1024 * 1024)
#define MB (1024 * 1024)
#define KB 1024
#define BLOCK_SIZE 512
#define NREPEAT_KERNEL 30

#include "stencil_cuda.h"
#include "stencil_kokkos.h"

int main(int argc, char *argv[])
{

    std::cout << "\nKokkos and CUDA implementation comparisons\n";

    // Get device properties and size problems accordingly
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int GlobalGMem_MB = prop.totalGlobalMem / MB;
    std::cout << "\nDevice " << 0 << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << GlobalGMem_MB << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max shared memory per SM: " << prop.sharedMemPerMultiprocessor /KB<< " KB" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max sharedMemPerBlock: " << prop.sharedMemPerBlock / KB << "KB" << std::endl;

    // We use arrays of memsize of a 100th of the global GMem
    const int MemSizeArraysMB = GlobalGMem_MB / 100;
    const int stencil_radius = 10; // The larger the radius, the bigger the perf increase with shared mem

    // Initialize Kokkos runtime
    Kokkos::initialize(argc, argv);

    { // Kokkos tests
        // stencil_kokkos<float, stencil_radius>(MemSizeArraysMB, /*small size for testing kernel*/ 5000);
        stencil_kokkos<float, stencil_radius>(MemSizeArraysMB);
        //stencil_kokkos<double, stencil_radius>(MemSizeArraysMB);

    }

    // Finalize Kokkos runtime
    Kokkos::finalize();

    { // CUDA tests
        // stencil_cuda<float, stencil_radius>(MemSizeArraysMB, /*small size for testing kernel*/ 5000);
        stencil_cuda<float, stencil_radius>(MemSizeArraysMB);
        //stencil_cuda<double, stencil_radius>(MemSizeArraysMB);

    }


    return 0;
}
