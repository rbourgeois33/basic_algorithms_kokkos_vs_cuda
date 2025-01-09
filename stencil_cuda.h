#include "utils_check_stencil.h"

template <typename _TYPE_>
__global__ void set_to_idx(_TYPE_ *array, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx >= 0) && (idx < N))
        array[idx] = (_TYPE_)idx;
}

template <typename _TYPE_>
__global__ void set_to(_TYPE_ *array, _TYPE_ value, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx >= 0) && (idx < N))
        array[idx] = value;
}

template <typename _TYPE_, int radius, bool use_buffer=true>
__global__ void stencil_cuda_kernel(_TYPE_ *input, _TYPE_ *output, int N)
{
    // global index (spans from 0 to N)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // We only do the stencil reduction in cells that have ar least radius cells on their left/right
    if ((idx >= radius) && (idx < N - radius))
    {
        //Use a buffer ! crucial for performance

        _TYPE_ result = 0;

        // stencil operation (sum over neighbors)
        for (int i = -radius; i <= radius; i++)
        {
            if constexpr(use_buffer)
            {
                result += input[idx + i];
            }else
            {
                output[idx] += input[idx + i];
            }
        }
        if constexpr(use_buffer) output[idx] = result;
    }
}

template <typename _TYPE_, int radius, bool use_buffer=true>
__global__ void stencil_cuda_shared_memory_kernel(_TYPE_ *input, _TYPE_ *output, int N)
{
    // global index (spans from 0 to N)
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    // local index in the block (spans for 0 to blockDim.x)
    int idx_loc = threadIdx.x;
    // local index offset by radius to match the shared memory array
    int idx_shared = threadIdx.x + radius;

    // Allocation of the shared memory, the size is from the 3rd launch parameter (extern keyword)
    // Its the size of the bloc+2 radius for BC
    extern __shared__ _TYPE_ shared[];

    // Bound check (we fill the shared memory on the whole array)
    if ((gidx >= 0) && (gidx < N))
    {
        // Fill the shared memory
        shared[idx_shared] = input[gidx];

        // The left threads fills the halo of the shared memiry
        if ((idx_loc < radius) && (gidx - radius >= 0))
        {
            shared[idx_shared - radius] = input[gidx - radius];
        }

        // The right thread fills the halo of the shared memory
        if ((idx_shared >= blockDim.x) && (gidx + radius < N))
        {
            shared[idx_shared + radius] = input[gidx + radius];
        }

        // Memory needs to be synced
        __syncthreads();

        // Do the stencil operation
        if ((gidx >= radius) && (gidx < N - radius))
        {
            //Use a buffer ! crucial for performance
            _TYPE_ result = 0;

            for (int i = -radius; i <= radius; i++)
            {
                if constexpr(use_buffer)
                {
                    result += shared[idx_shared + i];
                }else
                {
                    output[gidx] += shared[idx_shared + i];
                }
                
            }
            if constexpr(use_buffer) output[gidx] = result;
        }
    }
}

template <typename _TYPE_, int radius>
void stencil_cuda(const int MemSizeArraysMB, const int N_imposed = -1)
{
    //Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    std::cout<<"\n ||| CUDA KERNELS, dtype = "<< typeid(_TYPE_).name()<<" |||\n";

    // If N_imposed is provided, we scale the problem accordingly
    // then we also assume that the user wants to test the kernel.
    // We set input[i]=i and check the validity of the result to detect any indexing error
    // This fails however for large value of N because of round-off error
    // In non-test mode, we simply impose input[i]=1 and check that output[i]=2*radius+1

    // dimension problem
    const bool test_mode = N_imposed > 0 ? true : false;
    const int N = test_mode ? N_imposed : MB * MemSizeArraysMB / sizeof(_TYPE_);
    const int dataSize = N * sizeof(_TYPE_);

    //Operation and accesses counts
    // number of operations, stencil operation on the vector
    long operations = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * (2 * radius + 1); 
    // number of memory accesses, assuming perfect caching, input is read, output is modified and read 3xN
    long mem_accesses = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * 2;

    // Allocate GPU memory
    _TYPE_ *input, *output, *h_output;

    cudaMalloc((void **)&input, dataSize);
    cudaMalloc((void **)&output, dataSize);

    // Allocate CPU memory (for check)
    h_output = (_TYPE_ *)malloc(dataSize);

    // Dimension block and grid
    int NBLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "\nN, NBLOCKS, BLOCK_SIZE= " << N << " " << NBLOCKS << " " << BLOCK_SIZE << "\n";


    // Shared memory size computation
    const int shared_memory_size = (BLOCK_SIZE + 2 * radius) * sizeof(_TYPE_);
    std::cout << "shared memory allocated per block = " << ((float)shared_memory_size) / KB << "KB \n";

    // --------------------------------Verification stencil_cuda_kernel-------------------------------- //
    // Initialize data
    if (test_mode)
    {
        set_to_idx<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(input, N);
    }
    else
    {
        set_to<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(input, 1, N);
    }
    set_to<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(output, 0, N);

    // Check 1st kernel works
    stencil_cuda_kernel<_TYPE_, radius><<<NBLOCKS, BLOCK_SIZE>>>(input, output, N);
    cudaMemcpy(h_output, output, dataSize, cudaMemcpyDeviceToHost);

    _TYPE_ err = 0;
    for (int i = radius; i < N - radius; i++)
    {
        _TYPE_ sol_i = test_mode ? exact_result_stencil_kernel<_TYPE_, radius>(i) : (2 * radius + 1);
        err += abs(h_output[i] - sol_i);
    }
    std::cout << "stencil_cuda_kernel error (must be 0)="<< err<<std::endl;

    // --------------------------------Verification stencil_cuda_shared_memory_kernel -------------------------------- //

    // re-initialize data
    if (test_mode)
    {
        set_to_idx<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(input, N);
    }
    else
    {
        set_to<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(input, 1, N);
    }
    set_to<_TYPE_><<<NBLOCKS, BLOCK_SIZE>>>(output, 0, N);

    // Check if we have enough shared memory to launch our configuration
    // This is not check automatically and can lead to errors
    //  Get device properties and size problems accordingly
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // const int n_simultaneous_blocs_per_sm = prop.maxThreadsPerMultiProcessor / BLOCK_SIZE;
    // const int total_simultaneous_shared_mem_requested_per_sm = n_simultaneous_blocs_per_sm * shared_memory_size;

    // if (total_simultaneous_shared_mem_requested_per_sm > prop.sharedMemPerMultiprocessor)
    // {
    //     std::cout << "Warning: Total shared memory requested per SM " << total_simultaneous_shared_mem_requested_per_sm / KB << "KB exceeds the maximum shared memory available per SM " << prop.sharedMemPerMultiprocessor / KB << " KB" << std::endl;
    // }
    // if (shared_memory_size > prop.sharedMemPerBlock)
    // {
    //     std::cout << "Error: Shared memory allocation exceeds the limit. Reduce radius or block size." << std::endl;
    // }
    // End of check

    // Check 2n kernel works
    stencil_cuda_shared_memory_kernel<_TYPE_, radius><<<NBLOCKS, BLOCK_SIZE, shared_memory_size>>>(input, output, N);
    cudaMemcpy(h_output, output, dataSize, cudaMemcpyDeviceToHost);

    _TYPE_ err_shared = 0;
    for (int i = radius + 1; i < N - radius; i++)
    {
        _TYPE_ sol_i = test_mode ? exact_result_stencil_kernel<_TYPE_, radius>(i) : (2 * radius + 1);
        err_shared += abs(h_output[i] - sol_i);
    }
    std::cout << "stencil_cuda_shared_memory_kernel error (must be 0)="<<err_shared<<std::endl;

    // --------------------------------Timing stencil_cuda_kernel-------------------------------- //

    // Measure 1st kernel execution time
    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_kernel<_TYPE_, radius><<<NBLOCKS, BLOCK_SIZE>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float tflops = operations / (ms / 1000.0f) / 1e12;
    float bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_cuda_kernel **");
    
    // --------------------------------Timing stencil_cuda_kernel with no buffer-------------------------------- //

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_kernel<_TYPE_, radius, false><<<NBLOCKS, BLOCK_SIZE>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    tflops = operations / (ms / 1000.0f) / 1e12;
    bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_cuda_kernel, no buffer **");

    // --------------------------------Timing stencil_cuda_shared_memory_kernel-------------------------------- //

    // Measure 2nd kernel execution time

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_shared_memory_kernel<_TYPE_, radius><<<NBLOCKS, BLOCK_SIZE, shared_memory_size>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_shared;
    cudaEventElapsedTime(&ms_shared, start, stop);

    float tflops_shared = operations / (ms_shared / 1000.0f) / 1e12;
    float bw_shared = sizeof(_TYPE_) * mem_accesses / (ms_shared / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms_shared, "** stencil_cuda_shared_memory_kernel **");
    
    // --------------------------------Timing stencil_cuda_shared_memory_kernel with no buffer-------------------------------- //

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_shared_memory_kernel<_TYPE_, radius, false><<<NBLOCKS, BLOCK_SIZE, shared_memory_size>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms_shared, start, stop);

    tflops_shared = operations / (ms_shared / 1000.0f) / 1e12;
    bw_shared = sizeof(_TYPE_) * mem_accesses / (ms_shared / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms_shared, "** stencil_cuda_shared_memory_kernel, no buffer **");

    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_output);

    return;
}
