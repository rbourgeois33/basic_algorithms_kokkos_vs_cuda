template <typename _TYPE_>
__global__ void set_to(_TYPE_ *array, _TYPE_ value, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = value;
}

template <typename _TYPE_, int radius>
__global__ void stencil_cuda_kernel(_TYPE_ *input, _TYPE_ *output, int N)
{   
    //global index (spans from 0 to N)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //We only do the stencil reduction in cells that have ar leats radius cells on their left/right
    if ((idx >= radius) && (idx < N - radius))
    {   
        //stencil operation (sum over neighbors)
        for (int i = -radius; i <= radius; i++) 
            output[idx] += input[idx + i];
    }
}

template <typename _TYPE_, int radius>
__global__ void stencil_cuda_shared_memory_kernel(_TYPE_ *input, _TYPE_ *output, int N)
{   
    //global index (spans from 0 to N)
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    //local index in the block (spans for 0 to blockDim.x)
    int idx_loc = threadIdx.x;
    //local index offset by radius to match the shared memory array
    int idx_shared = threadIdx.x + radius;

    //Allocation of the shared memory, the size is from the 3rd launch parameter (extern keyword)
    //Its the size of the bloc+2 radius for BC
    extern __shared__ _TYPE_ shared[];

    //Bound check (we fill the shared memory on the whole array)
    if ((gidx >= 0) && (gidx < N))
    {
        //Fill the shared memory
        shared[idx_shared] = input[gidx];

        //The left threads fills the halo of the shared memiry
        if ((idx_loc < radius) && (gidx-radius>=0))
        {
            shared[idx_shared - radius] = input[gidx - radius];
        }

        //The right thread fills the halo of the shared memory
        if ((idx_shared >= blockDim.x) && (gidx+radius<N))
        {
            shared[idx_shared + radius] = input[gidx + radius];
        }

        //Memory needs to be synced
        __syncthreads();

        //Do the stencil operation
        if ((gidx >= radius) && (gidx < N - radius))
        {
            for (int i = -radius; i <= radius; i++)
            {
                output[gidx] += shared[idx_shared + i];
            }
        }
    }
}

template <typename _TYPE_, int radius>
void stencil_cuda(const int MemSizeArraysMB)
{

    // dimension problem
    const int N = MB * MemSizeArraysMB / sizeof(_TYPE_);
    const int dataSize = N * sizeof(_TYPE_);

    // Allocate GPU memory
    _TYPE_ *input, *output, *h_output;

    cudaMalloc((void **)&input, dataSize);
    cudaMalloc((void **)&output, dataSize);

    // Allocate CPU memory (for check)
    h_output = (_TYPE_ *)malloc(dataSize);

    // Dimension block and grid
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initialize data
    set_to<_TYPE_><<<grid, block>>>(input, 1, N);
    set_to<_TYPE_><<<grid, block>>>(output, 0, N);

    // Check kernel works
    stencil_cuda_kernel<_TYPE_, radius><<<grid, block>>>(input, output, N);
    cudaMemcpy(h_output, output, dataSize, cudaMemcpyDeviceToHost);

    _TYPE_ err = 0;
    for (int i = radius; i < N - radius; i++)
    {
        err += abs(h_output[i] - (2 * radius + 1));
    }

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_kernel<_TYPE_, radius><<<grid, block>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    long operations = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * (2 * radius + 1);         // number of operations
    //number of memory accesses, assuming perfect caching, input is read, output is modified and read 3xN
    long mem_accesses = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * 3; 

    float tflops = operations / (ms / 1000.0f) / 1e12;
    float bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;
    std::cout << "\n** stencil_cuda_kernel **\n";
    std::cout << "error = " << err << "\n";
    std::cout << "elapsed time = " << ms << " ms\n";
    std::cout << "FLOPS        = " << tflops << " TFLOPS\n";
    std::cout << "bandwith (assuming perfect caching) = " << bw << " GB/s\n";

    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}

template <typename _TYPE_, int radius>
void stencil_cuda_shared_memory(const int MemSizeArraysMB)
{

    // dimension problem
    const int N = MB * MemSizeArraysMB / sizeof(_TYPE_);
    const int dataSize = N * sizeof(_TYPE_);

    // Allocate GPU memory
    _TYPE_ *input, *output, *h_output;

    cudaMalloc((void **)&input, dataSize);
    cudaMalloc((void **)&output, dataSize);

    // Allocate CPU memory (for check)
    h_output = (_TYPE_ *)malloc(dataSize);

    // Dimension block and grid
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Shared memory size computation
    const int shared_memory_size = (BLOCK_SIZE + 2 * radius) * sizeof(_TYPE_);

    // Initialize data
    set_to<_TYPE_><<<grid, block>>>(input, 1, N);
    set_to<_TYPE_><<<grid, block>>>(output, 0, N);

    // Check kernel works
    stencil_cuda_shared_memory_kernel<_TYPE_, radius><<<grid, block, shared_memory_size>>>(input, output, N);
    cudaMemcpy(h_output, output, dataSize, cudaMemcpyDeviceToHost);

    _TYPE_ err = 0;
    for (int i = radius + 1; i < N - radius; i++)
    {
        err += abs(h_output[i] - (2 * radius + 1));
    }

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_cuda_shared_memory_kernel<_TYPE_, radius><<<grid, block, shared_memory_size>>>(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    long operations = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * (2 * radius + 1);// number of operations
    //number of memory accesses, assuming perfect caching, input is read, output is modified and read 3xN
    long mem_accesses = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * 3; 

    float tflops = operations / (ms / 1000.0f) / 1e12;
    float bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    std::cout << "\n** stencil_cuda_shared_memory_kernel **\n";
    std::cout << "error = " << err << "\n";
    std::cout << "elapsed time = " << ms << " ms\n";
    std::cout << "FLOPS        = " << tflops << " TFLOPS\n";
    std::cout << "bandwith     = " << bw << " GB/s\n";
    std::cout << "shared memory allocated per block = " << ((float)shared_memory_size)/KB << "KB \n";


    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}