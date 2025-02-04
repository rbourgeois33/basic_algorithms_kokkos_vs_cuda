
// Compute bound kernel: performs multiply-add operations
template <typename _TYPE_>
__global__ void cuda_kernel(_TYPE_ *data, const int offset, const int dataSize)
{

    int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dataSize)
    {
        _TYPE_ a = sqrt(data[idx]);
        _TYPE_ b = cos(a);
        _TYPE_ c = sin(b);

        data[idx] = c * a * b * data[idx];
    }
}

template <typename _TYPE_>
void streams_cuda(const int MemSizeArraysMB, const int N_imposed = -1)
// Showcase the use of streams to overlap H2D /D2H transfers with host/device kernels
{
    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n ||| CUDA STREAMS, dtype = " << typeid(_TYPE_).name() << " |||\n";

    // If N_imposed is provided, we scale the problem accordingly
    // then we also assume that the user wants to test the kernel.
    // We set input[i]=i and check the validity of the result to detect any indexing error
    // This fails however for large value of dataSize because of round-off error
    // In non-test mode, we simply impose input[i]=1 and check that output[i]=2*radius+1

    // dimension problem
    const int dataSize = MB * MemSizeArraysMB / sizeof(_TYPE_);
    const int dataBytes = dataSize * sizeof(_TYPE_);

    // Allocate GPU memory
    _TYPE_ *data, *h_data;

    // host mem Must be pinned
    cudaMallocHost((void **)&h_data, dataBytes);
    cudaMalloc((void **)&data, dataBytes);

    // Dimension block and grid
    int NBLOCKS = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "\ndataSize, NBLOCKS, BLOCK_SIZE= " << dataSize << " " << NBLOCKS << " " << BLOCK_SIZE << "\n";

    // Naive implem
    cudaEventRecord(start);
    cudaMemcpy(data, h_data, dataBytes, cudaMemcpyHostToDevice);
    cuda_kernel<<<NBLOCKS, BLOCK_SIZE>>>(data, 0, dataSize);
    cudaMemcpy(h_data, data, dataBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    std::cout << "cuda naive time: " << ms_naive << "\n";

    //try several nStreams
    for (int nStreams = 20; nStreams <= 20; nStreams += 1)
    {
        const int streamSize = dataSize / nStreams;
        int streamBytes = streamSize * sizeof(_TYPE_);
        cudaStream_t streams[nStreams];
        int NBLOCKSstream = (streamSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (size_t i = 0; i < nStreams; i++)
        {
            cudaStreamCreate(&streams[i]);
        }

        cudaEventRecord(start);
        for (size_t i = 0; i < nStreams; i++)
        {
            const int offset = i * streamSize;
            if (offset + streamSize >= dataSize)
            {
                streamBytes = (dataSize - offset) * sizeof(_TYPE_);
            }
            // you have to use &. data[offset] is the offset-th element, & data[offset] is the adress of the offset-th element
            cudaMemcpyAsync(&data[offset], &h_data[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
            // 0 shared memory allocated
            cuda_kernel<<<NBLOCKSstream, BLOCK_SIZE, 0, streams[i]>>>(data, offset, dataSize);
            cudaMemcpyAsync(&h_data[offset], &data[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms_stream;
        cudaEventElapsedTime(&ms_stream, start, stop);
        std::cout << "cuda nStreams" << nStreams << "  time: " << ms_stream << "\n";

        for (size_t i = 0; i < nStreams; i++)
        {
            cudaStreamDestroy(streams[i]);
        };
    }

    //Only two streams, sharing chunks of ndiv-th of the work
    const int ndiv=20;
    const int streamSize = dataSize / ndiv;
    int streamBytes = streamSize * sizeof(_TYPE_);
    cudaStream_t streams[2];
    int NBLOCKSstream = (streamSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    cudaEventRecord(start);

    for (size_t i = 0; i < ndiv; i++)
    {
        int stream_id = i%2;
        const int offset = i * streamSize;
        if (offset + streamSize >= dataSize)
        {
            streamBytes = (dataSize - offset) * sizeof(_TYPE_);
        }
        // you have to use &. data[offset] is the offset-th element, & data[offset] is the adress of the offset-th element
        cudaMemcpyAsync(&data[offset], &h_data[offset], streamBytes, cudaMemcpyHostToDevice, streams[stream_id]);
        // 0 shared memory allocated
        cuda_kernel<<<NBLOCKSstream, BLOCK_SIZE, 0, streams[stream_id]>>>(data, offset, dataSize);
        cudaMemcpyAsync(&h_data[offset], &data[offset], streamBytes, cudaMemcpyDeviceToHost, streams[stream_id]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_stream_2;
    cudaEventElapsedTime(&ms_stream_2, start, stop);
    std::cout << "cuda two streams, ndiv:" << ndiv << "  time: " << ms_stream_2 << "\n";
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    // Free memory
    cudaFreeHost(h_data);
    cudaFree(data);
}