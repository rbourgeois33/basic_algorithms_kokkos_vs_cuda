template <typename _TYPE_>
void kokkos_kernel(const View<_TYPE_> &data, const policy_t &policy)
{
    Kokkos::parallel_for("kokkos_kernel", policy, KOKKOS_LAMBDA(const int idx) {
        _TYPE_ buff = data[idx];
        _TYPE_ a = sqrt(buff);
        _TYPE_ b = cos(a);
        _TYPE_ c = sin(b);

        data(idx) = c * a * b * buff;
    });
}

template <typename _TYPE_>
void streams_kokkos(const int MemSizeArraysMB)
{
    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n ||| KOKKOS Streams (ExecSpaces), dtype = " << typeid(_TYPE_).name() << " |||\n";

    // dimension problem
    const int dataSize = MB * MemSizeArraysMB / sizeof(_TYPE_);

    // Allocate Views on GPU and CPU
    auto data = View<_TYPE_>("data", dataSize);
    auto h_data = HostView<_TYPE_>("h_data", dataSize);

    // Naive implem
    auto policy = policy_t(0, dataSize);

    cudaEventRecord(start);

    Kokkos::deep_copy(data, h_data);
    kokkos_kernel<_TYPE_>(data, policy);
    Kokkos::deep_copy(h_data, data);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    std::cout << "kokkos naive time (longer than cuda because non pinned !): " << ms_naive << "\n";

    Device device;
    
    int nInstances = 5;
    // Create a vector of size nInstances, initialized with 1
    std::vector<int> weights(nInstances, 1); 

    auto Instances = Kokkos::Experimental::partition_space(device, weights);
    policy_t policies[nInstances];
    View<_TYPE_> subviews[nInstances];
    HostView<_TYPE_> h_subviews[nInstances];

    const int streamSize = dataSize / nInstances;

    for (size_t i = 0; i < nInstances; i++)
    {
        const int beg = i * streamSize;
        const int end = (i + 1) * streamSize > dataSize ? dataSize - beg : (i + 1) * streamSize;
        policies[i] = policy_t(Instances[i], beg, end);
        subviews[i] = Kokkos::subview(data, std::make_pair(beg, end));
        h_subviews[i] = Kokkos::subview(h_data, std::make_pair(beg, end));
    }

    cudaEventRecord(start);
    for (size_t i = 0; i < nInstances; i++)
    {
        Kokkos::deep_copy(Instances[i], subviews[i], h_subviews[i]);
        //kokkos_kernel<_TYPE_>(data, policies[i]);
        Kokkos::deep_copy(Instances[i], h_subviews[i], subviews[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_stream;
    cudaEventElapsedTime(&ms_stream, start, stop);
    std::cout << "kokkos nInstances" << nInstances << "  time: " << ms_stream << "\n";
}