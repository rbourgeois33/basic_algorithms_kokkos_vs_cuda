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
    auto hp_data = HostPinnedView<_TYPE_>("hp_data", dataSize);

    // -- Naive implem
    auto policy = policy_t(0, dataSize);

    cudaEventRecord(start);

    Kokkos::deep_copy(data, h_data);
    kokkos_kernel<_TYPE_>(data, policy);
    Kokkos::deep_copy(h_data, data);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive_host;
    cudaEventElapsedTime(&ms_naive_host, start, stop);

    std::cout << "kokkos default host: " << ms_naive_host << "\n";

    // -- Naive implem - pinned
    HostPinned hostPinned;

    cudaEventRecord(start);

    Kokkos::deep_copy(data, hp_data);
    kokkos_kernel<_TYPE_>(data, policy);
    Kokkos::deep_copy(hp_data, data);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive_host_pinned;
    cudaEventElapsedTime(&ms_naive_host_pinned, start, stop);

    std::cout << "kokkos pinned host: " << ms_naive_host_pinned << "\n";

    //We divide the work into ndiv parts that will take place in two instances (2 streams)
    int nInstances = 2;
    const int ndiv = 20; 

    // Create the instances
    Device device; //exec space to pass as argument below
    std::vector<int> weights(nInstances, 1); //vector of weights 1
    auto Instances = Kokkos::Experimental::partition_space(device, weights);

    //declare subviews and policies
    View<_TYPE_> subviews[ndiv];
    HostPinnedView<_TYPE_> hp_subviews[ndiv];   //Pinned !!! otherwise D2H is blocking !!
    policy_t policies[ndiv]; 

    const int streamSize = dataSize / ndiv;
    
    //initialize subviews to separate the work in ndiv parts (unmanaged, no allocation)
    //also initialize the policies
    for (size_t i = 0; i < ndiv; i++)
    {
        int instance_id = i%2;
        const int beg = i * streamSize;
        const int end = (i + 1) * streamSize > dataSize ? dataSize - beg : (i + 1) * streamSize;
        policies[i] = policy_t(Instances[instance_id], beg, end);
        subviews[i] = Kokkos::subview(data, std::make_pair(beg, end));
        hp_subviews[i] = Kokkos::subview(hp_data, std::make_pair(beg, end));
    }

    cudaEventRecord(start);
    for (size_t i = 0; i < ndiv; i++)
    {
        int instance_id = i%2;
        Kokkos::deep_copy(Instances[instance_id], subviews[i], hp_subviews[i]);
        kokkos_kernel<_TYPE_>(data, policies[i]);
        Kokkos::deep_copy(Instances[instance_id], hp_subviews[i], subviews[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_stream;
    cudaEventElapsedTime(&ms_stream, start, stop);
    std::cout << "kokkos nInstances" << nInstances << "  time: " << ms_stream << "\n";
    std::cout << "NOTE: Concurrency requires pinned memory on the host !\n";
}