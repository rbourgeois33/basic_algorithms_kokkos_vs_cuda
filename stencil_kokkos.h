
// Kokkos aliases
template <typename _TYPE_>
using View = Kokkos::View<_TYPE_ *>;
using Device = Kokkos::DefaultExecutionSpace;
using policy_t = Kokkos::RangePolicy<Device>;
using team_policy_t = Kokkos::TeamPolicy<Device>;
using member_type = team_policy_t::member_type;
using ScratchSpace = Device::scratch_memory_space;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

//Scratch views must be unmanaged
template <typename _TYPE_>
using ScratchView = Kokkos::View<_TYPE_ *, ScratchSpace, Unmanaged>;

template <typename _TYPE_, int radius, bool use_buffer=true>
void stencil_kokkos_kernel(const View<_TYPE_> &input, View<_TYPE_> &output, const policy_t &policy)
{
    Kokkos::parallel_for("stencil operation", policy, KOKKOS_LAMBDA(const int idx) {
        
        //Use a buffer ! crucial for performance
        _TYPE_ result = 0;

        if (use_buffer)
        {
        for (int i = -radius; i <= radius; i++)
        {
            result += input[idx + i];
        }
        output[idx] = result;
        }
        else
        {
        for (int i = -radius; i <= radius; i++)
        {
             output[idx] += input[idx + i];
        }
        }

    });
}

template <typename _TYPE_, int radius, bool use_buffer=true>
void stencil_kokkos_scratch_memory_kernel(const View<_TYPE_> &input, View<_TYPE_> &output, const team_policy_t &team_policy)
{
    Kokkos::parallel_for("stencil operation scratch memory", team_policy, KOKKOS_LAMBDA(member_type team_member) {
        // Allocate scratch memory
        auto scratch = ScratchView<_TYPE_>(team_member.team_scratch(0), team_member.team_size() + 2 * radius);

        // Global index (spans from 0 to N)
        const int gidx = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
        // Local index in the block (spans from 0 to blockDim.x)
        const int idx_loc = team_member.team_rank();
        // Local index offset by radius to match the shared memory array
        const int idx_scratch = team_member.team_rank() + radius;

        // Check bounds
        if (gidx < input.extent(0)) {
            // Fill scratch memory
            scratch(idx_scratch) = input(gidx);

            // Left halo
            if (idx_loc < radius && gidx - radius >= 0) {
                scratch(idx_scratch - radius) = input(gidx - radius);
            }

            // Right halo
            if (idx_loc >= team_member.team_size() - radius && gidx + radius < input.extent(0)) {
                scratch(idx_scratch + radius) = input(gidx + radius);
            }

            // Synchronize the team
            team_member.team_barrier();

            // Perform stencil computation
            if (gidx >= radius && gidx < input.extent(0) - radius) {
                _TYPE_ result = 0;

                for (int i = -radius; i <= radius; i++) {
                    result += scratch(idx_scratch + i);
                }

                output(gidx) = result;
            }
        }
    });
}


template <typename _TYPE_, int radius>
void stencil_kokkos(const int MemSizeArraysMB, const int N_imposed = -1)
{   
    //Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout<<"\n ||| KOKKOS KERNELS, dtype = "<< typeid(_TYPE_).name()<<" |||\n";
    
    // If N_imposed is provided, we scale the problem accordingly
    // then we also assume that the user wants to test the kernel.
    // We set input[i]=i and check the validity of the result to detect any indexing error
    // This fails however for large value of N because of round-off error
    // In non-test mode, we simply impose input[i]=1 and check that output[i]=2*radius+1

    // dimension problem
    const bool test_mode = N_imposed > 0 ? true : false;
    const int N = test_mode ? N_imposed : MB * MemSizeArraysMB / sizeof(_TYPE_);

    //Operation and accesses counts
    // number of operations, stencil operation on the vector
    long operations = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * (2 * radius + 1); 
    // number of memory accesses, assuming perfect caching, input is read, output is modified and read 3xN
    long mem_accesses = NREPEAT_KERNEL * static_cast<long>(N - 2 * radius) * 2;

    // Allocate Views on GPU and CPU
    auto input = View<_TYPE_>("input", N);
    auto output = View<_TYPE_>("output", N);
    auto mirror_output = Kokkos::create_mirror_view(output);

    // --------------------------------Verification stencil_kokkos_kernel-------------------------------- //

    // Initialize Views
    Kokkos::parallel_for("init input", N, KOKKOS_LAMBDA(const int idx) { input[idx] = test_mode ? idx : 1; });
    Kokkos::parallel_for("init output", N, KOKKOS_LAMBDA(const int idx) { output[idx] = 0; });

    auto stencil_policy = policy_t(radius, N - radius);

    // Stencil operation
    stencil_kokkos_kernel<_TYPE_, radius>(input, output, stencil_policy);
    Kokkos::fence();
    Kokkos::deep_copy(mirror_output, output);

    _TYPE_ err = 0;
    for (int i = radius; i < N - radius; i++)
    {
        _TYPE_ sol_i = test_mode ? exact_result_stencil_kernel<_TYPE_, radius>(i) : (2 * radius + 1);
        err += abs(mirror_output[i] - sol_i);
    }
    std::cout << "stencil_kokkos_kernel error (must be 0)= "<<err<<std::endl;

    // --------------------------------Timing stencil_kokkos_kernel-------------------------------- //
    
    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_kokkos_kernel<_TYPE_, radius>(input, output, stencil_policy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float tflops = operations / (ms / 1000.0f) / 1e12;
    float bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_kokkos_kernel **");

    // --------------------------------Timing stencil_kokkos_kernel with no buffer-------------------------------- //

    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_kokkos_kernel<_TYPE_, radius, false>(input, output, stencil_policy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    tflops = operations / (ms / 1000.0f) / 1e12;
    bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_kokkos_kernel, no buffer **");


    // --------------------------------Team policy definition-------------------------------- //
    //league size is the amount of blocks
    const int league_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //Size of the shared memory
    const int scratch_memory_size = (BLOCK_SIZE + 2 * radius) * sizeof(_TYPE_);

    //kokkos::auto: amount of threads per block
    //level of the scratch memory: 0--> shared/L1, 1=global
    team_policy_t team_policy = team_policy_t(league_size, BLOCK_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size));
    std::cout << "\nN, league_size, team_size= " << N << " " <<team_policy.league_size() << " " << team_policy.team_size() << "\n";
    
    // --------------------------------Verification stencil_kokkos_scratch_memory_kernel-------------------------------- //

    // Initialize Views
    Kokkos::parallel_for("init input", N, KOKKOS_LAMBDA(const int idx) { input[idx] = test_mode ? idx : 1; });
    Kokkos::parallel_for("init output", N, KOKKOS_LAMBDA(const int idx) { output[idx] = 0; });

    // Stencil operation
    stencil_kokkos_scratch_memory_kernel<_TYPE_, radius>(input, output, team_policy);
    Kokkos::deep_copy(mirror_output, output);

    err = 0;
    for (int i = radius; i < N - radius; i++)
    {
        _TYPE_ sol_i = test_mode ? exact_result_stencil_kernel<_TYPE_, radius>(i) : (2 * radius + 1);
        err += abs(mirror_output[i] - sol_i);
    }
    std::cout << "stencil_kokkos_scratch_memory_kernel error (must be 0)= "<<err<<std::endl;

    // --------------------------------Timing stencil_kokkos_scratch_memory_kernel-------------------------------- //
    
    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_kokkos_scratch_memory_kernel<_TYPE_, radius>(input, output, team_policy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    tflops = operations / (ms / 1000.0f) / 1e12;
    bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_kokkos_scratch_memory_kernel **");

    // --------------------------------Timing stencil_kokkos_scratch_memory_kernel no buffer-------------------------------- //
    
    cudaEventRecord(start);
    for (size_t n = 0; n < NREPEAT_KERNEL; n++)
    {
        stencil_kokkos_scratch_memory_kernel<_TYPE_, radius, false>(input, output, team_policy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    tflops = operations / (ms / 1000.0f) / 1e12;
    bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    print_perf<_TYPE_>(operations, mem_accesses, ms, "** stencil_kokkos_scratch_memory_kernel, no buffer **");    
}