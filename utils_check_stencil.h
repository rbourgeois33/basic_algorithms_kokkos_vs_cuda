// Utils function to check that our code is working

template <typename _TYPE_>
_TYPE_ cumulative_sum(const int n)
{
    return ((_TYPE_)n * (n + 1)) / 2;
}

template <typename _TYPE_, const int radius>
_TYPE_ exact_result_stencil_kernel(const int i)
{
    return cumulative_sum<_TYPE_>(i + radius) - cumulative_sum<_TYPE_>(i - radius - 1);
}

template<typename _TYPE_>
void print_perf(long operations, long mem_accesses, float ms, std::string name)
{
    float tflops = operations / (ms / 1000.0f) / 1e12;
    float bw = sizeof(_TYPE_) * mem_accesses / (ms / 1000.0f) / GB;

    std::cout << "\n"+name+"\n";
    std::cout << "elapsed time = " << ms << " ms\n";
    std::cout << "FLOPS        = " << tflops << " TFLOPS\n";
    std::cout << "bandwith (assuming perfect caching) = " << bw << " GB/s\n";
}
