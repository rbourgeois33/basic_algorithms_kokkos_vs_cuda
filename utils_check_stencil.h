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
