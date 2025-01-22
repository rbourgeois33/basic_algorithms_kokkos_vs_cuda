# basic_algorithms_kokkos_vs_cuda

Basic implementation of common operations in cuda and kokkos, for now:

-A 1D stencil operation, with and without shared/scratch memory

-An optimized D2H / H2D data transfer overlaped with CUDA-streams / Kokkos-instances

### compile
```bash
git clone https://github.com/rbourgeois33/basic_algorithms_kokkos_vs_cuda.git
git submodule update --init --recursive
cd basic_algorithms_kokkos_vs_cuda.git
mkdir build ; cd build
```
in build, for A5000:
```bash
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON ..
```

in build, for Ada:
```bash
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON ..
```

```bash
make -j 12
./my_program
```

