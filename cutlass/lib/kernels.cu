#include <iostream>
#include "Kernels.cuh"

#ifdef USE_DX
    #include "CublasDx.cuh"
#else
    #include "Cutlass.cuh"
#endif

#include <cutlass/version.h>

extern "C" void print_cutlass_version() {
  std::printf("CUTLASS %d.%d.%d \n",
              CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
}


void test_kernel() {

    print_cutlass_version();
    printf("Running test kernel \n");
    #ifdef USE_DX
        static_lib_func();
    #else
        lib_func();
    #endif
}
