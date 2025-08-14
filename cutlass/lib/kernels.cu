#include <iostream>
#include "Kernels.cuh"

#ifdef USE_DX
#include "CublasDx.cuh"
#else
#include "Cutlass.cuh"
#endif


void test_kernel() {
    printf("Running test kernel \n");
    #ifdef USE_DX
    static_lib_func();
    #else
    lib_func();
    #endif
}