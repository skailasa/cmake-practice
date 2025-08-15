#include "kernels_dx.cuh"
#include <cublasdx.hpp>
#include <cutlass/version.h>

void test_dx() {

    printf("using cublasdx with cutlass %d %d %d \n", CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
}