#include "kernels_cutlass.cuh"
#include <cutlass/cutlass.h>
#include <cute/layout.hpp>
#include <cutlass/version.h>

using namespace cute;

void test_cutlass() {
    // auto shape = make_shape(_2{}, _2{});
    printf("using cutlass %d %d %d \n", CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
}
