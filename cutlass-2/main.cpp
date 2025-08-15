#include <kernels_cutlass.cuh>
#include <kernels_dx.cuh>

int main() {

    test_dx();
    test_cutlass();
    return 0;
}