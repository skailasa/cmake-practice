#include <iostream>
#include <vector>

#include <cublasdx.hpp>
#include <cuda_runtime_api.h>

using namespace cublasdx;


void static_lib_func() {

    using GEMM = decltype(Size<32, 32, 32>() + Precision<double>() + Type<type::real>() + Function<function::MM>() + Arrangement<cublasdx::row_major, cublasdx::col_major>());

    printf("Using CuBLASdx\n");
}