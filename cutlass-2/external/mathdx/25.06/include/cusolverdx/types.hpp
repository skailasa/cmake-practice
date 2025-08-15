// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_TYPES_HPP
#define CUSOLVERDX_TYPES_HPP

#include <cuComplex.h>

#include <cutlass/numeric_types.h>
#include <cutlass/complex.h>

#ifndef COMMONDX_DETAIL_ENABLE_CUTLASS_TYPES
#define COMMONDX_DETAIL_ENABLE_CUTLASS_TYPES
#endif
#include "commondx/complex_types.hpp"

namespace cusolverdx {
    // imported types, aliases and conversion utilities
    template<typename T>
    struct convert_to_cuda_type {
        using type = T;
    };

    template<>
    struct convert_to_cuda_type<commondx::complex<float>> {
        using type = cuFloatComplex;
        static_assert((alignof(commondx::complex<float>) >= alignof(type)),
                      "commondx type has stricter alignment requirement.");
    };

    template<>
    struct convert_to_cuda_type<commondx::complex<double>> {
        using type = cuDoubleComplex;
        static_assert((alignof(commondx::complex<double>) >= alignof(type)),
                      "commondx type has stricter alignment requirement.");
    };


    template<class T>
    using complex = commondx::complex<T>;
} // namespace cusolverdx

#endif // CUSOLVERDX_TYPES_HPP
