// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

namespace cublasdx {
    namespace detail {
        using cute::Layout;
        using cute::Shape;
        using cute::Stride;
        using cute::Int;
        using cute::Tensor;

        // General FMA
        template <class A, class B, class C>
        CUTE_HOST_DEVICE constexpr
        void
        fma(C& d, A const& a, B const& b, C const& c)
        {
            if constexpr (cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e4m3_t, float_e4m3_t, double>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e5m2_t, float_e5m2_t, double>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e4m3_t, float_e4m3_t, float>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e5m2_t, float_e5m2_t, float>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e4m3_t, float_e4m3_t, half_t>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e5m2_t, float_e5m2_t, half_t>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e4m3_t, float_e4m3_t, bfloat16_t>> ||
                          cute::is_same_v<cute::tuple<A, B, C>, cute::tuple<float_e5m2_t, float_e5m2_t, bfloat16_t>>) {
                d = static_cast<C>(static_cast<float>(a) * static_cast<float>(b) + c);
            } else if constexpr(commondx::is_integral_v<A>) {
                d = static_cast<C>(a) * static_cast<C>(b) + c;
            } else {
                using cute::fma;
                fma(d, a, b, c);
            }
        }

        template <class A, class B, class C>
        CUTE_HOST_DEVICE constexpr
        void
        fma(cutlass::complex<C>      & d,
            cutlass::complex<A> const& a,
            cutlass::complex<B> const& b,
            cutlass::complex<C> const& c)
        {
            fma(d.real(),  a.real(), b.real(), c.real());
            fma(d.imag(),  a.real(), b.imag(), c.imag());
            // NVCC produces incorrect code for int8/int8/int64 dynamic LD GEMMs
            if constexpr(commondx::is_integral_v<A>) {
                fma(d.real(),  static_cast<A>(-a.imag()), b.imag(), d.real());
            } else {
                fma(d.real(),                 -a.imag(),  b.imag(), d.real());
            }
            fma(d.imag(),  a.imag(), b.real(), d.imag());
        }

        // Universal FMA
        template <class A, class B, class C>
        struct UniversalFMA
        {
            using DRegisters = C[1];
            using ARegisters = A[1];
            using BRegisters = B[1];
            using CRegisters = C[1];

            CUTE_HOST_DEVICE static constexpr void
            fma(C      & d,
                A const& a,
                B const& b,
                C const& c)
            {
                using cublasdx::detail::fma;
                fma(d, a, b, c);
            }
        };

    }
}

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP
