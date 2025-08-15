// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_DETAIL_CUTE_TENSOR_HPP
#define COMMONDX_DETAIL_CUTE_TENSOR_HPP

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wexpansion-to-defined"
#endif

#ifdef __clang__
#pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wunused-parameter"
// #pragma clang diagnostic ignored "-Wunknown-pragmas"
// #pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wexpansion-to-defined"
#endif

// Modify PTX instruction support scope
#if defined (CUTLASS_ARCH_MMA_SM100_ENABLED) || defined (CUTLASS_ARCH_MMA_SM103_ENABLED)
# define CUTE_ARCH_FFMA2_SM100_ENABLED
# define CUTE_ARCH_FLOAT2_MATH_ENABLED
#endif

#include <cute/tensor.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // COMMONDX_DETAIL_CUTE_TENSOR_HPP
