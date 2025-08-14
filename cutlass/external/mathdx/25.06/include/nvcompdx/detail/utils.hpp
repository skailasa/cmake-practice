// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

namespace nvcompdx::detail {

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const unit)
{
    return (num + unit - 1) / unit;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const unit)
{
    return roundUpDiv(num, unit) * unit;
}

template<typename T>
constexpr __host__ __device__ T roundUpPow2(const T x)
{
  size_t res = 1;
  while(res < x) {
    res *= 2;
  }
  return res;
}

template<typename T>
constexpr __host__ __device__ T min(const T& a, const T& b) {
  return a < b ? a : b;
}

template<typename T>
constexpr __host__ __device__ T max(const T& a, const T& b) {
  return a < b ? b : a;
}

} // namespace nvcompdx::detail
