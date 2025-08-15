// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_FUNCTIONAL_HPP
#define CUSOLVERDX_DETAIL_FUNCTIONAL_HPP

namespace cusolverdx {

    struct identity {
        template<typename T>
        __host__ __device__ __forceinline__ T operator()(const T& v) const {
            return v;
        }
    };

    template<typename Precision, int Num, int Den>
    struct rational_scaler {
        static constexpr Precision value = Precision {Num} / Precision {Den};

        template<typename T>
        __host__ __device__ __forceinline__ T operator()(const T& v) const {
            return value * v;
        }

        template<typename T>
        __host__ __device__ __forceinline__ complex<T> operator()(const complex<T>& v) const {
            return {value * v.real(), value * v.imag()};
        }
    };

} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_COPY_HPP
