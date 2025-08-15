// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_COPY_HPP
#define CUBLASDX_DETAIL_COPY_HPP

#include "cublasdx/database/cute.hpp"

namespace cublasdx {
    CUBLASDX_DEVICE
    void copy_wait() {
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
    }

    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE
    void copy(const unsigned int                            tid,
              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                      "cublasdx::copy requires static tensor layouts");

        using src_elem_with_const = cute::remove_reference_t<typename SrcEngine::reference>;
        constexpr bool is_const_source = cute::is_const_v<src_elem_with_const>;
        constexpr auto copy_policy = cute::conditional_return<is_const_source>(cublasdx::detail::auto_copy_async_cache_global_cublasdx{}, 
                                                                               cublasdx::detail::auto_copy_async_cache_always_cublasdx{});

        constexpr int max_vec_bits = cute::gcd(AlignmentInBytes * 8, 
                                               cute::max_alignment(SrcLayout{}) * cute::sizeof_bits_v<typename SrcEngine::value_type>,
                                               cute::max_alignment(DstLayout{}) * cute::sizeof_bits_v<typename DstEngine::value_type>);
        
        if(tid < NumThreads) {
            return cute::cooperative_copy<NumThreads, max_vec_bits>(tid, src, dst, copy_policy);
        }
    }

    template<uint32_t NumThreads,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE
    void copy(const unsigned int                            tid,
              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        constexpr unsigned int max_vec_bits = cute::sizeof_bits_v<typename SrcEngine::value_type>;
        copy<NumThreads, max_vec_bits>(tid, src, dst);
    }

    // This overload uses as many threads as defined in BLAS::block_dim
    template<class BLAS, uint32_t AlignmentInBytes, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUBLASDX_DEVICE
    void copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");
        constexpr unsigned int num_threads = BLAS::block_dim.x * BLAS::block_dim.y * BLAS::block_dim.z;
        constexpr unsigned int block_dim_rank = (BLAS::block_dim.z > 1) ? 3 : ((BLAS::block_dim.y > 1) ? 2 : 1);
        unsigned int tid = detail::get_thread_idx<block_dim_rank>();
        copy<num_threads, AlignmentInBytes>(tid, src, dst);
    }

    // Allow mutable temporaries
    template<class BLAS, uint32_t AlignmentInBytes, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUBLASDX_DEVICE
    void copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&&      dst) {
        copy<BLAS, AlignmentInBytes>(src, dst);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_COPY_HPP
