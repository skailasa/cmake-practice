// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_HPP
#define CUBLASDX_DATABASE_CUTE_HPP

#include "cublasdx/database/cute_tensor.hpp"
#include "cublasdx/database/cute_db.hpp"
#include "cublasdx/database/cute_utils.hpp"
#include "cublasdx/database/suggested_layouts.hpp"
#include "cublasdx/detail/blas_partition.hpp"

namespace cublasdx::detail {
    template<class BlockSize>
    CUBLASDX_DEVICE static constexpr unsigned get_threads() {
        unsigned ret = 128;
        if constexpr(not cute::is_void_v<BlockSize>) {
            ret = BlockSize::flat_size;
        }
        return ret;
    }

    template<class TiledMMA>
    CUBLASDX_DEVICE static constexpr unsigned get_mma_threads() {
        unsigned ret = 0;
        if constexpr(not cute::is_void_v<TiledMMA>) {
            ret = cute::size(TiledMMA{});
        }
        return ret;
    }

    template<class BlockSize>
    CUBLASDX_DEVICE static constexpr unsigned get_block_rank() {
        unsigned ret = 1;
        if constexpr(not cute::is_void_v<BlockSize>) {
            ret = BlockSize::rank;
        }
        return ret;
    }

    template<int BlockRank>
    CUBLASDX_DEVICE static unsigned int get_thread_idx() {
        constexpr int block_rank = BlockRank;
        if constexpr (block_rank == 3) {
            return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        } else if constexpr (block_rank == 2) {
            return threadIdx.x + threadIdx.y * blockDim.x;
        } else {
            return threadIdx.x;
        }
    }

    template<class BlockSize>
    CUBLASDX_DEVICE static unsigned int get_thread_idx() {
        constexpr int block_rank = get_block_rank<BlockSize>();
        return get_thread_idx<block_rank>();
    }

    namespace cute_backend {
        template<typename TypeA,
                 typename TypeB,
                 typename TypeC,
                 typename InputA,
                 typename InputB,
                 typename InputC,
                 typename Alignment,
                 int SizeM,
                 int SizeN,
                 int SizeK,
                 typename Arrangement,
                 typename TransposeMode,
                 typename SM,
                 class HasStaticBlockDim,
                 typename BlockSize, // void if empty
                 typename OverloadedTileOperator>
        struct execution {
            private:
            static constexpr bool is_tile_overloaded = OverloadedTileOperator::valid;
            
            static constexpr int blocksize_threads = cute::size(get_database_threads_t<TypeA, TypeB, TypeC, SizeM, SizeN, SizeK, SM, BlockSize>{});
            static constexpr int overloaded_threads = cute::size(get_database_threads_t<TypeA, TypeB, TypeC, SizeM, SizeN, SizeK, SM, BlockSize, OverloadedTileOperator>{});
            static_assert(blocksize_threads == overloaded_threads);

            // Necessary only for pointer API
            using blas_transpose_mode = TransposeMode;
            static constexpr auto tr_mode_a = blas_transpose_mode::a_transpose_mode;
            static constexpr auto tr_mode_b = blas_transpose_mode::b_transpose_mode;
            static constexpr auto tr_mode_c = transpose_mode::non_transposed;

            using blas_blockdim = BlockSize;

            // Necessary only for pointer API
            using blas_arrangement = Arrangement;
            static constexpr auto arr_a = blas_arrangement::a;
            static constexpr auto arr_b = blas_arrangement::b;
            static constexpr auto arr_c = blas_arrangement::c;

            using blas_alignment = Alignment;
            static constexpr auto align_a = blas_alignment::a;
            static constexpr auto align_b = blas_alignment::b;
            static constexpr auto align_c = blas_alignment::c;

            // These are "safe" because they can always be passed to cute::copy, not the case
            // with e.g. LDSM
            using safe_a_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_a * 8>;
            using safe_b_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_b * 8>;
            using safe_c_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_c * 8>;

            // Necessary for both APIs
            static constexpr unsigned int m = SizeM;
            static constexpr unsigned int n = SizeN;
            static constexpr unsigned int k = SizeK;

            static constexpr auto has_static_block_dim = HasStaticBlockDim{};

            static constexpr int block_size = get_threads<BlockSize>();
            using swizzled_meta_info = cublasdx::detail::layout_database::optimal_config<
                block_size, SM::value,
                TypeA, arr_a == arrangement::col_major, align_a,
                TypeB, arr_b == arrangement::col_major, align_b,
                TypeC, arr_c == arrangement::col_major, align_c,
                m, n, k>;

            using swizzled_config    = typename swizzled_meta_info::tiled_mma;
            using swizzled_a_layout  = typename swizzled_meta_info::a_layout;
            using swizzled_b_layout  = typename swizzled_meta_info::b_layout;
            using swizzled_c_layout  = typename swizzled_meta_info::c_layout;
            using swizzled_a_copy_op = typename swizzled_meta_info::a_copy_op;
            using swizzled_b_copy_op = typename swizzled_meta_info::b_copy_op;
            using swizzled_c_copy_load_op = typename swizzled_meta_info::c_copy_load_op;
            using swizzled_c_copy_store_op = typename swizzled_meta_info::c_copy_store_op;

            static constexpr bool is_swizzled_config_viable = not is_tile_overloaded and swizzled_meta_info::valid;

            // This is necessary for a case where swizzled config is available but the user
            // does not utilize suggested layout and db_entry has higher number of threads
            // than 128
            using db_config_meta = get_database_config<TypeA, TypeB, TypeC, m, n, k, arr_a, arr_b, arr_c, align_a, align_b, align_c, SM, BlockSize, OverloadedTileOperator>;
            using db_config = typename db_config_meta::type;
            using db_a_copy_op = typename db_config_meta::a_copy_op;
            using db_b_copy_op = typename db_config_meta::b_copy_op;
            using db_c_copy_load_op = typename db_config_meta::c_copy_load_op;
            using db_c_copy_store_op = typename db_config_meta::c_copy_store_op;

            using default_config = db_config;
            using default_a_copy_op = db_a_copy_op;
            using default_b_copy_op = db_b_copy_op;
            using default_c_copy_load_op = db_c_copy_load_op;
            using default_c_copy_store_op = db_c_copy_store_op;

            using suggested_threads_config = cute::conditional_t<is_swizzled_config_viable, swizzled_config, default_config>;

            public:
            // Helper traits for knowing fragment type upfront
            using c_frag_t = decltype(cute::partition_fragment_C(default_config{}, cute::Shape<Int<SizeM>, Int<SizeN>>{}));
            using c_frag_suggested_t = decltype(cute::partition_fragment_C(suggested_threads_config{}, cute::Shape<Int<SizeM>, Int<SizeN>>{}));

            // If blocksize is not specified, this one will be used
            // it should be the most performant block size for this problem
            static constexpr unsigned int suggested_threads = cute::size(suggested_threads_config{});

            // Partitioner getters
            CUBLASDX_DEVICE static auto get_partitioner() {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();
                using shape_mn_t = cute::Shape<cute::Int<SizeM>, cute::Int<SizeN>>;
                return blas_partitioner<default_config, default_c_copy_load_op, default_c_copy_store_op, shape_mn_t, InputC, cute::Int<align_c>, HasStaticBlockDim, cute::Int<blocksize_threads>>(thread_idx);
            }

            CUBLASDX_DEVICE static auto suggest_partitioner() {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();
                if constexpr(is_swizzled_config_viable) {
                    using shape_mn_t = cute::Shape<cute::Int<SizeM>, cute::Int<SizeN>>;
                    return blas_partitioner<swizzled_config, swizzled_c_copy_load_op, swizzled_c_copy_store_op, shape_mn_t, InputC, cute::Int<align_c>, HasStaticBlockDim, cute::Int<blocksize_threads>>(thread_idx);
                } else {
                    return get_partitioner();
                }
            }

            template<typename ALayout, typename BLayout>
            CUBLASDX_DEVICE static auto get_partitioner(ALayout, BLayout) {
                constexpr bool is_suggested_mma =
                    is_swizzled_config_viable and
                    cute::is_same_v<ALayout, swizzled_a_layout> and
                    cute::is_same_v<BLayout, swizzled_b_layout>;

                if constexpr(is_suggested_mma) {
                    return suggest_partitioner();
                } else {
                    return get_partitioner();
                }

                CUTE_GCC_UNREACHABLE;
            }


            // C in registers API
            template<typename TSA,
                    typename ALayout,
                    typename TSB,
                    typename BLayout,
                    typename TRC,
                    typename CLayout,
                    typename ALoadOp = identity,
                    typename BLoadOp = identity,
                    __CUTE_REQUIRES(cute::is_smem_v<TSA> and cute::is_smem_v<TSB> and cute::is_rmem_v<TRC>)>
            CUBLASDX_DEVICE static void tensor_gemm(cute::Tensor<TSA, ALayout> const& smem_tensor_a,
                                                    cute::Tensor<TSB, BLayout> const& smem_tensor_b,
                                                    cute::Tensor<TRC, CLayout>      & rmem_tensor_c,
                                                    const ALoadOp&                    a_load_op = identity {},
                                                    const BLoadOp&                    b_load_op = identity {}) {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();

                static_assert(cute::is_same_v<CLayout, decltype(c_frag_suggested_t().layout())> or
                              cute::is_same_v<CLayout, decltype(c_frag_t().layout())>,
                              "Incompatible C fragment type used");

                constexpr bool is_suggested_mma =
                    is_swizzled_config_viable and
                    cute::is_same_v<CLayout, decltype(c_frag_suggested_t().layout())>;

                // Check if decoupled precision was used for A
                constexpr int a_layout_alignment = cute::is_static_v<ALayout> ? cute::gcd(align_a, cute::max_alignment(ALayout{}) * sizeof(TypeA)) : sizeof(TypeA);
                constexpr bool is_type_copy_compatible_a = 
                    cute::is_static_v<ALayout> and (a_layout_alignment == align_a) and
                    (sizeof(typename TSA::value_type) == sizeof(TypeA) and alignof(typename TSA::value_type) == alignof(TypeA));

                // Check if decoupled precision was used for B
                constexpr int b_layout_alignment = cute::is_static_v<BLayout> ? cute::gcd(align_b, cute::max_alignment(BLayout{}) * sizeof(TypeB)) : sizeof(TypeB);
                constexpr bool is_type_copy_compatible_b = 
                    cute::is_static_v<BLayout> and (b_layout_alignment == align_b) and
                    (sizeof(typename TSB::value_type) == sizeof(TypeB) and alignof(typename TSB::value_type) == alignof(TypeB));

                constexpr bool is_suggested_copy =
                    is_suggested_mma and
                    cute::is_same_v<ALayout, swizzled_a_layout> and 
                    cute::is_same_v<BLayout, swizzled_b_layout>;

                auto tiled_mma = cute::conditional_t<is_suggested_mma, swizzled_config, default_config>{};
                using a_fast_copy_op = cute::conditional_t<is_suggested_copy, swizzled_a_copy_op, default_a_copy_op>;
                using a_vec_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<a_layout_alignment * 8>;
                auto a_copy_op = cute::conditional_t<is_type_copy_compatible_a, a_fast_copy_op, a_vec_copy_op>{};
                using b_fast_copy_op = cute::conditional_t<is_suggested_copy, swizzled_b_copy_op, default_b_copy_op>;
                using b_vec_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<b_layout_alignment * 8>;
                auto b_copy_op = cute::conditional_t<is_type_copy_compatible_b, b_fast_copy_op, b_vec_copy_op>{};

                if ((has_static_block_dim and blocksize_threads == cute::size(tiled_mma)) or (thread_idx < cute::size(tiled_mma))) {
                    cute::cooperative_gemm(thread_idx,
                                           tiled_mma,
                                           smem_tensor_a,
                                           swap_tensor_modes(smem_tensor_b),
                                           rmem_tensor_c,
                                           a_load_op,
                                           b_load_op,
                                           a_copy_op,
                                           b_copy_op);
                }
            }

            // Accept mutable temporaries
            template<typename TSA,
                    typename ALayout,
                    typename TSB,
                    typename BLayout,
                    typename TRC,
                    typename CLayout,
                    typename ALoadOp = identity,
                    typename BLoadOp = identity,
                    __CUTE_REQUIRES(cute::is_smem_v<TSA> and cute::is_smem_v<TSB> and cute::is_rmem_v<TRC>)>
            CUBLASDX_DEVICE static void tensor_gemm(cute::Tensor<TSA, ALayout> const& smem_tensor_a,
                                                    cute::Tensor<TSB, BLayout> const& smem_tensor_b,
                                                    cute::Tensor<TRC, CLayout>     && rmem_tensor_c,
                                                    const ALoadOp&                    a_load_op = identity {},
                                                    const BLoadOp&                    b_load_op = identity {}) {
                tensor_gemm(smem_tensor_a, smem_tensor_b, rmem_tensor_c, a_load_op, b_load_op);
            }

            // This operates on assumption (checked in BLAS.execute()) that tensor sizes agree with operator sizes
            template<typename TA,
                     typename ALayout,
                     typename TB,
                     typename BLayout,
                     typename TC,
                     typename CLayout,
                     typename Alpha,
                     typename Beta,
                     typename ALoadOp = identity,
                     typename BLoadOp = identity,
                     typename CLoadOp = identity,
                     typename CStoreOp = identity,
                     __CUTE_REQUIRES(cute::is_smem_v<TA> and cute::is_smem_v<TB> and cute::is_smem_v<TC>)>
            CUBLASDX_DEVICE static void tensor_gemm(cute::Tensor<TA, ALayout> const& smem_tensor_a,
                                                    cute::Tensor<TB, BLayout> const& smem_tensor_b,
                                                    cute::Tensor<TC, CLayout>      & smem_tensor_c,
                                                    Alpha                            alpha,
                                                    Beta                             beta,
                                                    const ALoadOp&                   a_load_op  = identity {},
                                                    const BLoadOp&                   b_load_op  = identity {},
                                                    const CLoadOp&                   c_load_op  = identity {},
                                                    const CStoreOp&                  c_store_op = identity {}) {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();

                constexpr bool is_suggested =
                    is_swizzled_config_viable and
                    cute::is_same_v<ALayout, swizzled_a_layout> and
                    cute::is_same_v<BLayout, swizzled_b_layout> and
                    cute::is_same_v<CLayout, swizzled_c_layout>;

                // Check if decoupled precision was used for A
                constexpr int a_layout_alignment = cute::is_static_v<ALayout> ? cute::gcd(align_a, cute::max_alignment(ALayout{}) * sizeof(TypeA)) : sizeof(TypeA);
                constexpr bool is_type_copy_compatible_a = 
                    cute::is_static_v<ALayout> and (a_layout_alignment == align_a) and
                    (sizeof(typename TA::value_type) == sizeof(TypeA) and alignof(typename TA::value_type) == alignof(TypeA));

                // Check if decoupled precision was used for B
                constexpr int b_layout_alignment = cute::is_static_v<BLayout> ? cute::gcd(align_b, cute::max_alignment(BLayout{}) * sizeof(TypeB)) : sizeof(TypeB);
                constexpr bool is_type_copy_compatible_b = 
                    cute::is_static_v<BLayout> and (b_layout_alignment == align_b) and
                    (sizeof(typename TB::value_type) == sizeof(TypeB) and alignof(typename TB::value_type) == alignof(TypeB));

                constexpr int c_layout_alignment = cute::is_static_v<CLayout> ? cute::gcd(align_c, cute::max_alignment(CLayout{}) * sizeof(TypeC)) : sizeof(TypeC);
                constexpr bool is_type_copy_compatible_c =
                    cute::is_static_v<CLayout> and (c_layout_alignment == align_c) and
                    (sizeof(typename TC::value_type) == sizeof(TypeC) and alignof(typename TC::value_type) == alignof(TypeC));

                auto tiled_mma = cute::conditional_t<is_suggested, swizzled_config, default_config>{};
                using a_fast_copy_op = cute::conditional_t<is_suggested, swizzled_a_copy_op, default_a_copy_op>;
                using a_vec_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<a_layout_alignment * 8>;
                auto a_copy_op = cute::conditional_t<is_type_copy_compatible_a, a_fast_copy_op, a_vec_copy_op>{};
                using b_fast_copy_op = cute::conditional_t<is_suggested, swizzled_b_copy_op, default_b_copy_op>;
                using b_vec_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<b_layout_alignment * 8>;
                auto b_copy_op = cute::conditional_t<is_type_copy_compatible_b, b_fast_copy_op, b_vec_copy_op>{};
                using c_vec_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<c_layout_alignment * 8>;
                using c_fast_copy_load_op = cute::conditional_t<is_suggested, swizzled_c_copy_load_op, default_c_copy_load_op>;
                auto c_copy_load_op = cute::conditional_t<is_type_copy_compatible_c, c_fast_copy_load_op, c_vec_copy_op>{};
                using c_fast_copy_store_op = cute::conditional_t<is_suggested, swizzled_c_copy_store_op, default_c_copy_store_op>;
                auto c_copy_store_op = cute::conditional_t<is_type_copy_compatible_c, c_fast_copy_store_op, c_vec_copy_op>{};
                
                if ((has_static_block_dim and blocksize_threads == cute::size(tiled_mma)) or (thread_idx < cute::size(tiled_mma))) {
                    cute::cooperative_gemm(thread_idx,
                                           tiled_mma,
                                           alpha,
                                           smem_tensor_a,
                                           swap_tensor_modes(smem_tensor_b),
                                           beta,
                                           smem_tensor_c,
                                           a_load_op,
                                           b_load_op,
                                           c_load_op,
                                           c_store_op,
                                           a_copy_op,
                                           b_copy_op,
                                           c_copy_load_op,
                                           c_copy_store_op);
                }
            }

            // Accept mutable temporaries
            template<typename TA,
                     typename ALayout,
                     typename TB,
                     typename BLayout,
                     typename TC,
                     typename CLayout,
                     typename Alpha,
                     typename Beta,
                     typename ALoadOp = identity,
                     typename BLoadOp = identity,
                     typename CLoadOp = identity,
                     typename CStoreOp = identity,
                     __CUTE_REQUIRES(cute::is_smem_v<TA> and cute::is_smem_v<TB> and cute::is_smem_v<TC>)>
            CUBLASDX_DEVICE static void tensor_gemm(cute::Tensor<TA, ALayout> const& smem_tensor_a,
                                                               cute::Tensor<TB, BLayout> const& smem_tensor_b,
                                                               cute::Tensor<TC, CLayout>     && smem_tensor_c,
                                                               Alpha                            alpha,
                                                               Beta                             beta,
                                                               const ALoadOp&                   a_load_op  = identity {},
                                                               const BLoadOp&                   b_load_op  = identity {},
                                                               const CLoadOp&                   c_load_op  = identity {},
                                                               const CStoreOp&                  c_store_op = identity {}) {
                tensor_gemm(smem_tensor_a, smem_tensor_b, smem_tensor_c, alpha, beta, a_load_op, b_load_op, c_load_op, c_store_op);
            }
        };
    } // namespace cute_backend
} // namespace cublasdx::detail

#endif // CUBLASDX_DATABASE_CUTE_HPP
