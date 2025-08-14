// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_PARTITION_HPP
#define CUBLASDX_DETAIL_BLAS_PARTITION_HPP

#include "cublasdx/database/cute.hpp"

namespace cublasdx {
    namespace detail {
        template<class T>
        struct type_wrapper {
            using type = T;
        };

        template<class TiledMMA, class LoadInstruction, class StoreInstruction, class ShapeMN, class InputTypeC, class Alignment, class HasStaticBlockDim, class BlockSize>
        struct blas_partitioner;

        template<class ... Args, class LoadInstruction, class StoreInstruction, class ShapeMN, class InputTypeC, class Alignment, class HasStaticBlockDim, class BlockSize>
        struct blas_partitioner<cute::TiledMMA<Args...>, LoadInstruction, StoreInstruction, ShapeMN, InputTypeC, Alignment, HasStaticBlockDim, BlockSize> {
            private:
            
            using this_type = blas_partitioner<cute::TiledMMA<Args...>, LoadInstruction, StoreInstruction, ShapeMN, InputTypeC, Alignment, HasStaticBlockDim, BlockSize>;
            // Empty 
            static constexpr cute::TiledMMA<Args...> tiled_mma = {};
            static constexpr LoadInstruction load_instruction = {};
            static constexpr StoreInstruction store_instruction = {};
            static constexpr ShapeMN shape_mn = {};
            static constexpr Alignment alignment = {};
            static constexpr HasStaticBlockDim has_static_block_dim = {};
            using thr_mma_t = decltype(tiled_mma.get_thread_slice(cute::declval<unsigned>()));
            using coord_tensor_t = decltype(cute::make_identity_tensor(ShapeMN{}));
            using coord_slice_t = decltype(cute::declval<thr_mma_t>().partition_C(coord_tensor_t{}));
            
            static constexpr bool is_partition_divisible =
                cute::evenly_divides(ShapeMN{}, cute::select<0, 1>(cute::tile_shape(tiled_mma)));

            // Dynamic: 
            // We could hold either 1 of them, and compute the other, 
            // or we could take thread idx from the user. There may be cases
            // where holding extra ints on stack are costly.
            unsigned thr_idx;
            thr_mma_t thr_mma;
    
            public:

            using value_type = InputTypeC;

            CUBLASDX_DEVICE
            blas_partitioner(unsigned thread_idx)
                : thr_idx(thread_idx), thr_mma(cute::TiledMMA<Args...>().get_thread_slice(thread_idx)) {};

            CUBLASDX_DEVICE
            constexpr auto is_predicated() const {
                return cute::conditional_return<is_partition_divisible>(cute::false_type{}, cute::true_type{});
            }

            CUBLASDX_DEVICE
            constexpr auto get_alignment() const {
                return alignment;
            }

            CUBLASDX_DEVICE
            auto is_thread_active() const {
                if constexpr(has_static_block_dim and (BlockSize::value == cute::size(tiled_mma))) {
                    return cute::true_type{};
                } else {
                    return thr_idx < cute::size(tiled_mma);
                }
                CUTE_GCC_UNREACHABLE;
            }

            template<class CTensor>
            CUBLASDX_DEVICE
            auto partition_like_C(CTensor && ctensor) const {
                return thr_mma.partition_C(ctensor);
            }

            CUBLASDX_DEVICE
            constexpr auto make_accumulator_fragment() const {
                return cute::make_tensor<InputTypeC>(cute::partition_shape_C(tiled_mma, shape_mn));
            }

            template<class ... Coords>
            CUBLASDX_DEVICE
            auto map_fragment_index(Coords&& ... coords) const {
                const coord_slice_t thr_coord = thr_mma.partition_C(coord_tensor_t{});
                return thr_coord(static_cast<Coords&&>(coords)...);
            }

            template<class ... Coords>
            CUBLASDX_DEVICE
            bool is_index_in_bounds(Coords&& ... coords) const {
                const coord_slice_t thr_coord = thr_mma.partition_C(coord_tensor_t{});
                return cute::elem_less(thr_coord(static_cast<Coords&&>(coords)...), ShapeMN{});
            }

            CUBLASDX_DEVICE
            auto make_tiled_load_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<LoadInstruction, InputTypeC>{}, thr_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }

            CUBLASDX_DEVICE
            auto make_tiled_store_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<StoreInstruction, InputTypeC>{}, thr_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }
        };
    }

    using cute::clear;
    using cute::transform;
    using cute::make_fragment_like;

    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE
    COMMONDX_STL_NAMESPACE::enable_if_t<
         cute::is_rmem_v<TRC> and
        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TRC, CFragLayout> const& tS,
                  tensor<TC, CLayout>           & tD,
                  Partitioner              const& p) {
        auto tPtD = p.partition_like_C(tD);

        using src_type = typename TRC::value_type;
        using dst_type = typename TC::value_type;
        using partitioner_value_type = typename Partitioner::value_type;
        constexpr bool GEMM_compliant_copy = cute::is_same_v<src_type, partitioner_value_type> and
                                             cute::is_same_v<dst_type, partitioner_value_type> and
                                             cute::is_smem_v<TC> and cute::is_rmem_v<TRC> and
                                             AlignmentInBytes == decltype(p.get_alignment())::value;

        using src_shape = decltype(tS.shape());
        using dst_shape = decltype(tPtD.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");

        auto predicated = p.is_predicated();

        if(p.is_thread_active()) {
            if constexpr(predicated) {
                cute::copy_if(cute::FunctionPredTensor([&](auto ... idx) { return p.is_index_in_bounds(idx ...); }), tS, tPtD);
            } else if constexpr(GEMM_compliant_copy) {
                auto [tiled_copy, thr_copy] = p.make_tiled_store_copy_c();
                cute::Tensor tCsD            = thr_copy.partition_D(tD);
                cute::Tensor tCrS_copy_view = thr_copy.retile_S(tS);
                CUTE_STATIC_ASSERT_V(cute::size<1>(tCsD) == cute::size<1>(tCrS_copy_view));             // CPY_M
                CUTE_STATIC_ASSERT_V(cute::size<2>(tCsD) == cute::size<2>(tCrS_copy_view));             // CPY_N
                copy(tiled_copy, tCrS_copy_view, tCsD);
            } else {
                constexpr int max_vec_bits = cute::gcd(AlignmentInBytes * 8, cute::max_alignment(CLayout{}) * cute::sizeof_bits_v<dst_type>);
                using tile_m_dimension = decltype(cute::select<1>(tPtD.layout()));
                using tile_n_dimension = decltype(cute::select<2>(tPtD.layout()));
                if constexpr(not cute::is_static<tile_m_dimension>::value
                             and cute::is_static<tile_n_dimension>::value) { // is row major
                    // Permute for vectorization, so that all static dimensions are in front
                    const auto permuted_ts = cute::make_tensor(tS.data(), cute::flatten(cute::select<0, 2, 1>(tS.layout())));
                    auto permuted_tptd = cute::make_tensor(tPtD.data(), cute::flatten(cute::select<0, 2, 1>(tPtD.layout())));
                    cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits>{}, permuted_ts, permuted_tptd);
                } else {
                    const auto flat_ts = cute::make_tensor(tS.data(), cute::flatten(tS.layout()));
                    auto flat_tptd = cute::make_tensor(tPtD.data(), cute::flatten(tPtD.layout()));
                    cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits>{}, flat_ts, flat_tptd);
                }
            }
        }
    }


    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE
    COMMONDX_STL_NAMESPACE::enable_if_t<
         cute::is_rmem_v<TRC> and
        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TC, CLayout>      const& tS,
                  tensor<TRC, CFragLayout>      & tD,
                  Partitioner              const& p) {
        auto tPtS = p.partition_like_C(tS);

        using src_type = typename TC::value_type;
        using dst_type = typename TRC::value_type;
        using partitioner_value_type = typename Partitioner::value_type;
        constexpr bool GEMM_compliant_copy = cute::is_same_v<src_type, partitioner_value_type> and
                                             cute::is_same_v<dst_type, partitioner_value_type> and
                                             cute::is_smem_v<TC> and cute::is_rmem_v<TRC> and
                                             AlignmentInBytes == decltype(p.get_alignment())::value;

        using src_shape = decltype(tS.shape());
        using dst_shape = decltype(tPtS.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");

        auto predicated = p.is_predicated();

        if(p.is_thread_active()) {
            if constexpr(predicated) {
                cute::copy_if(cute::FunctionPredTensor([&](auto ... idx) { return p.is_index_in_bounds(idx ...); }), tPtS, tD);
            } else if constexpr(GEMM_compliant_copy) {
                auto [tiled_copy, thr_copy] = p.make_tiled_load_copy_c();
                cute::Tensor tCsS           = thr_copy.partition_S(tS);
                cute::Tensor tCrD_copy_view = thr_copy.retile_D(tD);
                CUTE_STATIC_ASSERT_V(cute::size<1>(tCsS) == cute::size<1>(tCrD_copy_view));             // CPY_M
                CUTE_STATIC_ASSERT_V(cute::size<2>(tCsS) == cute::size<2>(tCrD_copy_view));             // CPY_N
                copy(tiled_copy, tCsS, tCrD_copy_view);
            } else {
                constexpr int max_vec_bits = cute::gcd(AlignmentInBytes * 8, cute::max_alignment(CLayout{}) * cute::sizeof_bits_v<src_type>);
                using tile_m_dimension = decltype(cute::select<1>(tPtS.layout()));
                using tile_n_dimension = decltype(cute::select<2>(tPtS.layout()));

                if constexpr(not cute::is_static<tile_m_dimension>::value
                             and cute::is_static<tile_n_dimension>::value) { // is row major
                    // Permute for vectorization, put all static dimensions in the front
                    const auto permuted_tpts = cute::make_tensor(tPtS.data(), cute::flatten(cute::select<0, 2, 1>(tPtS.layout())));
                    auto permuted_td = cute::make_tensor(tD.data(), cute::flatten(cute::select<0, 2, 1>(tD.layout())));
                    cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits>{}, permuted_tpts, permuted_td);
                } else {
                    const auto flat_tpts = cute::make_tensor(tPtS.data(), cute::flatten(tPtS.layout()));
                    auto flat_td = cute::make_tensor(tD.data(), cute::flatten(tD.layout()));
                    cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits>{}, flat_tpts, flat_td);
                }
            }
        }
    }

    namespace detail {
        template<class Beta>
        CUBLASDX_HOST_DEVICE constexpr
        bool is_zero(Beta beta) {
            if constexpr (cutlass::is_complex<Beta>::value) {
                using vt = typename Beta::value_type;
                const auto zero = static_cast<vt>(0.f);
                return beta.real() == zero && beta.imag() == zero;
            }
            else {
                const auto zero = static_cast<Beta>(0.f);
                return beta == zero;
            }
            CUTE_GCC_UNREACHABLE;
        }

        template<class Alpha,
                class XEngine, class XLayout,
                class Beta,
                class YEngine, class YLayout>
        CUBLASDX_HOST_DEVICE void
        axpby_impl(Alpha                          const& alpha,
                   cute::Tensor<XEngine, XLayout> const& x_tensor,
                   Beta                           const& beta,
                   cute::Tensor<YEngine, YLayout>      & y_tensor) {
            if(is_zero(beta)) {
                CUTE_UNROLL
                for(int i = 0; i < cute::size(y_tensor); ++i) {
                    y_tensor(i) = alpha * x_tensor(i);
                }
            } else {
                CUTE_UNROLL
                for(int i = 0; i < cute::size(y_tensor); ++i) {
                    y_tensor(i) = alpha * x_tensor(i) + beta * y_tensor(i);
                }
            }
        }

        // Accept mutable temporaries
        template<class Alpha,
                class XEngine, class XLayout,
                class Beta,
                class YEngine, class YLayout>
        CUBLASDX_HOST_DEVICE void
        axpby_impl(Alpha                          const&  alpha,
                   cute::Tensor<XEngine, XLayout> const&  x_tensor,
                   Beta                           const&  beta,
                   cute::Tensor<YEngine, YLayout>      && y_tensor) {
            axpby_impl(alpha, x_tensor, beta, y_tensor);
        }
    }

    template<class Alpha,
             class XEngine, class XLayout,
             class Beta,
             class YEngine, class YLayout>
    CUBLASDX_HOST_DEVICE void
    axpby(Alpha                          const& alpha,
          cute::Tensor<XEngine, XLayout> const& x_tensor,
          Beta                           const& beta,
          cute::Tensor<YEngine, YLayout>      & y_tensor) {

        using x_value_type = typename XEngine::value_type;
        using y_value_type = typename YEngine::value_type;

        static_assert(sizeof(Alpha) == sizeof(x_value_type) and alignof(Alpha) == alignof(x_value_type));
        static_assert(sizeof(Beta) == sizeof(y_value_type) and alignof(Beta) == alignof(y_value_type));

        detail::axpby_impl(
            reinterpret_cast<detail::convert_to_cutlass_type_t<x_value_type>const&>(alpha),
            cute::recast<detail::convert_to_cutlass_type_t<x_value_type>>(x_tensor),
            reinterpret_cast<detail::convert_to_cutlass_type_t<y_value_type>const&>(beta),
            cute::recast<detail::convert_to_cutlass_type_t<y_value_type>>(y_tensor)
        );
    }

    // Accept mutable temporaries
    template<class Alpha,
             class XEngine, class XLayout,
             class Beta,
             class YEngine, class YLayout>
    CUBLASDX_HOST_DEVICE void
    axpby(Alpha                          const&  alpha,
          cute::Tensor<XEngine, XLayout> const&  x_tensor,
          Beta                           const&  beta,
          cute::Tensor<YEngine, YLayout>      && y_tensor) {
        cublasdx::axpby(alpha, x_tensor, beta, y_tensor);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_PARTITION_HPP
