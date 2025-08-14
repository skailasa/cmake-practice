// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "cublasdx/detail/blas_description.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/database/cute.hpp"
#include "cublasdx/database/suggested_layouts.hpp"
#include "cublasdx/traits.hpp"

namespace cublasdx {
    namespace detail {

        template<class Functor, class ... Args>
        struct execute_invoke_result {
            using type = COMMONDX_STL_NAMESPACE::invoke_result_t<Functor, Args...>;
        };

        template<class Engine, class Layout, class ... Args>
        struct execute_invoke_result<cute::Tensor<Engine, Layout>, Args...> {
            // Could be left empty to exploit SFINAE
            using type = void;
        };

        template<class BlasBlockExecution, typename = void>
        class layout_type_provider { };

        template<class BlasBlockExecution>
        class layout_type_provider<BlasBlockExecution, COMMONDX_STL_NAMESPACE::enable_if_t<is_complete_blas_execution<BlasBlockExecution>::value>> {
            protected:
            using default_static_layout_a = decltype(BlasBlockExecution::template get_cute_layout<matrix::A>());
            using default_static_layout_b = decltype(BlasBlockExecution::template get_cute_layout<matrix::B>());
            using default_static_layout_c = decltype(BlasBlockExecution::template get_cute_layout<matrix::C>());
            using default_layout_rmem_c = typename BlasBlockExecution::gemm_backend::c_frag_t;

            using suggested_layout_smem_a = decltype(BlasBlockExecution::template suggest_cute_layout_smem<matrix::A>());
            using suggested_layout_smem_b = decltype(BlasBlockExecution::template suggest_cute_layout_smem<matrix::B>());
            using suggested_layout_smem_c = decltype(BlasBlockExecution::template suggest_cute_layout_smem<matrix::C>());
            using suggested_layout_rmem_c = typename BlasBlockExecution::gemm_backend::c_frag_suggested_t;
        };

        template<class TypeTuple>
        static constexpr bool CUBLASDX_HOST_DEVICE
        is_type_pair_compatible() {
            using TA = typename COMMONDX_STL_NAMESPACE::tuple_element<0, TypeTuple>::type;
            using TB = typename COMMONDX_STL_NAMESPACE::tuple_element<1, TypeTuple>::type;

            // sizeof(void) is illegal, so it needs to be avoided
            if constexpr(COMMONDX_STL_NAMESPACE::is_void_v<TA> or (COMMONDX_STL_NAMESPACE::is_void_v<TB>)) {
                return false;
            } else {
                return ((sizeof(TA)  == sizeof(TB)) && (alignof(TA) == alignof(TB))) || cute::is_convertible_v<TB, TA>;
            }

            CUTE_GCC_UNREACHABLE;
        }

        template<typename ... TypeTuples>
        static constexpr bool CUBLASDX_HOST_DEVICE
        are_types_compatible_impl() {
            return (is_type_pair_compatible<TypeTuples>() && ...);
        }

        template<class... Operators>
        class blas_execution: public blas_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type = blas_description<Operators...>;
            using this_type = blas_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_blas_precision;

            /// ---- Constraints

            // We need Block operator to be specified exactly once
            static constexpr bool has_one_block = has_at_most_one_of<operator_type::block, this_type>::value;
            static_assert(has_one_block, "Can't create blas function with two execution operators");
        };

        template<class value_type>
        struct cutlass_value_type {
            using a_type = convert_to_cutlass_type_t<typename value_type::a_type>;
            using b_type = convert_to_cutlass_type_t<typename value_type::b_type>;
            using c_type = convert_to_cutlass_type_t<typename value_type::c_type>;
        };

        template<typename MemOp, typename Input, typename Output>
        CUBLASDX_HOST_DEVICE
        constexpr bool is_functor_compatible() {
            using invoke_result =
                COMMONDX_STL_NAMESPACE::decay_t<
                    COMMONDX_STL_NAMESPACE::invoke_result_t<
                        MemOp, Input>>;
            return (((sizeof(invoke_result) == sizeof(Output)) && (alignof(invoke_result) == alignof(Output))) || cute::is_convertible_v<invoke_result, Output>);
        }

        template<class trans_op, class default_input_type>
        struct transform_op_wrapper {
            trans_op op;

            using result_t = COMMONDX_STL_NAMESPACE::decay_t<typename execute_invoke_result<trans_op, default_input_type>::type>;
            using cutlass_result_t = convert_to_cutlass_type_t<result_t>;

            CUBLASDX_HOST_DEVICE
            auto default_call(default_input_type const arg) const {
                auto result = op(arg);
                return cast_to_cutlass_type<cutlass_result_t>(result);
            }

            template<typename T>
            CUBLASDX_HOST_DEVICE
            auto operator()(const T arg) const {
                // Is invokable with CUTLASS type
                if constexpr (COMMONDX_STL_NAMESPACE::is_invocable_v<trans_op, T>) {
                    using cutlass_input_result_t = COMMONDX_STL_NAMESPACE::decay_t<typename execute_invoke_result<trans_op, T>::type>;
                    using cutlass_result_t = convert_to_cutlass_type_t<cutlass_input_result_t>;
                    if constexpr(COMMONDX_STL_NAMESPACE::is_convertible_v<cutlass_input_result_t, cutlass_result_t>) {
                        return static_cast<cutlass_result_t>(op(arg));
                    } else {
                        return default_call(cast_from_cutlass_type<default_input_type>(arg));
                    }
                } else {
                    return default_call(cast_from_cutlass_type<default_input_type>(arg));
                }
            }
        };

#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
        template<class... Operators>
        class blas_block_execution_partial: public blas_execution<Operators...>
        {
            using base_type = blas_execution<Operators...>;
            using typename base_type::this_blas_precision;
            using this_blas_value_type = map_value_type<base_type::this_blas_type_v, this_blas_precision>;

        public:
            using a_value_type  = typename this_blas_value_type::a_type;
            using b_value_type  = typename this_blas_value_type::b_type;
            using c_value_type  = typename this_blas_value_type::c_type;
        };
#endif

        template<class... Operators>
        class blas_block_execution: public blas_execution<Operators...>
        {
            using this_type = blas_block_execution<Operators...>;
            using base_type = blas_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_blas_precision;

            // GCC7 workaround
            using this_blas_size = typename base_type::this_blas_size;

            // Value type
            using this_blas_value_type = map_value_type<base_type::this_blas_type_v, this_blas_precision>;
            using this_cutlass_value_type = cutlass_value_type<this_blas_value_type>;

            /// ---- Suggestions
            using overloaded_tile_operator = typename base_type::this_blas_overloaded_tile;

            using execution_suggestion_db_t =
                cute_backend::get_database_threads<typename this_cutlass_value_type::a_type,
                                                   typename this_cutlass_value_type::b_type,
                                                   typename this_cutlass_value_type::c_type,
                                                   this_blas_size::m,
                                                   this_blas_size::n,
                                                   this_blas_size::k,
                                                   typename base_type::this_blas_sm,
                                                   void,
                                                   overloaded_tile_operator>;

            /// ---- Traits

            // Block Dimension
            // * Default value: selected by implementation
            static constexpr bool has_block_dim = has_operator<operator_type::block_dim, base_type>::value;
            using default_blas_threads          = cute::conditional_t<base_type::is_complete, execution_suggestion_db_t, cute::Int<128>>;
            using this_blas_block_dim           = get_or_default_t<operator_type::block_dim, base_type, BlockDim<default_blas_threads::value>>;
            static constexpr auto this_blas_block_dim_v = this_blas_block_dim::value;

            static constexpr bool has_ld = has_operator<operator_type::ld, base_type>::value;

            /// ---- Checks

            static constexpr bool valid_block_dim = this_blas_block_dim::flat_size >= 32 && this_blas_block_dim::flat_size <= 1024;
            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't "
                          "have more than 1024 threads.");

            /// ---- Bug Checks
            #if AFFECTED_BY_NVBUG_5218000 && !defined(CUBLASDX_IGNORE_NVBUG_5218000_ASSERT)
            static constexpr bool can_be_impacted_by_nvbug_5218000 =
                (base_type::this_blas_sm_v == 800 or base_type::this_blas_sm_v == 900 or
                base_type::this_blas_sm_v == 1000 or base_type::this_blas_sm_v == 1010 or
                base_type::this_blas_sm_v == 1030 or base_type::this_blas_sm_v == 1200 or
                base_type::this_blas_sm_v == 1210) &&
                (sizeof(typename this_blas_precision::a_type) < sizeof(float) && sizeof(typename this_blas_precision::b_type) < sizeof(float)) &&
                (this_blas_size::m % 16 != 0 || this_blas_size::n % 16 != 0 || this_blas_size::k % 16 != 0) &&
                has_ld;

            static_assert(not can_be_impacted_by_nvbug_5218000,
                          "This configuration can be impacted by CUDA 12.8.0, 12.8.1 and 12.9.0 bug 5218000. \n"
                          "Please either update to the latest CUDA Toolkit and NVCC compiler (12.9.1 is known to work) \n"
                          "or define CUBLASDX_IGNORE_NVBUG_5218000_ASSERT to ignore this check "
                          "and verify correctness of the results. \n"
                          "For more details please consult cuBLASDx documentation at https://docs.nvidia.com/cuda/cublasdx/index.html");
            #endif
            /// ---- Backend

            // CuTe backend implementation
            using gemm_backend = cute_backend::execution<typename this_cutlass_value_type::a_type,
                                                         typename this_cutlass_value_type::b_type,
                                                         typename this_cutlass_value_type::c_type,
                                                         typename this_blas_value_type::a_type,
                                                         typename this_blas_value_type::b_type,
                                                         typename this_blas_value_type::c_type,
                                                         typename base_type::this_blas_alignment,
                                                         this_blas_size::m,
                                                         this_blas_size::n,
                                                         this_blas_size::k,
                                                         typename base_type::this_blas_arrangement,
                                                         typename base_type::this_blas_transpose_mode,
                                                         typename base_type::this_blas_sm,
                                                         typename base_type::this_blas_static_block_dim,
                                                         this_blas_block_dim,
                                                         overloaded_tile_operator>;

            static constexpr CUBLASDX_HOST_DEVICE dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete, "Can't provide suggested block dimensions, description is not complete");
                return default_blas_threads::value;
            }

            static constexpr CUBLASDX_HOST_DEVICE dim3 get_block_dim() {
                static_assert(base_type::is_complete, "Can't provide block dimensions, description is not complete");
                if constexpr(has_block_dim) {
                    return this_blas_block_dim_v;
                }
                return get_suggested_block_dim();
            }

            template<typename TA, typename TB>
            static constexpr bool are_types_compatible() {
                return detail::are_types_compatible_impl<
                    COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                    COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>
                >();
            }

            template<typename TA, typename TB, typename TC>
            static constexpr bool are_types_compatible() {
                return detail::are_types_compatible_impl<
                    COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                    COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>
                >();
            }

            template<typename Alpha, typename TA, typename TB, typename Beta, typename TC>
            static constexpr bool are_types_compatible() {
                return detail::are_types_compatible_impl<
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, Alpha>,
                    COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                    COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, Beta>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>
                >();
            }

            template<class Functor, class ... Args>
            using res_t = COMMONDX_STL_NAMESPACE::decay_t<typename execute_invoke_result<Functor, Args...>::type>;

            template<typename ... Ts>
            using execute_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
                (are_types_compatible<cute::decay_t<Ts>...>())>;

            template<typename ... Ts>
            using execute_disable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
                not (are_types_compatible<cute::decay_t<Ts>...>())>;


            template<class ... Layouts>
            static constexpr bool are_layout_acceptable() {
                return (cute::is_layout<Layouts>::value && ...);
            }

            template<matrix M, class MemTag>
            static constexpr int get_default_ld() {
                static_assert(cute::is_same_v<MemTag, commondx::detail::smem_tag> or cute::is_same_v<MemTag, commondx::detail::gmem_tag>);
                int ret = 0;
                if constexpr(cute::is_same_v<MemTag, commondx::detail::smem_tag>) {
                    ret = choose<M>(base_type::this_blas_lda, base_type::this_blas_ldb, base_type::this_blas_ldc);
                } else {
                    constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                          base_type::this_blas_arrangement_b,
                                                          base_type::this_blas_arrangement_c);
                    ret = (arr == col_major)
                        ? choose<M>(this_blas_size::m, this_blas_size::k, this_blas_size::m)
                        : choose<M>(this_blas_size::k, this_blas_size::n, this_blas_size::n);
                }
                return ret;
            }

            template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
            CUBLASDX_HOST_DEVICE constexpr static auto get_cute_layout(LD ld = {}) {
                static_assert(cute::is_integral<LD>::value);

                using a_type = typename this_blas_value_type::a_type;
                using b_type = typename this_blas_value_type::b_type;
                using c_type = typename this_blas_value_type::c_type;

                using rows                = cute::Int<M == matrix::B ? this_blas_size::k : this_blas_size::m>;
                using columns             = cute::Int<M == matrix::A ? this_blas_size::k : this_blas_size::n>;
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);

                if constexpr(cute::is_static_v<LD>) {
                    constexpr bool valid_ld =
                        (ld >= ((arr == arrangement::col_major) ? rows::value : columns::value));

                    static_assert(valid_ld || (M != matrix::A),
                        "Incorrect leading dimension for A matrix, LDA must be greater than its leading size");
                    static_assert(valid_ld || (M != matrix::B),
                        "Incorrect leading dimension for B matrix, LDB must be greater than its leading size");
                    static_assert(valid_ld || (M != matrix::C),
                        "Incorrect leading dimension for C matrix, LDC must be greater than its leading size");
                }

                return cute_backend::make_layout_from_arrangement<arr>(rows{}, columns{}, ld);
            }

            template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
            CUBLASDX_HOST_DEVICE constexpr static auto tag_cute_layout(LD ld = {}) {
                return commondx::detail::pointer_layout {MemTag {}, get_cute_layout<M, MemTag>(ld)};
            }

            template<matrix M>
            CUBLASDX_HOST_DEVICE constexpr static auto
            suggest_cute_layout_smem() {
                // If an overload is used, turn off suggested mechanism
                if constexpr(base_type::has_overloaded_tile) {
                    return get_cute_layout<M, commondx::detail::smem_tag>();
                } else {
                    using a_type = typename this_blas_value_type::a_type;
                    using b_type = typename this_blas_value_type::b_type;
                    using c_type = typename this_blas_value_type::c_type;

                    using db_a_type = typename this_cutlass_value_type::a_type;
                    using db_b_type = typename this_cutlass_value_type::b_type;
                    using db_c_type = typename this_cutlass_value_type::c_type;

                    constexpr bool is_a_left = base_type::this_blas_arrangement_a == arrangement::col_major;
                    constexpr bool is_b_left = base_type::this_blas_arrangement_b == arrangement::col_major;
                    constexpr bool is_c_left = base_type::this_blas_arrangement_c == arrangement::col_major;

                    constexpr bool is_swizzle_available = cublasdx::detail::layout_database::has_optimal_config<
                        max_threads_per_block,
                        base_type::this_blas_sm_v,
                        db_a_type, is_a_left, a_alignment,
                        db_b_type, is_b_left, b_alignment,
                        db_c_type, is_c_left, c_alignment,
                        this_blas_size::m,
                        this_blas_size::n,
                        this_blas_size::k>();

                    if constexpr(is_swizzle_available) {
                        return cublasdx::detail::layout_database::get_optimal_layout<
                            M,
                            max_threads_per_block,
                            base_type::this_blas_sm_v,
                            db_a_type, is_a_left, a_alignment,
                            db_b_type, is_b_left, b_alignment,
                            db_c_type, is_c_left, c_alignment,
                            this_blas_size::m,
                            this_blas_size::n,
                            this_blas_size::k>();
                    } else {
                        return get_cute_layout<M, commondx::detail::smem_tag>();
                    }
                }

                CUTE_GCC_UNREACHABLE;
            }

            template<matrix M>
            CUBLASDX_HOST_DEVICE constexpr static auto
            tag_suggested_cute_layout_smem() {
                return commondx::detail::pointer_layout {commondx::detail::smem_tag {}, suggest_cute_layout_smem<M>()};
            }

        public:
            using a_value_type = typename this_blas_value_type::a_type;
            using b_value_type = typename this_blas_value_type::b_type;
            using c_value_type = typename this_blas_value_type::c_type;

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_gmem_a(Ts ... ts) {
                return tag_cute_layout<matrix::A, commondx::detail::gmem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_gmem_b(Ts ... ts) {
                return tag_cute_layout<matrix::B, commondx::detail::gmem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_gmem_c(Ts ... ts) {
                return tag_cute_layout<matrix::C, commondx::detail::gmem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_smem_a(Ts ... ts) {
                return tag_cute_layout<matrix::A, commondx::detail::smem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_smem_b(Ts ... ts) {
                return tag_cute_layout<matrix::B, commondx::detail::smem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            get_layout_smem_c(Ts ... ts) {
                return tag_cute_layout<matrix::C, commondx::detail::smem_tag>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            suggest_layout_smem_a(Ts ... ts) {
                return tag_suggested_cute_layout_smem<matrix::A>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            suggest_layout_smem_b(Ts ... ts) {
                return tag_suggested_cute_layout_smem<matrix::B>(ts...);
            }

            template<typename ... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto
            suggest_layout_smem_c(Ts ... ts) {
                return tag_suggested_cute_layout_smem<matrix::C>(ts...);
            }

            template<class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class CEngine, class CLayout,
                     class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE auto execute(const cublasdx::tensor<AEngine, ALayout>&  tensor_a,
                                           const cublasdx::tensor<BEngine, BLayout>&  tensor_b,
                                           cublasdx::tensor<CEngine, CLayout>      &  tensor_c,
                                           const ALoadOp&                             a_load_op  = {},
                                           const BLoadOp&                             b_load_op  = {})
                -> execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                       res_t<BLoadOp, typename BEngine::value_type>,
                                       typename CEngine::value_type>{
                using AShape = decltype(shape(ALayout{}));
                using BShape = decltype(shape(BLayout{}));

                using a_engine_t = typename AEngine::value_type;
                using b_engine_t = typename BEngine::value_type;
                using a_result_t = res_t<ALoadOp, a_engine_t>;
                using b_result_t = res_t<BLoadOp, b_engine_t>;
                using cutlass_a_engine_t = convert_to_cutlass_type_t<a_engine_t>;
                using cutlass_b_engine_t = convert_to_cutlass_type_t<b_engine_t>;

                // Check if sizes are static
                static_assert(cute::is_static_v<AShape> && cute::is_static_v<BShape>,
                              "All layout shapes must be static, only strides can be dynamic");

                // Check if layout shapes are 2D and non-hierarchical
                // Check if layout shapes are compatible with operator defined shapes
                static_assert(
                    rank(AShape{}) == 2 && size(cute::get<0>(AShape{})) == this_blas_size::m &&
                                           size(cute::get<1>(AShape{})) == this_blas_size::k &&

                    rank(BShape{}) == 2 && size(cute::get<0>(BShape{})) == this_blas_size::k &&
                                           size(cute::get<1>(BShape{})) == this_blas_size::n,
                    "Tensor API currently supports only \
                     hierarchical 2D tensors sizes of which \
                     match operator provided sizes"
                );

                // Input types check
                static_assert(
                    (sizeof(a_engine_t) == sizeof(a_value_type) and alignof(a_engine_t) == alignof(a_value_type) and
                     sizeof(b_engine_t) == sizeof(b_value_type) and alignof(b_engine_t) == alignof(b_value_type)) or
                    base_type::has_alignment, "If using data types decoupled from computation precision, Alignment operator must be set"
                );

                // Alignment checks
                static_assert(((base_type::this_blas_alignment_a % alignof(a_engine_t)) == 0) &&
                               (base_type::this_blas_alignment_a >= alignof(a_engine_t)),
                    "Incorrect alignment for matrix A; it has to be a multiple of type of matrix A");
                static_assert(((base_type::this_blas_alignment_b % alignof(b_engine_t)) == 0) &&
                               (base_type::this_blas_alignment_b >= alignof(b_engine_t)),
                    "Incorrect alignment for matrix B; it has to be a multiple of type of matrix B");

                // Functor checks
                static_assert(is_functor_compatible<ALoadOp, a_engine_t, a_value_type>(),
                    "ALoadOp functor must accept value of tensor_a type and return value convertible to tensor_a type");
                static_assert(is_functor_compatible<BLoadOp, b_engine_t, b_value_type>(),
                    "BLoadOp functor must accept value of tensor_b type and return value convertible to tensor_b type");

                transform_op_wrapper<ALoadOp, a_engine_t> cutlass_a_load_op{a_load_op};
                transform_op_wrapper<BLoadOp, b_engine_t> cutlass_b_load_op{b_load_op};

                gemm_backend::tensor_gemm(cute::recast<cutlass_a_engine_t>(tensor_a),
                                          cute::recast<cutlass_b_engine_t>(tensor_b),
                                          tensor_c,
                                          cutlass_a_load_op,
                                          cutlass_b_load_op);
            }

            template<class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class ALoadOp = identity, class BLoadOp = identity,
                     execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                         res_t<BLoadOp, typename BEngine::value_type>>* = nullptr>
            CUBLASDX_DEVICE auto execute(const cublasdx::tensor<AEngine, ALayout>&  tensor_a,
                                           const cublasdx::tensor<BEngine, BLayout>&  tensor_b,
                                           const ALoadOp&                             a_load_op  = {},
                                           const BLoadOp&                             b_load_op  = {}) {
                // Create fragment
                auto partitioner = gemm_backend::get_partitioner(ALayout{}, BLayout{});
                auto c_frag = partitioner.make_accumulator_fragment();
                cublasdx::clear(c_frag);

                // Call GEMM
                execute(tensor_a, tensor_b, c_frag, a_load_op, b_load_op);

                // Return fragment (results) and partitioner
                return cute::make_tuple(c_frag, partitioner);
            }


            // C in Shared Memory API
            template<class Alpha,
                     class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class Beta,
                     class CEngine, class CLayout,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            CUBLASDX_DEVICE auto execute(const Alpha&                               alpha,
                                           const cublasdx::tensor<AEngine, ALayout>&  tensor_a,
                                           const cublasdx::tensor<BEngine, BLayout>&  tensor_b,
                                           const Beta&                                beta,
                                           cublasdx::tensor<CEngine, CLayout>&        tensor_c,
                                           const ALoadOp&                             a_load_op  = {},
                                           const BLoadOp&                             b_load_op  = {},
                                           const CLoadOp&                             c_load_op  = {},
                                           const CStoreOp&                            c_store_op = {})
                -> execute_enable_if_t<Alpha, // Alpha
                                       res_t<ALoadOp, typename AEngine::value_type>,
                                       res_t<BLoadOp, typename BEngine::value_type>,
                                       Beta, // Beta
                                       res_t<CLoadOp, typename CEngine::value_type>>{

                using AShape = decltype(shape(ALayout{}));
                using BShape = decltype(shape(BLayout{}));
                using CShape = decltype(shape(CLayout{}));

                using a_engine_t = typename AEngine::value_type;
                using b_engine_t = typename BEngine::value_type;
                using c_engine_t = typename CEngine::value_type;
                using a_result_t = res_t<ALoadOp, a_engine_t>;
                using b_result_t = res_t<BLoadOp, b_engine_t>;
                using c_result_t = res_t<CLoadOp, c_engine_t>;
                using cutlass_a_engine_t = convert_to_cutlass_type_t<a_engine_t>;
                using cutlass_b_engine_t = convert_to_cutlass_type_t<b_engine_t>;
                using cutlass_c_engine_t = convert_to_cutlass_type_t<c_engine_t>;

                // Check if sizes are static
                static_assert(cute::is_static_v<AShape> && cute::is_static_v<BShape> && cute::is_static_v<CShape>,
                              "All layout shapes must be static, only strides can be dynamic");

                // Check if layout shapes are 2D and non-hierarchical
                // Check if layout shapes are compatible with operator defined shapes
                static_assert(
                    rank(AShape{}) == 2 && size(cute::get<0>(AShape{})) == this_blas_size::m &&
                                           size(cute::get<1>(AShape{})) == this_blas_size::k &&

                    rank(BShape{}) == 2 && size(cute::get<0>(BShape{})) == this_blas_size::k &&
                                           size(cute::get<1>(BShape{})) == this_blas_size::n &&

                    rank(CShape{}) == 2 && size(cute::get<0>(CShape{})) == this_blas_size::m &&
                                           size(cute::get<1>(CShape{})) == this_blas_size::n,
                    "Tensor API currently supports only \
                     hierarchical 2D tensors sizes of which \
                     match operator provided sizes"
                );

                static_assert(
                    (sizeof(a_engine_t) == sizeof(a_value_type) and alignof(a_engine_t) == alignof(a_value_type) and
                     sizeof(b_engine_t) == sizeof(b_value_type) and alignof(b_engine_t) == alignof(b_value_type) and
                     sizeof(c_engine_t) == sizeof(c_value_type) and alignof(c_engine_t) == alignof(c_value_type)) or
                    base_type::has_alignment, "If using data types decoupled from computation precision, Alignment operator must be set"
                );

                // Alignment check
                static_assert(((base_type::this_blas_alignment_a % alignof(a_engine_t)) == 0) &&
                               (base_type::this_blas_alignment_a >= alignof(a_engine_t)),
                    "Incorrect alignment for matrix A; it has to be a multiple of type of matrix A");
                static_assert(((base_type::this_blas_alignment_b % alignof(b_engine_t)) == 0) &&
                               (base_type::this_blas_alignment_b >= alignof(b_engine_t)),
                    "Incorrect alignment for matrix B; it has to be a multiple of type of matrix B");
                static_assert(((base_type::this_blas_alignment_c % alignof(c_engine_t)) == 0) &&
                               (base_type::this_blas_alignment_c >= alignof(c_engine_t)),
                    "Incorrect alignment for matrix C; it has to be a multiple of type of matrix C");

                // Functor checks
                static_assert(is_functor_compatible<ALoadOp, a_engine_t, a_value_type>(),
                    "ALoadOp functor must accept value of tensor_a type and return value convertible to BLAS::a_value_type");
                static_assert(is_functor_compatible<BLoadOp, b_engine_t, b_value_type>(),
                    "BLoadOp functor must accept value of tensor_b type and return value convertible to BLAS::b_value_type");
                static_assert(is_functor_compatible<CLoadOp, c_engine_t, c_value_type>(),
                    "CLoadOp functor must accept value of tensor_c type and return value convertible to BLAS::c_value_type");
                static_assert(is_functor_compatible<CStoreOp, c_value_type, c_engine_t>(),
                    "CStoreOp functor must accept value of tensor_c type and return value convertible to BLAS::c_value_type");

                transform_op_wrapper<ALoadOp, a_engine_t> cutlass_a_load_op{a_load_op};
                transform_op_wrapper<BLoadOp, b_engine_t> cutlass_b_load_op{b_load_op};
                transform_op_wrapper<CLoadOp, c_engine_t> cutlass_c_load_op{c_load_op};
                transform_op_wrapper<CStoreOp, c_result_t> cutlass_c_store_op{c_store_op};

                gemm_backend::tensor_gemm(cute::recast<cutlass_a_engine_t>(tensor_a),
                                          cute::recast<cutlass_b_engine_t>(tensor_b),
                                          cute::recast<cutlass_c_engine_t>(tensor_c),
                                          cast_to_cutlass_type<typename this_cutlass_value_type::c_type>(alpha),
                                          cast_to_cutlass_type<typename this_cutlass_value_type::c_type>(beta),
                                          cutlass_a_load_op,
                                          cutlass_b_load_op,
                                          cutlass_c_load_op,
                                          cutlass_c_store_op);
            }

            template<class Alpha,
                     class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class Beta,
                     class CEngine, class CLayout,
                     class ALoadOp, class BLoadOp,
                     class CLoadOp, class CStoreOp>
            CUBLASDX_DEVICE auto execute(const Alpha&                               /* alpha */,
                                           const cublasdx::tensor<AEngine, ALayout>&  /* tensor_a */,
                                           const cublasdx::tensor<BEngine, BLayout>&  /* tensor_b */,
                                           const Beta&                                /* beta */,
                                           cublasdx::tensor<CEngine, CLayout>&        /* tensor_c */,
                                           [[maybe_unused]] const ALoadOp&  a_load_op  = identity {},
                                           [[maybe_unused]] const BLoadOp&  b_load_op  = identity {},
                                           [[maybe_unused]] const CLoadOp&  c_load_op  = identity {},
                                           [[maybe_unused]] const CStoreOp& c_store_op = identity {})
                -> execute_disable_if_t<Alpha, // Must be compatible (size, alignment) with c_value_type
                                        res_t<ALoadOp, typename AEngine::value_type>,
                                        res_t<BLoadOp, typename BEngine::value_type>,
                                        Beta, // Must be compatible (size, alignment) with c_value_type
                                        res_t<CLoadOp, typename CEngine::value_type>> {

                constexpr bool condition = are_types_compatible<
                    Alpha,
                    res_t<ALoadOp, typename AEngine::value_type>,
                    res_t<BLoadOp, typename BEngine::value_type>,
                    Beta,
                    res_t<CLoadOp, typename CEngine::value_type>>();

                static_assert(condition, "Incorrect types for inputs, LD operator used or TransposeMode used. \
                    Ensure input types for A, B and C match the types \
                    indicated in the Precision<...> operator.");
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class Alpha, class TA, class LDA, class TB, class LDB, class Beta, class TC, class LDC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            CUBLASDX_DEVICE auto execute(const Alpha&          alpha,
                                           TA*                   matrix_a,
                                           const LDA             lda,
                                           TB*                   matrix_b,
                                           const LDB             ldb,
                                           const Beta&           beta,
                                           TC*                   matrix_c,
                                           const LDC             ldc,
                                           const ALoadOp&        a_load_op  = {},
                                           const BLoadOp&        b_load_op  = {},
                                           const CLoadOp&        c_load_op  = {},
                                           const CStoreOp&       c_store_op = {}) //
                -> execute_enable_if_t<Alpha, TA, TB, Beta, TC> {
                    static_assert(cute::is_integral<LDA>::value and
                                  cute::is_integral<LDB>::value and
                                  cute::is_integral<LDC>::value,
                                  "LD values must be either static or dynamic integral types");
                cute::Tensor ta = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_a),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_a>(
                        cute::Int<this_blas_size::m>{}, cute::Int<this_blas_size::k>{}, lda
                    )
                );
                cute::Tensor tb = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_b),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_b>(
                        cute::Int<this_blas_size::k>{}, cute::Int<this_blas_size::n>{}, ldb
                    )
                );
                cute::Tensor tc = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_c),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_c>(
                        cute::Int<this_blas_size::m>{}, cute::Int<this_blas_size::n>{}, ldc
                    )
                );

                execute(alpha, ta, tb, beta, tc,
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_a>(), a_load_op),
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_b>(), b_load_op),
                        c_load_op,
                        c_store_op);
            }

            template<class Alpha, class TA, class TB, class Beta, class TC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            CUBLASDX_DEVICE auto execute(const Alpha     alpha,
                                           TA*             matrix_a,
                                           TB*             matrix_b,
                                           const Beta      beta,
                                           TC*             matrix_c,
                                           const ALoadOp&  a_load_op  = {},
                                           const BLoadOp&  b_load_op  = {},
                                           const CLoadOp&  c_load_op  = {},
                                           const CStoreOp& c_store_op = {}) //
                -> execute_enable_if_t<Alpha, TA, TB, Beta, TC> {

                auto lda = cute::Int<this_type::lda>{};
                auto ldb = cute::Int<this_type::ldb>{};
                auto ldc = cute::Int<this_type::ldc>{};

                execute(alpha,
                        matrix_a, lda,
                        matrix_b, ldb,
                        beta,
                        matrix_c, ldc,
                        a_load_op,
                        b_load_op,
                        c_load_op,
                        c_store_op);
            }

            template<class Alpha, class TA, class TB, class Beta, class TC,
                     class ALoadOp = identity,  class BLoadOp  = identity,
                     class CLoadOp = identity,  class CStoreOp = identity>
            CUBLASDX_DEVICE auto execute(const TC /* alpha */,
                                           TA* /* matrix_a */,
                                           const unsigned int /* lda */,
                                           TB* /* matrix_b */,
                                           const unsigned int /* ldb */,
                                           const TC /* beta */,
                                           TC* /* matrix_c */,
                                           const unsigned int /* ldc */,
                                           const ALoadOp& /* a_load_op */ = {},
                                           const BLoadOp& /* b_load_op */ = {},
                                           const CLoadOp& /* c_load_op */ = {},
                                           const CStoreOp& /* c_store_op */ = {}) //
                -> execute_disable_if_t<Alpha, TA, TB, Beta, TC> {
                static constexpr bool condition = are_types_compatible<Alpha, TA, TB, Beta, TC>();

                static_assert(condition,
                "Incorrect types for inputs or lacking TransposeMode operator.  \
                Ensure input types for A, B and C match the types \
                indicated in the Precision<...> operator.");
            }

            template<class Alpha, class TA, class TB, class Beta, class TC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            CUBLASDX_DEVICE auto execute(const Alpha /* alpha */,
                                           TA* /* matrix_a */,
                                           TB* /* matrix_b */,
                                           const Beta /* beta */,
                                           TC* /* matrix_c */,
                                           const ALoadOp& /* a_load_op */ = {},
                                           const BLoadOp& /* b_load_op */ = {},
                                           const CLoadOp& /* c_load_op */ = {},
                                           const CStoreOp& /* c_store_op */ = {}) //
                -> execute_disable_if_t<Alpha, TA, TB, Beta, TC> {
                static constexpr bool condition = are_types_compatible<Alpha, TA, TB, Beta, TC>();

                static_assert(condition,
                "Incorrect types for inputs or lacking TransposeMode operator.  \
                Ensure input types for A, B and C match the types \
                indicated in the Precision<...> operator.");
            }

            static CUBLASDX_DEVICE auto get_partitioner() {
                return gemm_backend::get_partitioner();
            }

            static CUBLASDX_DEVICE auto suggest_partitioner() {
                return gemm_backend::suggest_partitioner();
            }

            // Number of elements in A, B, C matrices (includes padding / leading dimensions)
            // (ld * cols)
            static constexpr unsigned int a_size = base_type::this_blas_a_size;
            static constexpr unsigned int b_size = base_type::this_blas_b_size;
            static constexpr unsigned int c_size = base_type::this_blas_c_size;

            // Leading dimensions of A, B, C matrices
            static constexpr unsigned int lda = base_type::this_blas_lda;
            static constexpr unsigned int ldb = base_type::this_blas_ldb;
            static constexpr unsigned int ldc = base_type::this_blas_ldc;

            // Pointer alignments of A, B, C matrices in bytes
            static constexpr unsigned int a_alignment = base_type::this_blas_alignment_a;
            static constexpr unsigned int b_alignment = base_type::this_blas_alignment_b;
            static constexpr unsigned int c_alignment = base_type::this_blas_alignment_c;

            // Logical dimensions of A, B, C matrices
            // (row; cols)
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> a_dim = base_type::this_blas_a_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> b_dim = base_type::this_blas_b_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> c_dim = base_type::this_blas_c_dim;

            static constexpr dim3         suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3         block_dim           = get_block_dim();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
            static constexpr unsigned int min_blocks_per_multiprocessor = 1;

            friend class layout_type_provider<this_type>;
        };

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator      = has_operator<operator_type::block, blas_operator_wrapper<Operators...>>::value;
            static constexpr bool has_execution_operator  = has_block_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called blas_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before BLAS description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using operator_wrapper_type = blas_operator_wrapper<Operators...>;
            using execution_type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_complete_blas<operator_wrapper_type>::value,
                                                             blas_block_execution<Operators...>,
                                                             blas_block_execution_partial<Operators...>>::type;
#else
            using execution_type = blas_block_execution<Operators...>;
#endif
            using description_type = blas_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    CUBLASDX_HOST_DEVICE auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    CUBLASDX_HOST_DEVICE auto operator+(const detail::blas_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    CUBLASDX_HOST_DEVICE auto operator+(const Operator1&,
                                                       const detail::blas_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    CUBLASDX_HOST_DEVICE auto operator+(const detail::blas_description<Operators1...>&,
                                                       const detail::blas_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
