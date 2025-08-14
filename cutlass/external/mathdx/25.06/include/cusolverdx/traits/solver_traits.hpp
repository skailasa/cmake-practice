// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP
#define CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/detail/solver_description_fd.hpp"
#include "cusolverdx/operators.hpp"
#include "cusolverdx/types.hpp"
#include "cusolverdx/traits/detail/description_traits.hpp"
#include "cusolverdx/traits/detail/is_complete.hpp"

namespace cusolverdx {
    namespace detail {
        template<commondx::data_type data_type, class Precision>
        struct map_value_type {
            using a_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::a_type>, typename Precision::a_type>;
            using x_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::x_type>, typename Precision::x_type>;
            using b_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::b_type>, typename Precision::b_type>;
        };
    } // namespace detail

    // ------------------
    // Execution checkers
    // ------------------
    // is_block
    template<class Description>
    struct is_block {
    public:
        static constexpr bool value = detail::has_operator_v<operator_type::block, Description>;
    };

    template<class Description>
    inline constexpr bool is_block_v = is_block<Description>::value;

    // ----------------
    // Operator getters
    // ----------------

    // precision_of
    template<class Description>
    struct precision_of {
    private:
        using description = commondx::detail::get_or_default_t<operator_type, operator_type::precision, Description, detail::default_precision_operator>;

    public:
        using a_type = typename description::a_type;
        using x_type = typename description::x_type;
        using b_type = typename description::b_type;
    };

    template<class Description>
    using precision_of_a_t = typename precision_of<Description>::a_type;
    template<class Description>
    using precision_of_x_t = typename precision_of<Description>::x_type;
    template<class Description>
    using precision_of_b_t = typename precision_of<Description>::b_type;

    // type_of
    template<class Description>
    using type_of = commondx::data_type_of<operator_type, Description, detail::default_type_operator>;
    template<class Description>
    inline constexpr type type_of_v = type_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<operator_type, Description>;
    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    // size_of
    template<class Description>
    struct size_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

    public:
        using value_type                   = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int m    = detail::get_t<operator_type::size, Description>::m;
        static constexpr unsigned int n    = detail::get_t<operator_type::size, Description>::n;
        static constexpr unsigned int k    = detail::get_t<operator_type::size, Description>::k;

        static constexpr value_type value = value_type {m, n, k};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> size_of_v = size_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int size_of_v_m = size_of<Description>::m;
    template<class Description>
    inline constexpr unsigned int size_of_v_n = size_of<Description>::n;
    template<class Description>
    inline constexpr unsigned int size_of_v_k = size_of<Description>::k;

    // function_of
    template<class Description>
    struct function_of {
    private:
        static constexpr bool has_function = detail::has_operator_v<operator_type::function, Description>;

    public:
        using value_type = function;

        static constexpr value_type value = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>::value;
    };

    template<class Description>
    inline constexpr function function_of_v = function_of<Description>::value;

    // fill_mode_of
    template<class Description>
    struct fill_mode_of {
        using value_type                  = fill_mode;
        static constexpr value_type value = detail::get_or_default_t<operator_type::fill_mode, Description, detail::default_fill_mode_operator>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fill_mode fill_mode_of<Description>::value;
    template<class Description>
    inline constexpr fill_mode fill_mode_of_v = fill_mode_of<Description>::value;

    // arrangement_of
    template<class Description>
    struct arrangement_of {
    private:
        using arrangement_type  = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
    public:
        using value_type               = COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement>;
        static constexpr arrangement a = arrangement_type::a;
        static constexpr arrangement b = arrangement_type::b;


        static constexpr value_type value = value_type {a, b};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement> arrangement_of_v = arrangement_of<Description>::value;
    template<class Description>
    inline constexpr arrangement arrangement_of_v_a = arrangement_of<Description>::a;
    template<class Description>
    inline constexpr arrangement arrangement_of_v_b = arrangement_of<Description>::b;

    // transpose_mode_of
    template<class Description>
    struct transpose_mode_of {
        using value_type                  = transpose;
        static constexpr value_type value = detail::get_or_default_t<operator_type::transpose, Description, detail::default_transpose_operator>::value;
    };

    template<class Description>
    inline constexpr transpose transpose_mode_of_v = transpose_mode_of<Description>::value;

    // side_of
    template<class Description>
    struct side_of {
        using value_type                  = side;
        static constexpr value_type value = detail::get_or_default_t<operator_type::side, Description, Side<side::left>>::value;
    };

    template<class Description>
    inline constexpr side side_of_v = side_of<Description>::value;

    // diag_of
    template<class Description>
    struct diag_of {
        using value_type                  = diag;
        static constexpr value_type value = detail::get_or_default_t<operator_type::diag, Description, Diag<diag::non_unit>>::value;
    };

    template<class Description>
    inline constexpr diag diag_of_v = diag_of<Description>::value;

    // leading dimension of
    namespace detail {
        template<unsigned int M, unsigned int N, unsigned int K, arrangement Arr, arrangement Brr, function Func = function::potrf, side Side = side::left>
        struct default_leading_dimension {
            static constexpr bool is_function_unmq = is_unmq<Func>::value;

            // Regular leading dimension calculations
            static constexpr unsigned int lda_regular = (Arr == arrangement::col_major) ? M : N;
            static constexpr unsigned int ldb_regular = (Brr == arrangement::col_major) ? const_max(M, N) : K;

            // UNMQ specific leading dimension calculations
            static constexpr unsigned int lda_unmq = (Func == function::unmqr) ?
                ((Arr == arrangement::col_major) ? 
                    ((Side == side::left) ? M : N) : K) :
                ((Arr == arrangement::col_major) ? K :
                    ((Side == side::left) ? M : N));
            static constexpr unsigned int ldb_unmq = (Brr == arrangement::col_major) ? M : N;

            // Choose between regular and UNMQ leading dimensions
            static constexpr unsigned int lda = is_function_unmq ? lda_unmq : lda_regular;
            static constexpr unsigned int ldb = is_function_unmq ? ldb_unmq : ldb_regular;

            using type = LeadingDimension<lda, ldb>;
        };

        template<unsigned int M, unsigned int N, unsigned int K, arrangement Arr, arrangement Brr, function Func = function::potrf, side Side = side::left>
        using default_leading_dimension_t = typename default_leading_dimension<M, N, K, Arr, Brr, Func, Side>::type;
    } // namespace detail

    template<class Description>
    struct leading_dimension_of {
    private:
        using this_size = detail::get_or_default_t<operator_type::size, Description, Size<1, 1, 1>>;
        using this_arrangement = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
        using this_function = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>;
        using this_side = detail::get_or_default_t<operator_type::side, Description, detail::default_side_operator>;

        using default_ld = detail::default_leading_dimension_t<
            this_size::m, 
            this_size::n, 
            this_size::k, 
            this_arrangement::a, 
            this_arrangement::b,
            this_function::value,
            this_side::value>;

    public:
        using value_type = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int>;
        static constexpr unsigned int a = detail::get_or_default_t<operator_type::leading_dimension, Description, default_ld>::a;
        static constexpr unsigned int b = detail::get_or_default_t<operator_type::leading_dimension, Description, default_ld>::b;

        static constexpr value_type value = value_type{a, b};
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> leading_dimension_of_v = leading_dimension_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_a = leading_dimension_of<Description>::a;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_b = leading_dimension_of<Description>::b;

    // --------------------------
    // Matrix size calculations
    // --------------------------
    template<class Description>
    struct matrix_size_of {
    private:
        using this_size = detail::get_or_default_t<operator_type::size, Description, Size<1, 1, 1>>;
        using this_arrangement = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
        using this_function = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>;
        using this_side = detail::get_or_default_t<operator_type::side, Description, detail::default_side_operator>;

        static constexpr bool is_function_unmq = detail::is_unmq<this_function::value>::value;
        static constexpr bool is_function_solver = detail::is_solver<this_function::value>::value;
        static constexpr bool is_function_qr = detail::is_qr<this_function::value>::value;
        static constexpr bool is_function_lu_pp = detail::is_lu_partial_pivot<this_function::value>::value;

        // Regular matrix size calculations
        static constexpr unsigned int a_size_regular = leading_dimension_of_v_a<Description> * 
            (this_arrangement::a == arrangement::col_major ? this_size::n : this_size::m);
        static constexpr unsigned int b_size_regular = leading_dimension_of_v_b<Description> * 
            (this_arrangement::b == arrangement::col_major ? this_size::k : detail::const_max(this_size::m, this_size::n));

        // UNMQ specific matrix size calculations
        static constexpr unsigned int am = (this_function::value == function::unmqr) ? 
            (this_side::value == side::left ? this_size::m : this_size::n) : this_size::k;
        static constexpr unsigned int an = (this_function::value == function::unmqr) ? 
            this_size::k : (this_side::value == side::left ? this_size::m : this_size::n);
        static constexpr unsigned int a_size_unmq = leading_dimension_of_v_a<Description> * 
            (this_arrangement::a == arrangement::col_major ? an : am);
        static constexpr unsigned int b_size_unmq = leading_dimension_of_v_b<Description> * 
            (this_arrangement::b == arrangement::col_major ? this_size::n : this_size::m);

        static constexpr unsigned int extra_size_qr = is_function_unmq ? this_size::k : detail::const_min(this_size::m, this_size::n);
        static constexpr unsigned int extra_bytes_lu_pp = sizeof(int) * detail::const_min(this_size::m, this_size::n);


    public:
        using value_type = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a_size = is_function_unmq ? a_size_unmq : a_size_regular;
        static constexpr unsigned int b_size = is_function_solver ? (is_function_unmq ? b_size_unmq : b_size_regular) : 0;
        static constexpr unsigned int extra_size = is_function_qr ? extra_size_qr : 0;
        static constexpr unsigned int extra_bytes = is_function_lu_pp ? extra_bytes_lu_pp : 0;

        static constexpr value_type value = value_type{a_size, b_size, extra_size, extra_bytes};
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int, unsigned int> matrix_size_of_v = matrix_size_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_a = matrix_size_of<Description>::a_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_b = matrix_size_of<Description>::b_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_extra_size = matrix_size_of<Description>::extra_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_extra_bytes = matrix_size_of<Description>::extra_bytes;

    // --------------------------
    // General Description traits
    // --------------------------

    template<class Description>
    using is_solver = commondx::is_dx_expression<Description>;

    template<class Description>
    inline constexpr bool is_solver_v = is_solver<Description>::value;

    template<class Description>
    using is_solver_execution = commondx::is_dx_execution_expression<operator_type, Description>;

    template<class Description>
    inline constexpr bool is_solver_execution_v = is_solver_execution<Description>::value;

    template<class Description>
    using is_complete_solver = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_solver_v = is_complete_solver<Description>::value;

    template<class Description>
    using is_complete_solver_execution = commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_execution_description>;

    template<class Description>
    inline constexpr bool is_complete_solver_execution_v = is_complete_solver_execution<Description>::value;

    template<class Description>
    using extract_solver_description = commondx::extract_dx_description<detail::solver_description, Description, operator_type>;

    template<class Description>
    using extract_solver_description_t = typename extract_solver_description<Description>::type;

    // --------------------------------
    // Suggested_leading_dimension_of
    // --------------------------------
    // Recommend users to use suggested_leading_dimension_of trait to improve perf
    namespace detail {

        template<typename T, unsigned size>
        static constexpr unsigned pad_if_needed() {
            constexpr unsigned bytes       = sizeof(T) * size;
            constexpr bool     add_padding = (bytes >= 128 && bytes % 128 == 0);
            return add_padding ? size + 16 / sizeof(T) : size;
        }

        // forward declaration
        template<class Precision, type Type, arrangement Arr, arrangement Brr, class Size, class LD, unsigned int Arch>
        struct is_supported_impl;

        template<class Precision, type Type, arrangement Arr, arrangement Brr, class Size, unsigned int Arch>
        struct suggested_leading_dimension_of_impl {

            using a_value_type = typename map_value_type<Type, Precision>::a_type;

            static constexpr unsigned int best_lda = (Arr == arrangement::col_major) ? pad_if_needed<a_value_type, Size::m>() : pad_if_needed<a_value_type, Size::n>();

            // Default leading dimensions
            using default_ld = default_leading_dimension_t<Size::m, Size::n, Size::k, Arr, Brr, function::potrf, side::left>;

            // Check if the operation is supported with LD(best_lda, best_ldb, best_ldc)
            static constexpr bool padding_supported = is_supported_impl<Precision, Type, Arr, Brr, Size, LeadingDimension<best_lda, default_ld::b>, Arch>::value;

            // Use default or best leading dimensions
            static constexpr unsigned int lda = padding_supported ? best_lda : default_ld::a;
            static constexpr unsigned int ldb = default_ld::b;
        };
    } // namespace detail


    template<class Description, unsigned int Architecture>
    struct suggested_leading_dimension_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

        using suggested = detail::suggested_leading_dimension_of_impl<precision_of<Description>,
                                                                      type_of_v<Description>,
                                                                      arrangement_of_v_a<Description>,
                                                                      arrangement_of_v_b<Description>,
                                                                      detail::get_t<operator_type::size, Description>,
                                                                      Architecture>;

    public:
        using value_type                  = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int>;
        static constexpr unsigned int lda = suggested::lda;
        static constexpr unsigned int ldb = suggested::ldb;

        using type = LeadingDimension<lda, ldb>;

        static constexpr value_type value = value_type {lda, ldb};
        constexpr             operator type() const noexcept { return value; }
    };

    template<class Description, unsigned int Architecture>
    using suggested_leading_dimension_of_t = typename suggested_leading_dimension_of<Description, Architecture>::type;

    template<class Description, unsigned int Architecture>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> suggested_leading_dimension_of_v = suggested_leading_dimension_of<Description, Architecture>::value;
    template<class Description, unsigned int Architecture>
    inline constexpr unsigned int suggested_leading_dimension_of_v_a = suggested_leading_dimension_of<Description, Architecture>::lda;
    template<class Description, unsigned int Architecture>
    inline constexpr unsigned int suggested_leading_dimension_of_v_b = suggested_leading_dimension_of<Description, Architecture>::ldb;

} // namespace cusolverdx

#endif // CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP
