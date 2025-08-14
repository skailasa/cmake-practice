// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_TRANSPOSE_HPP
#define CUSOLVERDX_OPERATORS_TRANSPOSE_HPP

namespace cusolverdx {
    enum class transpose
    {
        non_transposed,
        transposed,
        conj_transposed,
    };
    inline constexpr auto non_trans  = transpose::non_transposed;
    inline constexpr auto trans      = transpose::transposed;
    inline constexpr auto conj_trans = transpose::conj_transposed;

    template<transpose Value>
    struct TransposeMode: commondx::detail::constant_operator_expression<transpose, Value> {};

    namespace detail {
        using default_transpose_operator = TransposeMode<non_trans>;
    } // namespace detail
}

namespace commondx::detail {
    template<cusolverdx::transpose Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::transpose, cusolverdx::TransposeMode<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::transpose Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::TransposeMode<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::transpose;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_TRANSPOSE_HPP
