// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP
#define CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cusolverdx {
    enum class arrangement
    {
        col_major,
        row_major,
    };

    inline constexpr auto col_major   = arrangement::col_major;
    inline constexpr auto left_layout = arrangement::col_major;
    inline constexpr auto row_major   = arrangement::row_major;
    inline constexpr auto right_major = arrangement::row_major;

    template<arrangement Arr, arrangement Brr = Arr>
    struct Arrangement: commondx::detail::operator_expression {
        static_assert(Arr == arrangement::col_major || Arr == arrangement::row_major || Brr == arrangement::row_major || Brr == arrangement::col_major, "Order has to be col_major or row_major");

        static constexpr arrangement a = Arr;
        static constexpr arrangement b = Brr;
    };

    namespace detail {
        using default_arrangement_operator = Arrangement<arrangement::col_major, arrangement::col_major>;
    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::arrangement Arr, cusolverdx::arrangement Brr>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::arrangement, cusolverdx::Arrangement<Arr, Brr>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::arrangement Arr, cusolverdx::arrangement Brr>
    struct get_operator_type<cusolverdx::operator_type,cusolverdx::Arrangement<Arr, Brr>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::arrangement;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP
