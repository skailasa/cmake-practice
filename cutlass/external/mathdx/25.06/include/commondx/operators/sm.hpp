// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_SM_HPP
#define COMMONDX_OPERATORS_SM_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int Architecture>
    struct SM;

    template<>
    struct SM<700>: public detail::constant_operator_expression<unsigned int, 700> {};

    template<>
    struct SM<720>: public detail::constant_operator_expression<unsigned int, 720> {
        [[deprecated("Use of SM<720> (Xavier architecture) has been deprecated and support will be removed in the future.")]] SM(){}
    };

    template<>
    struct SM<750>: public detail::constant_operator_expression<unsigned int, 750> {};

    template<>
    struct SM<800>: public detail::constant_operator_expression<unsigned int, 800> {};

    template<>
    struct SM<860>: public detail::constant_operator_expression<unsigned int, 860> {};

    template<>
    struct SM<870>: public detail::constant_operator_expression<unsigned int, 870> {};

    template<>
    struct SM<890>: public detail::constant_operator_expression<unsigned int, 890> {};

    template<>
    struct SM<900>: public detail::constant_operator_expression<unsigned int, 900> {};

    template<>
    struct SM<1000>: public detail::constant_operator_expression<unsigned int, 1000> {};

    template<>
    struct SM<1010>: public detail::constant_operator_expression<unsigned int, 1010> {}
    ;
    template<>
    struct SM<1030>: public detail::constant_operator_expression<unsigned int, 1030> {};

    template<>
    struct SM<1200>: public detail::constant_operator_expression<unsigned int, 1200> {};

    template<>
    struct SM<1210>: public detail::constant_operator_expression<unsigned int, 1210> {};
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_SM_HPP
