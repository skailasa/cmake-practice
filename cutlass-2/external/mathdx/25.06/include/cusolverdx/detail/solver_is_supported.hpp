// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_IS_SUPPORTED_HPP
#define CUSOLVERDX_DETAIL_IS_SUPPORTED_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/device_info.hpp"

#include "cusolverdx/operators.hpp"
#include "cusolverdx/traits.hpp"

namespace cusolverdx {
    namespace detail {

        template<class Precision, type Type, unsigned int ASize, unsigned int BSize = 0, unsigned int ExtraSize = 0, unsigned int ExtraBytes = 0, unsigned int Arch = 800>
        struct is_supported_shared_size {
            static constexpr unsigned int type_multiplier = (Type == cusolverdx::type::real) ? 1 : 2;
            static constexpr size_t       required_shared_memory =
                sizeof(typename Precision::a_type) * type_multiplier * (ASize) +
                sizeof(typename Precision::b_type) * type_multiplier * (BSize) +
                sizeof(typename Precision::a_type) * type_multiplier * (ExtraSize) +
                ExtraBytes;
            static constexpr bool value = (required_shared_memory <= commondx::device_info<Arch>::shared_memory());
        };

        // TODO: Right now it is used in suggested_leading_dimension_of, which needs to be reworked in the future
        template<class Precision,
                 type        Type,
                 arrangement Arr,
                 arrangement Brr,
                 class Size,
                 class LD, // Leading Dimension
                 unsigned int Arch>
        struct is_supported_impl {
            static constexpr auto lda = LD::a;
            static constexpr auto ldb = LD::b;
            static constexpr auto a_size = lda * ((Arr == arrangement::col_major) ? Size::n : Size::m);
            static constexpr auto b_size = ldb * ((Brr == arrangement::col_major) ? Size::k : Size::n);

            static constexpr bool value = is_supported_shared_size<Precision, Type, a_size, b_size, 0, 0, Arch>::value;
        };
    } // namespace detail

    // Check if a description is supported on a given CUDA architecture
    template<class Description, unsigned int Arch>
    struct is_supported:
        public COMMONDX_STL_NAMESPACE::bool_constant<
            detail::is_supported_shared_size<precision_of<Description>,
                                      type_of_v<Description>,
                                      matrix_size_of_v_a<Description>,
                                      matrix_size_of_v_b<Description>,
                                      matrix_size_of_v_extra_size<Description>,
                                      matrix_size_of_v_extra_bytes<Description>,
                                      Arch>::value> {};

    template<class Description, unsigned int Arch>
    inline constexpr bool is_supported_v = is_supported<Description, Arch>::value;
} // namespace cusolverdx

#endif // CUDescriptionDX_DETAIL_IS_SUPPORTED_HPP
