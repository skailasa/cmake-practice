// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_TENSOR_HPP
#define CUBLASDX_DETAIL_TENSOR_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/tensor.hpp"

#include "cublasdx/database/cute_tensor.hpp"

namespace cublasdx {
    using commondx::tensor;
    using commondx::is_layout;
    using commondx::is_layout_v;
    using commondx::make_tensor;
    using commondx::cosize;
    using commondx::size;
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_TENSOR_HPP
