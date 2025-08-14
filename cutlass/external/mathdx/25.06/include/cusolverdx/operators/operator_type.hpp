// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP

namespace cusolverdx {
    enum class operator_type
    {
        size,
        function,
        precision,
        type,
        fill_mode, // only used by trsm, potrf, potrs, posv
        leading_dimension,
        arrangement,
        transpose, // only used by trsm, getrs, gesv, unmqr, unmlq, gels
        diag, // only used by trsm
        side, // only used by trsm, unmqr, unmlq
        sm,
        // execution
        thread,
        block,
        // block only
        block_dim,
        batches_per_block
    };
} // namespace cusolverdx

#endif // CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP
