// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CONFIGS_HPP
#define CUBLASDX_DATABASE_CONFIGS_HPP

#include "commondx/detail/stl/tuple.hpp"
#include "commondx/type_list.hpp"

#include "cublasdx/database/cute_tensor_configs.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {
            namespace database {

template<int BlockDim, mma_atom MMAAtom, int TileX, int TileY>
struct generated_config_impl {
    static constexpr mma_atom mma      = MMAAtom;
    static constexpr auto     tiles    = cute::make_tuple(TileX, TileY);
    static constexpr int      blockdim = BlockDim;
};

template<typename... ListElements>
struct generated_config_list {
    static constexpr bool defined = true;
    using list                    = commondx::type_list<ListElements...>;
};

template<typename AType,
         typename BType,
         typename CType,
         int                        M,
         int                        N,
         int                        K,
         typename SM>
struct generated_config {
    static constexpr bool defined = false;
};

#define CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(SMValue)                                      \
    template<typename AType, typename BType, typename CType>                                  \
    struct generated_config<AType, BType, CType, 4, 4, 4, SM<SMValue>>:                       \
        generated_config_list<generated_config_impl<64, mma_atom::universal_fma, 4, 16>,      \
                              generated_config_impl<128, mma_atom::universal_fma, 4, 32>,     \
                              generated_config_impl<256, mma_atom::universal_fma, 4, 64>,     \
                              generated_config_impl<512, mma_atom::universal_fma, 4, 128>,    \
                              generated_config_impl<1024, mma_atom::universal_fma, 4, 256>> { \
    };                                                                                        \
    template<typename AType, typename BType, typename CType>                                  \
    struct generated_config<AType, BType, CType, 8, 8, 8, SM<SMValue>>:                       \
        generated_config_list<generated_config_impl<64, mma_atom::universal_fma, 8, 8>,       \
                              generated_config_impl<128, mma_atom::universal_fma, 8, 16>,     \
                              generated_config_impl<256, mma_atom::universal_fma, 8, 32>,     \
                              generated_config_impl<512, mma_atom::universal_fma, 8, 64>,     \
                              generated_config_impl<1024, mma_atom::universal_fma, 8, 128>> { \
    };

CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(700)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(720)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(750)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(800)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(860)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(870)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(890)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(900)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(1000)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(1010)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(1030)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(1200)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(1210)

#undef CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS
#undef CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1

// SM70
#include "cublasdx/database/sm70/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm70/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm70/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm70/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm70/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm70/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm70/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm70/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm70/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm70/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm70/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm70/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm70/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm70/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm70/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm70/cint8_cint8_cint32_tn.hpp.inc"

// SM75
#include "cublasdx/database/sm75/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm75/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm75/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm75/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm75/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm75/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm75/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm75/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm75/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm75/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm75/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm75/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm75/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm75/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm75/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm75/cint8_cint8_cint32_tn.hpp.inc"

// SM80
#include "cublasdx/database/sm80/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm80/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm80/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm80/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm80/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm80/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm80/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm80/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm80/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm80/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm80/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm80/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm80/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm80/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm80/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm80/cint8_cint8_cint32_tn.hpp.inc"

// SM86
#include "cublasdx/database/sm86/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm86/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm86/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm86/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm86/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm86/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm86/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm86/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm86/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm86/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm86/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm86/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm86/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm86/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm86/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm86/cint8_cint8_cint32_tn.hpp.inc"

// SM89
// Real
#include "cublasdx/database/sm89/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm89/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm89/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm89/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm89/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm89/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm89/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm89/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm89/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm89/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm89/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm89/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm89/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm89/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm89/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm89/cint8_cint8_cint32_tn.hpp.inc"

// SM90
// Real
#include "cublasdx/database/sm90/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm90/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm90/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm90/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm90/fp32_fp32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm90/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm90/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm90/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm90/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm90/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm90/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm90/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm90/cfp32_cfp32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm90/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm90/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm90/cint8_cint8_cint32_tn.hpp.inc"

// SM100
// Real
#include "cublasdx/database/sm100/e5m2_e5m2_fp16_tn.hpp.inc"
#include "cublasdx/database/sm100/e5m2_e5m2_fp32_tn.hpp.inc"
#include "cublasdx/database/sm100/fp16_fp16_fp16_tn.hpp.inc"
#include "cublasdx/database/sm100/fp16_fp16_fp32_tn.hpp.inc"
#include "cublasdx/database/sm100/fp32_fp32_fp32_nn.hpp.inc"
#include "cublasdx/database/sm100/fp64_fp64_fp64_tn.hpp.inc"
#include "cublasdx/database/sm100/tf32_tf32_fp32_tn.hpp.inc"
#include "cublasdx/database/sm100/int8_int8_int32_tn.hpp.inc"

// Complex
#include "cublasdx/database/sm100/ce5m2_ce5m2_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm100/ce5m2_ce5m2_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm100/cfp16_cfp16_cfp16_tn.hpp.inc"
#include "cublasdx/database/sm100/cfp16_cfp16_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm100/cfp32_cfp32_cfp32_nn.hpp.inc"
#include "cublasdx/database/sm100/cfp64_cfp64_cfp64_tn.hpp.inc"
#include "cublasdx/database/sm100/ctf32_ctf32_cfp32_tn.hpp.inc"
#include "cublasdx/database/sm100/cint8_cint8_cint32_tn.hpp.inc"

// Forward SM72 to SM70
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<720>>:
    public generated_config<AType, BType, CType, M, N, K, SM<700>> { };

// Forward SM87 to SM80
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<870>>:
    public generated_config<AType, BType, CType, M, N, K, SM<800>> { };

// Forward SM101 to SM100
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<1010>>:
    public generated_config<AType, BType, CType, M, N, K, SM<1000>> { };

// Forward SM103 to SM100
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<1030>>:
    public generated_config<AType, BType, CType, M, N, K, SM<1000>> { };

// Forward SM120 to SM89
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<1200>>:
    public generated_config<AType, BType, CType, M, N, K, SM<890>> { };

// Forward SM121 to SM120
template<typename AType, typename BType, typename CType, int M, int N, int K>
struct generated_config<AType, BType, CType, M, N, K, SM<1210>>:
    public generated_config<AType, BType, CType, M, N, K, SM<890>> { };
            } // namespace database
        }     // namespace cute_backend
    }         // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CONFIGS_HPP
