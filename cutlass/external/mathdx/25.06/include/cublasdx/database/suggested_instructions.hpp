// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP
#define CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP

#include "cublasdx/database/cute_tensor.hpp"
#include "cublasdx/database/cute_utils.hpp"
#include "cublasdx/database/cute_tensor_configs.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {

            template<bool Condition, class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result;

            template<class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result<true, ResultIfTrue, MetaFunctionIfFalse> {
                using type = ResultIfTrue;
            };

            template<class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result<false, ResultIfTrue, MetaFunctionIfFalse> {
                using type = typename MetaFunctionIfFalse::type;
            };

            template<bool Condition, class ResultIfTrue, class MetaFunctionIfFalse>
            using get_result_t = typename get_result<Condition, ResultIfTrue, MetaFunctionIfFalse>::type;

            using warp_thread_layout_supermma = cute::Shape<cute::_8, cute::_4>;

            template<class Instruction, unsigned SMMin>
            struct instruction {
                using type = Instruction;
                static constexpr unsigned sm_min = SMMin;
            };

            template<class ... Instructions>
            struct instruction_list {};

            // ==========================================================
            // Return first (best) instruction fulfilling the SM criteria
            template<int SM, class WarpTile, class GEMMShape, class List>
            struct search_mma_list {
                using type = void;
            };

            template<int SM, class WarpTile, class GEMMShape, class Instruction, unsigned SMMin, class ... Elems>
            struct search_mma_list<SM, WarpTile, GEMMShape, instruction_list<instruction<Instruction, SMMin>, Elems...>> {
                using instruction_shape = typename cute::MMA_Traits<Instruction>::Shape_MNK;
                static constexpr int warp_tiled_m = cute::get<0>(cute::shape(WarpTile{})) * cute::get<0>(instruction_shape{});
                static constexpr int warp_tiled_n = cute::get<1>(cute::shape(WarpTile{})) * cute::get<1>(instruction_shape{});
                static constexpr int warp_tiled_k = cute::get<2>(cute::shape(WarpTile{})) * cute::get<2>(instruction_shape{});
                using warp_tiled_instruction_shape = cute::Shape<cute::Int<warp_tiled_m>, cute::Int<warp_tiled_n>, cute::Int<warp_tiled_k>>;
                static constexpr bool condition = (SM >= SMMin) and cute::evenly_divides(GEMMShape{}, warp_tiled_instruction_shape{});
                
                using type = get_result_t<condition, Instruction, search_mma_list<SM, WarpTile, GEMMShape, instruction_list<Elems...>>>;
            };

            template<int SM, int AtomBytes, int MaxMult, class List>
            struct search_ldst_list {
                using type = void;
            };

            template<int SM, int AtomBytes, int MaxMult, class Instruction, unsigned SMMin, class ... Elems>
            struct search_ldst_list<SM, AtomBytes, MaxMult, instruction_list<instruction<Instruction, SMMin>, Elems...>> {
                static constexpr int copy_size_bytes = cute::size(typename cute::Copy_Traits<Instruction>::RefLayout{}) / 8;
                static constexpr bool condition = (SM >= SMMin) and ((MaxMult % cute::ceil_div(copy_size_bytes, AtomBytes)) == 0);
                using type = get_result_t<condition, Instruction, search_ldst_list<SM, AtomBytes, MaxMult, instruction_list<Elems...>>>;
            };

            // ==========================================================

            template<class TA, class TB, class TC>
            struct portable_mma_instructions {
                using type = instruction_list<>;
            };

            // ==========================================================
            // Lists of superMMA instructions sorted by size (preference)

            template<>
            struct portable_mma_instructions<int8_t, int8_t, int32_t> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x32_S32S8S8S32_TN, 800>,
                    instruction<cute::SM80_16x8x16_S32S8S8S32_TN, 800>,
                    instruction<cute::SM80_8x8x16_S32S8S8S32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<int8_t, uint8_t, int32_t> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x32_S32S8U8S32_TN, 800>,
                    instruction<cute::SM80_16x8x16_S32S8U8S32_TN, 800>,
                    instruction<cute::SM80_8x8x16_S32S8U8S32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<uint8_t, int8_t, int32_t> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x32_S32U8S8S32_TN, 800>,
                    instruction<cute::SM80_16x8x16_S32U8S8S32_TN, 800>,
                    instruction<cute::SM80_8x8x16_S32U8S8S32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<uint8_t, uint8_t, int32_t> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x32_S32U8U8S32_TN, 800>,
                    instruction<cute::SM80_16x8x16_S32U8U8S32_TN, 800>,
                    instruction<cute::SM80_8x8x16_S32U8U8S32_TN, 800>
                >;
            };

            #if CUBLASDX_SUPPORTS_SM89_MMA

            template<>
            struct portable_mma_instructions<float_e4m3_t, float_e4m3_t, float> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F32E4M3E4M3F32_TN, 890>
                    #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA
                        ,
                        instruction<cublasdx::detail::SM89_16x8x16_F32E4M3E4M3F32_TN, 890>
                    #endif
                >;
            };

            template<>
            struct portable_mma_instructions<float_e4m3_t, float_e5m2_t, float> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F32E4M3E5M2F32_TN, 890>
                    #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA
                        ,
                        instruction<cublasdx::detail::SM89_16x8x16_F32E4M3E5M2F32_TN, 890>
                    #endif
                >;
            };

            template<>
            struct portable_mma_instructions<float_e5m2_t, float_e4m3_t, float> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F32E5M2E4M3F32_TN, 890>
                    #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA
                        ,
                        instruction<cublasdx::detail::SM89_16x8x16_F32E5M2E4M3F32_TN, 890>
                    #endif
                >;
            };

            template<>
            struct portable_mma_instructions<float_e5m2_t, float_e5m2_t, float> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F32E5M2E5M2F32_TN, 890>
                    #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA
                        ,
                        instruction<cublasdx::detail::SM89_16x8x16_F32E5M2E5M2F32_TN, 890>
                    #endif
                >;
            };

            #endif

            #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA

            template<>
            struct portable_mma_instructions<float_e4m3_t, float_e4m3_t, cute::half_t> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F16E4M3E4M3F16_TN, 890>,
                    instruction<cublasdx::detail::SM89_16x8x16_F16E4M3E4M3F16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<float_e4m3_t, float_e5m2_t, cute::half_t> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F16E4M3E5M2F16_TN, 890>,
                    instruction<cublasdx::detail::SM89_16x8x16_F16E4M3E5M2F16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<float_e5m2_t, float_e4m3_t, cute::half_t> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F16E5M2E4M3F16_TN, 890>,
                    instruction<cublasdx::detail::SM89_16x8x16_F16E5M2E4M3F16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<float_e5m2_t, float_e5m2_t, cute::half_t> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_F16E5M2E5M2F16_TN, 890>,
                    instruction<cublasdx::detail::SM89_16x8x16_F16E5M2E5M2F16_TN, 890>
                >;
            };

            #endif

            template<>
            struct portable_mma_instructions<cute::half_t, cute::half_t, cute::half_t> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x16_F16F16F16F16_TN, 800>,
                    instruction<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN, 750>,
                >;
            };

            template<>
            struct portable_mma_instructions<cute::half_t, cute::half_t, float> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x16_F32F16F16F32_TN, 800>,
                    instruction<cublasdx::detail::SM75_16x8x8_F32F16F16F32_TN, 750>
                >;
            };

            template<>
            struct portable_mma_instructions<bfloat16_t, bfloat16_t, float> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x16_F32BF16BF16F32_TN, 800>,
                    instruction<cute::SM80_16x8x8_F32BF16BF16F32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<tfloat32_t, tfloat32_t, float> {
                using type = instruction_list<
                    instruction<cute::SM80_16x8x8_F32TF32TF32F32_TN, 800>,
                    instruction<cute::SM80_16x8x4_F32TF32TF32F32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<double, double, double> {
                using type = instruction_list<
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 900)
                    instruction<cute::SM90_16x8x4_F64F64F64F64_TN, 900>,
                    instruction<cute::SM90_16x8x8_F64F64F64F64_TN, 900>,
                    instruction<cute::SM90_16x8x16_F64F64F64F64_TN, 900>,
                #endif
                    instruction<cute::SM80_8x8x4_F64F64F64F64_TN, 800>
                >;
            };

            // Complex types
            template<>
            struct portable_mma_instructions<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x32_CS32CS8CS8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x16_CS32CS8CS8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_8x8x16_CS32CS8CS8CS32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<int8_t>, cutlass::complex<uint8_t>, cutlass::complex<int32_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x32_CS32CS8CU8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x16_CS32CS8CU8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_8x8x16_CS32CS8CU8CS32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<uint8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x32_CS32CU8CS8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x16_CS32CU8CS8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_8x8x16_CS32CU8CS8CS32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<uint8_t>, cutlass::complex<uint8_t>, cutlass::complex<int32_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x32_CS32CU8CU8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x16_CS32CU8CU8CS32_TN, 800>,
                    instruction<cublasdx::detail::SM80_8x8x16_CS32CU8CU8CS32_TN, 800>
                >;
            };

            #if CUBLASDX_SUPPORTS_SM89_MMA

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e4m3_t>, cutlass::complex<float_e4m3_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C32CE4M3CE4M3C32_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e4m3_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C32CE4M3CE5M2C32_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e4m3_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C32CE5M2CE4M3C32_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C32CE5M2CE5M2C32_TN, 890>
                >;
            };

            #endif

            #if CUBLASDX_SUPPORTS_EXTENDED_FP8_MMA

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e4m3_t>, cutlass::complex<float_e4m3_t>, cutlass::complex<cute::half_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C16CE4M3CE4M3C16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e4m3_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<cute::half_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C16CE4M3CE5M2C16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e4m3_t>, cutlass::complex<cute::half_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C16CE5M2CE4M3C16_TN, 890>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<cute::half_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM89_16x8x32_C16CE5M2CE5M2C16_TN, 890>
                >;
            };

            #endif

            template<>
            struct portable_mma_instructions<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<half_t>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x16_C16C16C16C16_TN, 800>,
                    instruction<cublasdx::detail::SM75_16x8x8_C16C16C16C16_TN, 750>,
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x16_C32C16C16C32_TN, 800>,
                    instruction<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 750>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<bfloat16_t>, cutlass::complex<bfloat16_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x16_C32BC16BC16C32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x8_C32BC16BC16C32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<tfloat32_t>, cutlass::complex<tfloat32_t>, cutlass::complex<float>> {
                using type = instruction_list<
                    instruction<cublasdx::detail::SM80_16x8x8_C32TC32TC32C32_TN, 800>,
                    instruction<cublasdx::detail::SM80_16x8x4_C32TC32TC32C32_TN, 800>
                >;
            };

            template<>
            struct portable_mma_instructions<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>> {
                using type = instruction_list<
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 900)
                    instruction<cute::SM90_16x8x4_C64C64C64C64_TN, 900>,
                    instruction<cute::SM90_16x8x8_C64C64C64C64_TN, 900>,
                    instruction<cute::SM90_16x8x16_C64C64C64C64_TN, 900>,
                #endif
                    instruction<cute::SM80_8x8x4_C64C64C64C64_TN, 800>
                >;
            };

            template<class TA, class TB, class TC>
            using portable_mma_instructions_t = typename portable_mma_instructions<TA, TB, TC>::type;
            // ==========================================================

            // List of necessary copy operations
            // =================================
            using auto_vectorizing_copy = cute::AutoVectorizingCopyWithAssumedAlignment<128>;

            template<bool IsKMajor, class ElemType, class = void>
            struct ldsm_copy_instructions {
              using type = instruction_list<>;
            };

            template<bool IsKMajor, class ElemType, class = void>
            struct stsm_copy_instructions {
              using type = instruction_list<>;
            };

            template<class ElemType>
            struct is_ldsm_n_compatible_type {
                static constexpr bool value = 
                    cute::is_same_v<ElemType, int8_t> or
                    cute::is_same_v<ElemType, uint8_t> or
                    cute::is_same_v<ElemType, float_e4m3_t> or
                    cute::is_same_v<ElemType, float_e5m2_t> or
                    cute::is_same_v<ElemType, half_t> or
                    cute::is_same_v<ElemType, bfloat16_t> or
                    cute::is_same_v<ElemType, tfloat32_t>;
            };

            template<class ElemType>
            struct is_ldsm_t_compatible_type {
                static constexpr bool value = 
                    cute::is_same_v<ElemType, half_t> or
                    cute::is_same_v<ElemType, bfloat16_t>;
            };

            template<class ElemType>
            struct ldsm_copy_instructions<true, ElemType,
                cute::enable_if_t<is_ldsm_n_compatible_type<ElemType>::value>
              > {
              using type = instruction_list<
                instruction<cute::SM75_U32x4_LDSM_N, 750>,
                instruction<cute::SM75_U32x2_LDSM_N, 750>,
                instruction<cute::SM75_U32x1_LDSM_N, 750>
              >;
            };

            template<class ElemType>
            struct ldsm_copy_instructions<false, ElemType,
                cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>
            > {
              using type = instruction_list<
                instruction<cute::SM75_U16x8_LDSM_T, 750>,
                instruction<cute::SM75_U16x4_LDSM_T, 750>,
                instruction<cute::SM75_U16x2_LDSM_T, 750>
              >;
            };

#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
            // Warning: STSM cannot be always used, only when output precision matches input precision
            // in size and alignment
            template<class ElemType>
            struct stsm_copy_instructions<true, ElemType, 
                cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>
            > {
              using type = instruction_list<
                instruction<cute::SM90_U32x4_STSM_N, 900>,
                instruction<cute::SM90_U32x2_STSM_N, 900>,
                instruction<cute::SM90_U32x1_STSM_N, 900>
              >;
            };

            template<class ElemType>
            struct stsm_copy_instructions<false, ElemType,
                cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>
            > {
              using type = instruction_list<
                instruction<cute::SM90_U16x8_STSM_T, 900>,
                instruction<cute::SM90_U16x4_STSM_T, 900>,
                instruction<cute::SM90_U16x2_STSM_T, 900>
              >;
            };
#endif
            template<bool IsKMajor, class ElemType>
            using ldsm_copy_instructions_t = typename ldsm_copy_instructions<IsKMajor, ElemType>::type;

            template<bool IsKMajor, class ElemType>
            using stsm_copy_instructions_t = typename stsm_copy_instructions<IsKMajor, ElemType>::type;

            // =================================

            // Public interface
            // =================================
            template<int SM, class WarpTile, class GEMMShape, class TA, class TB, class TC>
            struct get_best_portable_mma_instruction {
                using instruction_list = portable_mma_instructions_t<TA, TB, TC>;
                using type = typename search_mma_list<SM, WarpTile, GEMMShape, instruction_list>::type;
            };
  
            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            struct get_best_ldst_instruction {
              static constexpr int atom_bytes = cute::get<1>(AtomShape{}) * cute::get<2>(AtomShape{}) * sizeof(ElemType);
              static constexpr int tile_m = cute::get<0>(TileShape{});
              // AtomShape needs to be <WarpM, TileM, TileN>
              static constexpr int tiled_mma_m = cute::get<0>(AtomShape{}) * cute::get<1>(AtomShape{});

              static constexpr bool is_predicated = tile_m % tiled_mma_m != 0;
              static constexpr bool is_complex = has_complex_interface<ElemType>();
              static constexpr bool unsupported = is_predicated or is_complex;
              static constexpr int max_mult = tile_m / tiled_mma_m;
              using load_type = get_result_t<unsupported, void, search_ldst_list<SM, atom_bytes, max_mult, ldsm_copy_instructions_t<IsKMajor, ElemType>>>;
              using store_type = get_result_t<unsupported, void, search_ldst_list<SM, atom_bytes, max_mult, stsm_copy_instructions_t<IsKMajor, ElemType>>>;
            };

            // Aliases
            template<int SM, class WarpTile,class GEMMShape, class TA, class TB, class TC>
            using get_best_portable_mma_instruction_t = typename get_best_portable_mma_instruction<SM, WarpTile, GEMMShape, TA, TB, TC>::type;

            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            using get_best_ldsm_instruction_t = typename get_best_ldst_instruction<SM, TileShape, AtomShape, ElemType, IsKMajor>::load_type;

            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            using get_best_stsm_instruction_t = typename get_best_ldst_instruction<SM, TileShape, AtomShape, ElemType, IsKMajor>::store_type;
        }
    }
}

#endif // CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP
