#ifndef CUSOLVERDX_DATABASE_QR_DB_CUH
#define CUSOLVERDX_DATABASE_QR_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include <cuda/std/array>

// Tuning factors are currently borrowed in part from Cholesky
#include "cusolverdx/database/cholesky_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace qr {

            // Size thresholds for suggesting batches per block
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 5> suggested_bpb_size_thresholds() {
                if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                    return {5*5, 6*6, 7*7, 15*15, 32*32};
                } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                    return {5*5, 6*6, 9*9, 15*15, 24*24};
                } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, cuComplex>) {
                    return {5*5, 13*13, 16*16, 20*20, 28*28};
                } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, cuDoubleComplex>
                    return {5*5, 9*9, 14*14, 18*18, 21*21};
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 4> suggested_block_dim_size_thresholds() {
                if (Arch >= 1000) {
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        // tuned for 1 <= N <= 128
                        return {36*36, 48*48, 84*84, 88*88};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        // tuned for 1 <= N <= 128
                        return {28*28, 52*52, 72*72, 96*96};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, cuComplex>) {   
                        // tuned for 1 <= N <= 96
                        return {28*28, 36*36, 64*64, 88*88};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, cuDoubleComplex>
                        // tuned for 1 <= N <= 96
                        return {20*20, 32*32, 60*60, 84*84};
                    }
                } else { // Tuned for H100
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return {36*36, 64*64, 84*84, 88*88};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return {28*28, 63*63, 88*88, 100*100};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, cuComplex>) {
                        return {28*28, 48*48, 64*64, 88*88};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, cuDoubleComplex>
                        return {32*32, 40*40, 64*64, 96*96};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned tiny_threshold() {
                // Take tuning factors from Cholesky until qr can be tuned
                auto chol_tol = cholesky::tiny_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned small_threshold() {
                // Take tuning factors from Cholesky until qr can be tuned
                auto chol_tol = cholesky::small_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned med_threshold() {
                // Take tuning factors from Cholesky until qr can be tuned
                auto chol_tol = cholesky::med_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

        } // namespace qr
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_QR_DB_CUH
