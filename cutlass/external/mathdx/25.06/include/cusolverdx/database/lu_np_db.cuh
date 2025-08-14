#ifndef CUSOLVERDX_DATABASE_LU_NP_DB_CUH
#define CUSOLVERDX_DATABASE_LU_NP_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include <cuda/std/array>

// Tuning factors are currently borrowed in part from Cholesky
#include "cusolverdx/database/cholesky_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace lu_np {

            // Size thresholds for suggesting batches per block
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 5> suggested_bpb_size_thresholds() {
                // Experimentally tuned for H100-PCIe
                if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                    return {5*5, 6*6, 7*7, 15*15, 32*32};
                } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                    return {5*5, 6*6, 9*9, 15*15, 32*32};
                } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                    return {5*5, 13*13, 16*16, 20*20, 28*28};
                } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                    return {5*5, 9*9, 14*14, 18*18, 21*21};
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 4> suggested_block_dim_size_thresholds() {
                constexpr unsigned INF = unsigned(-1);
                if (Arch >= 1000) {
                    // Scanned BlockDims for B200 computelab-next
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        // tuned for 1 <= N <= 128
                        return {53*53, 72*72, 95*95, 115*115};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        // tuned for 1 <= N <= 128
                        return {48*48, 63*63, 99*99, 115*115}; 
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {28*28, 59*59, 67*67, 80*80};
                    } else { // complex<double>>
                        // tuned for 1 <= N <= 96
                        return {23*23, 43*43, 75*75, 80*80}; 
                    }
                } else {
                    // Experimentally tuned for H100-PCIe
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        // tuned for 1 <= N <= 128
                        return {53*53, 72*72, 95*95, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        // tuned for 1 <= N <= 128
                        return {48*48, 63*63, 88*88, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {16*16, 56*56, 80*80, INF};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        // tuned for 1 <= N <= 96
                        return {16*16, 40*40, 56*56, INF};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned tiny_threshold() {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = cholesky::tiny_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned small_threshold() {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = cholesky::small_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned med_threshold() {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = cholesky::med_threshold<T, Arch>();
                return chol_tol * chol_tol;
            }

        } // namespace lu_np
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_LU_NP_DB_CUH
