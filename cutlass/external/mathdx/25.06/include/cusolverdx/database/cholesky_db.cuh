#ifndef CUSOLVERDX_DATABASE_CHOLESKY_DB_HPP
#define CUSOLVERDX_DATABASE_CHOLESKY_DB_HPP

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include <cuda/std/array>

namespace cusolverdx {
    namespace detail {
        namespace cholesky {

            // Size thresholds for suggesting batches per block
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 5> suggested_bpb_size_thresholds() {
                if (Arch >= 900) {
                    // Experimentally tuned for H100-PCIe
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return {5, 6, 7, 15, 16};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return {4, 6, 7, 12, 14};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return {3, 5, 6, 12, 16};
                    } else { // complex<double>
                        return {3, 5, 6, 12, 14};
                    }
                } else {
                    // Experimentally tuned for A100-PCIe
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return {3, 5, 7, 10, 12};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return {3, 4, 7, 10, 14};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return {3, 4, 7, 10, 14};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        return {2, 3, 6, 8, 14};
                    }
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ cuda::std::array<unsigned, 5> suggested_block_dim_size_thresholds() {
                constexpr unsigned INF = unsigned(-1);
                if (Arch >= 1000) {
                    // Scanned BlockDims for B200 computelab-next
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        // tuned for N=1:128:4
                        return {40, 72, 96, 116, 124};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        // tuned for N=1:128:4
                        return {48, 72, 84, 108, 128};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        // tuned for N=1:96:4
                        return {24, 56, 64, 96, INF};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        // tuned for N=1:96:4
                        return {24, 56, 80, 88, 96};
                    }
                } else if (Arch >= 900) {
                    // Experimentally tuned for H100-PCIe
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        // tuned for 1 <= N <= 128
                        return {43, 64, 111, INF, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        // tuned for 1 <= N <= 96
                        return {25, 78, INF, INF, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {25, 78, INF, INF, INF};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        // tuned for 1 <= N <= 64
                        return {24, 48, INF, INF, INF};
                    }
                } else {
                    // Experimentally tuned for A100-PCIe for sizes <= 100
                    // Larger sizes use SM90 tuning
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return {46, 72, 111, INF, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return {22, 72, INF, INF, INF};
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return {21, 57, INF, INF, INF};
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        return {25, 56, INF, INF, INF};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned tiny_threshold() {
                if (Arch >= 900) {
                    // Based on H100-PCIe
                    return is_real_v<T> ? 6 : 5;
                } else if (Arch == 860) {
                    // Based on RTX A6000
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return 6;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return 4;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return 5;
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        return 2;
                    }
                } else {
                    // Based on A100-PCIe
                    return is_real_v<T> ? 6 : 4;
                }
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned small_threshold() {
                if (Arch == 860) {
                    // Based on RTX A6000
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return 60;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return 60;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return 40;
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        return 50;
                    }
                } else {
                    // Based on H100-PCIe and A100-PCIe
                    // Cross-over point was the same for both cards
                    if (COMMONDX_STL_NAMESPACE::is_same_v<T, float>) {
                        return 54;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<T, double>) {
                        return 54;
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<real_type_t<T>, float>) { //complex<float>
                        return 40;
                    } else { // COMMONDX_STL_NAMESPACE::is_same_v<T, commondx::detail::complex<double>>
                        return 36;
                    }
                }
            }

            template<class T, unsigned Arch>
            constexpr inline __device__ __host__ unsigned med_threshold() {
                // Based on H100
                return 64;
            }

        } // namespace cholesky
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_CHOLESKY_DB_HPP
