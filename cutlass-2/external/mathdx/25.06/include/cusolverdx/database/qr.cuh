#ifndef CUSOLVERDX_DATABASE_QR_CUH
#define CUSOLVERDX_DATABASE_QR_CUH

#include "cusolverdx/database/qr_db.cuh"

namespace cusolverdx {
    namespace detail {

        namespace qr {

            template<class T>       
            __device__ void thread_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned Batches);

            template<class T>
            __device__ void partial_warp_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned Batches);

            template<class T>
            __device__ void warp_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned Batches);

            template<class T>
            __device__ void cta_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned Batches);

            template<class T>
            __device__ void conj_tau(T* tau, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned NT, const unsigned Batches);
    
            // TODO tune size thresholds.  Current values are just guesses
            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches>
            constexpr inline __device__ __host__ bool use_thread_per_batch() {
                // Other implementations require NT to be divisible by 32
                if (NT % 32 != 0) {
                    return true;
                }

                return M < 8 || M * N < 8 * 8;
            }

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches>
            constexpr inline __device__ __host__ bool use_partial_warp_per_batch() {
                constexpr bool     even_warps        = NT % 32 == 0;
                constexpr unsigned threads_per_batch = NT / Batches;

                if (!even_warps || NT % Batches != 0 || 32 % threads_per_batch != 0) {
                    // Can't align batches perfectly to partial warps
                    return false;
                }
                if (threads_per_batch > 32 || threads_per_batch <= 1) {
                    // Other drivers are better
                    return false;
                }

                return M*N <= 24*24;
            }

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches>
            constexpr inline __device__ __host__ bool use_warp_per_batch() {
                constexpr bool even_warps = NT % 32 == 0;

                if (NT == 32) {
                    return true;
                }

                return Batches > 1 && even_warps && NT >= 64 && M*N <= 64*64;
            }

        } // namespace qr

        template<class T, unsigned M, unsigned N, int Arch>
        constexpr inline __device__ __host__ unsigned qr_suggested_batches() {
            constexpr auto thresholds = qr::suggested_bpb_size_thresholds<T, Arch>();
            if (M * N <= thresholds[0]) {
                return 32;
            } else if (M * N <= thresholds[1]) {
                return 16;
            } else if (M * N <= thresholds[2]) {
                return 8;
            } else if (M * N <= thresholds[3]) {
                return 4;
            } else if (M * N <= thresholds[4]) {
                return 2;
            } else {
                return 1;
            }
        }


        template<class T, unsigned M, unsigned N, unsigned Batches, int Arch>
        constexpr inline __device__ __host__ dim3 qr_suggested_block_dim() {
            // Targets throughput bound cases

            if constexpr (Batches > 1) {
                // The suggested batch counts all work out to prefer 1 warp when the suggestion is greater than 1.
                constexpr unsigned ideal_batches_per_warp = qr_suggested_batches<T, M, N, Arch>();
                constexpr bool     target_partial_warp    = ideal_batches_per_warp > 1;

                if (M * N <= 8) {
                    return Batches <= 32 ? 32 : 64;

                } else if (target_partial_warp && Batches <= ideal_batches_per_warp) {
                    return 32;

                } else if (target_partial_warp && Batches % ideal_batches_per_warp == 0) {
                    return 32 * Batches / ideal_batches_per_warp;

                } else if (M * N <= (sizeof(T) <= 8 ? 64 * 64 : 32 * 32)) {
                    if (Batches == 2) {
                        return 64;
                    } else {
                        // Use 3 or 4 warps, depending on what results in fewer idle warps for the last wave of batches
                        unsigned rem_3 = (Batches % 3) ? 0 : 3 - (Batches % 3);
                        unsigned rem_4 = (Batches % 4) ? 0 : 4 - (Batches % 4);
                        return rem_3 < rem_4 ? 96 : 128;
                    }
                } else {
                    // For large sizes, just use suggestion for 1 batch per block.
                    return qr_suggested_block_dim<T, M, N, 1, Arch>();
                }
            } else {
                constexpr auto thresholds = qr::suggested_block_dim_size_thresholds<T, Arch>();
                if (M * N <= thresholds[0]) {
                    return 32;
                } else if (M * N <= thresholds[1]) {
                    return 64;
                } else if (M * N <= thresholds[2]) {
                    return 128;
                } else if (M * N <= thresholds[3]) {
                    return 256;
                } else {
                    return 512;
                }
            }
        }

        template<class T, int M, int N, arrangement Arrange, int NT, unsigned Batches = 1>
        inline __device__ void geqrf_dispatch(T* A, const unsigned lda, T* tau, const unsigned thread_id) {

            if constexpr (qr::use_thread_per_batch<T, M, N, NT, Batches>()) {
                qr::thread_driver<T>(A, lda, tau, thread_id, M, N, Arrange, NT, Batches);
            } else if constexpr (qr::use_partial_warp_per_batch<T, M, N, NT, Batches>()) {
                T rmem[M * N + N]; // + N for the workspace

                qr::partial_warp_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, Batches);
            } else if constexpr (qr::use_warp_per_batch<T, M, N, NT, Batches>()) {
                T rmem[M * N + N]; // + N for the workspace

                qr::warp_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, Batches);
            } else {
                static_assert(NT % 32 == 0);
                T rmem[M * N + M + N];

                qr::cta_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, Batches);
            }
            __syncthreads();
        }

        template<class T, int M, int N, arrangement Arrange, int NT, unsigned Batches = 1>
        inline __device__ void gelqf_dispatch(T* A, const unsigned lda, T* tau, const unsigned thread_id) {
            // GELQF is equivalent to doing GEQRF on the transpose, then conjugating tau
            // The arrangement is used to effect the transposition
            constexpr arrangement new_arr = (Arrange == col_major) ? row_major : col_major;
            geqrf_dispatch<T, N, M, new_arr, NT, Batches>(A, lda, tau, thread_id);
            if constexpr (!is_real_v<T>) {
                qr::conj_tau<T>(tau, M, N, thread_id, NT, Batches);
                __syncthreads();
            }
        }
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_QR_CUH
