#ifndef CUSOLVERDX_DATABASE_CHOLESKY_CUH
#define CUSOLVERDX_DATABASE_CHOLESKY_CUH

#include "commondx/complex_types.hpp"

#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/database/cholesky_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace cholesky {

            ////////// Cholesky Driver functions //////////
            template<class T>
            __device__ void thread_driver(T*                A,
                                          const unsigned    lda,
                                          int*              info,
                                          unsigned          thread_id,
                                          T*                rmem,
                                          const unsigned    N,
                                          const unsigned    NT,
                                          const unsigned    Batches,
                                          const fill_mode   Fill,
                                          const arrangement Arrange);

            template<class T>
            __device__ void partial_warp_driver(T*                A,
                                                const unsigned    lda,
                                                int*              info,
                                                unsigned          thread_id,
                                                T*                rmem1,
                                                T*                rmem2,
                                                T*                rmem3,
                                                const unsigned    N,
                                                const unsigned    NT,
                                                const unsigned    Batches,
                                                const unsigned    p,
                                                const unsigned    q,
                                                const fill_mode   Fill,
                                                const arrangement Arrange);

            template<class T>
            __device__ void warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              info,
                                        unsigned          thread_id,
                                        T*                rmem1,
                                        T*                rmem2,
                                        T*                rmem3,
                                        const unsigned    N,
                                        const unsigned    NT,
                                        const unsigned    Batches,
                                        const unsigned    p,
                                        const unsigned    q,
                                        const fill_mode   Fill,
                                        const arrangement Arrange);

            template<class T>
            __device__ void cta_driver(T*                A,
                                       const unsigned    lda,
                                       int*              info,
                                       unsigned          thread_id,
                                       T*                rmem1,
                                       T*                rmem2,
                                       T*                rmem3,
                                       const unsigned    N,
                                       const unsigned    NT,
                                       const unsigned    Batches,
                                       const unsigned    p,
                                       const unsigned    q,
                                       const fill_mode   Fill,
                                       const arrangement Arrange);

            ////////// thresholds for implementation dispatch ////////////
            template<class T, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
            constexpr inline __device__ __host__ bool use_thread_per_batch() {
                // Check if matrix is small enough to always use thread per
                if (N <= tiny_threshold<T, Arch>())
                    return true;

                // If the number of batches is enough to saturate the threads, without excessive register strain
                // based on H100
                return Batches > (4 + sizeof(T)) * NT / 32 && N <= 8;
            }

            template<class T, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
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

                return N <= small_threshold<T, Arch>();
            }

            template<class T, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
            constexpr inline __device__ __host__ bool use_warp_per_batch() {
                constexpr bool even_warps = NT % 32 == 0;

                if (NT == 32) {
                    return true;
                }

                return Batches > 1 && even_warps && NT >= 64 && N <= med_threshold<T, Arch>();
            }

        } // namespace cholesky


        template<class T, unsigned N, int Arch>
        constexpr inline __device__ __host__ unsigned cholesky_suggested_batches() {
            constexpr auto thresholds = cholesky::suggested_bpb_size_thresholds<T, Arch>();
            if (N <= thresholds[0]) {
                return 32;
            } else if (N <= thresholds[1]) {
                return 16;
            } else if (N <= thresholds[2]) {
                return 8;
            } else if (N <= thresholds[3]) {
                return 4;
            } else if (N <= thresholds[4]) {
                return 2;
            } else {
                return 1;
            }
        }

        template<class T, unsigned N, unsigned Batches, int Arch>
        constexpr inline __device__ __host__ dim3 cholesky_suggested_block_dim() {
            // Targets throughput bound cases

            if constexpr (Batches > 1) {
                // The suggested batch counts all work out to prefer 1 warp when the suggestion is greater than 1.
                constexpr unsigned ideal_batches_per_warp = cholesky_suggested_batches<T, N, Arch>();
                constexpr bool     target_partial_warp    = ideal_batches_per_warp > 1;

                if (N <= 8) {
                    return Batches <= 32 ? 32 : 64;

                } else if (target_partial_warp && Batches <= ideal_batches_per_warp) {
                    return 32;

                } else if (target_partial_warp && Batches % ideal_batches_per_warp == 0) {
                    return 32 * Batches / ideal_batches_per_warp;

                } else if (N <= (sizeof(T) <= 8 ? 64 : 32)) {
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
                    return cholesky_suggested_block_dim<T, N, 1, Arch>();
                }
            } else {
                constexpr auto thresholds = cholesky::suggested_block_dim_size_thresholds<T, Arch>();
                if (N <= thresholds[0]) {
                    return 32;
                } else if (N <= thresholds[1]) {
                    return 64;
                } else if (N <= thresholds[2]) {
                    return 128;
                } else if (N <= thresholds[3]) {
                    return 256;
                } else if (N <= thresholds[4]){
                    return 320;
                } else {
                    return 256;
                }
            }
        }

        template<class T, unsigned N, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned Batches, unsigned Arch>
        inline __device__ void cholesky_dispatch(T* A, const unsigned lda, int* info, const unsigned thread_id) {

            if constexpr (cholesky::use_thread_per_batch<T, N, NT, Batches, Arch>()) {
                T rmem[N * N];

                cholesky::thread_driver<T>(A, lda, info, thread_id, rmem, N, NT, Batches, Fill, Arrange);
                __syncthreads();

            } else if constexpr (cholesky::use_partial_warp_per_batch<T, N, NT, Batches, Arch>()) {
                constexpr unsigned threads_per_batch = NT / Batches;

                constexpr unsigned p = threads_per_batch == 32 ? 8 : (threads_per_batch >= 8 ? 4 : 2);
                constexpr unsigned q = threads_per_batch >= 16 ? 4 : (threads_per_batch >= 4 ? 2 : 1);

                constexpr unsigned nrows = (N + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];

                cholesky::partial_warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, Batches, p, q, Fill, Arrange);
                __syncthreads();

                // If we only have 1 warp, just use the warp routine
            } else if constexpr (cholesky::use_warp_per_batch<T, N, NT, Batches, Arch>()) {
                constexpr unsigned p = 8;
                constexpr unsigned q = 4;

                constexpr unsigned nrows = (N + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];

                cholesky::warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, Batches, p, q, Fill, Arrange);
                __syncthreads();

            } else {
                constexpr unsigned p = (NT % 16 != 0) ? 1 : (NT <= 64 ? 8 : 16);
                constexpr unsigned q = NT / p;

                constexpr unsigned nrows = (N + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                static_assert(p * q == NT);
                static_assert(p <= 32); // column bcasts utilize warp shuffle instructions

                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];

                cholesky::cta_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, Batches, p, q, Fill, Arrange);
            }
        }

    } // namespace detail
} // namespace cusolverdx


#endif // CUSOLVERDX_DATABASE_CHOLESKY_CUH
