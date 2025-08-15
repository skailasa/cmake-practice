#ifndef CUSOLVERDX_DATABASE_LU_NP_CUH
#define CUSOLVERDX_DATABASE_LU_NP_CUH

#include "cusolverdx/database/lu_np_db.cuh"

#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx {
    namespace detail {
        namespace lu_np {

            template<class T>
            __device__ void thread_driver(T* A, const unsigned lda, int* info, unsigned thread_id, T* rmem, unsigned M, unsigned N, unsigned NT, unsigned Batches, const arrangement Arrange);

            template<class T>
            __device__ void partial_warp_driver(T*                A,
                                                const unsigned    lda,
                                                int*              info,
                                                unsigned          thread_id,
                                                T*                rmem1,
                                                T*                rmem2,
                                                T*                rmem3,
                                                unsigned          M,
                                                unsigned          N,
                                                unsigned          NT,
                                                unsigned          Batches,
                                                const unsigned    p,
                                                const unsigned    q,
                                                const arrangement Arrange);

            template<class T>
            __device__ void warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              info,
                                        unsigned          thread_id,
                                        T*                rmem1,
                                        T*                rmem2,
                                        T*                rmem3,
                                        unsigned          M,
                                        unsigned          N,
                                        unsigned          NT,
                                        unsigned          Batches,
                                        const unsigned    p,
                                        const unsigned    q,
                                        const arrangement Arrange);

            template<class T>
            __device__ void cta_driver(T*                A,
                                       const unsigned    lda,
                                       int*              info,
                                       unsigned          thread_id,
                                       T*                rmem1,
                                       T*                rmem2,
                                       T*                rmem3,
                                       unsigned          M,
                                       unsigned          N,
                                       unsigned          NT,
                                       unsigned          Batches,
                                       const unsigned    p,
                                       const unsigned    q,
                                       const arrangement Arrange);

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
            constexpr inline __device__ __host__ bool use_thread_per_batch() {
                // Check if matrix is small enough to always use thread per
                constexpr unsigned tiny_thresh = tiny_threshold<T, Arch>();
                if (M * N <= tiny_thresh)
                    return true;

                // If the number of batches is enough to saturate the threads, without excessive register strain
                // Based on Cholesky tuning
                return Batches > (4 + sizeof(T)) * NT / 32 && M * N <= 64;
            }

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
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

                return M * N <= small_threshold<T, Arch>();
            }

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches, unsigned Arch>
            constexpr inline __device__ __host__ bool use_warp_per_batch() {
                constexpr bool even_warps = NT % 32 == 0;

                if (NT == 32) {
                    return true;
                }

                return Batches > 1 && even_warps && NT >= 64 && M * N <= med_threshold<T, Arch>();
            }
        } // namespace lu_np

        template<class T, unsigned M, unsigned N, int Arch>
        constexpr inline __device__ __host__ unsigned lu_np_suggested_batches() {
            constexpr auto thresholds = lu_np::suggested_bpb_size_thresholds<T, Arch>();
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
        constexpr inline __device__ __host__ dim3 lu_np_suggested_block_dim() {
            // Targets throughput bound cases

            if constexpr (Batches > 1) {
                // The suggested batch counts all work out to prefer 1 warp when the suggestion is greater than 1.
                constexpr unsigned ideal_batches_per_warp = lu_np_suggested_batches<T, M, N, Arch>();
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
                    return lu_np_suggested_block_dim<T, M, N, 1, Arch>();
                }
            } else {
                constexpr auto thresholds = lu_np::suggested_block_dim_size_thresholds<T, Arch>();
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

        template<class T, unsigned _M, unsigned _N, arrangement Arrange, unsigned _NT, unsigned _Batches, int Arch>
        inline __device__ void lu_np_dispatch(T* A, const unsigned lda, int* info, const unsigned thread_id) {
            static constexpr unsigned M       = _M;
            static constexpr unsigned N       = _N;
            static constexpr unsigned NT      = _NT;
            static constexpr unsigned Batches = _Batches;

            if constexpr (lu_np::use_thread_per_batch<T, M, N, NT, Batches, Arch>()) {
                T rmem[M * N];

                lu_np::thread_driver<T>(A, lda, info, thread_id, rmem, M, N, NT, Batches, Arrange);
                __syncthreads();

            } else if constexpr (lu_np::use_partial_warp_per_batch<T, M, N, NT, Batches, Arch>()) {
                static constexpr unsigned threads_per_batch = NT / Batches;
                static constexpr auto     pq                = pq_selector(M, N, threads_per_batch);
                static constexpr unsigned p                 = pq.p;
                static constexpr unsigned q                 = pq.q;

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;
                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];

                lu_np::partial_warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, Batches, p, q, Arrange);
                __syncthreads();

            } else if constexpr (lu_np::use_warp_per_batch<T, M, N, NT, Batches, Arch>()) {
                static constexpr auto     pq                = pq_selector(M, N, 32);
                static constexpr unsigned p                 = pq.p;
                static constexpr unsigned q                 = pq.q;

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];

                lu_np::warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, Batches, p, q, Arrange);
                __syncthreads();

            } else {
                static constexpr auto     pq                = pq_selector(M, N, NT);
                static constexpr unsigned p                 = pq.p;
                static constexpr unsigned q                 = pq.q;

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                // register memory
                T rmem1[nrows * ncols];
                T rmem2[nrows];
                T rmem3[ncols];
                lu_np::cta_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, Batches, p, q, Arrange);
            }
        }

    } // namespace detail
} // namespace cusolverdx


#endif //CUSOLVERDX_DATABASE_LU_NP_CUH
