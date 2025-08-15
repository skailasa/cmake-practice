#ifndef CUSOLVERDX_DATABASE_LU_PP_CUH
#define CUSOLVERDX_DATABASE_LU_PP_CUH

#include "cusolverdx/database/lu_pp_db.cuh"

#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx {
    namespace detail {
        namespace lu_pp {

            template<class T>
            __device__ void thread_driver(T*                A,
                                          const unsigned    lda,
                                          int*              ipiv,
                                          int*              info,
                                          const unsigned    thread_id,
                                          const unsigned    M,
                                          const unsigned    N,
                                          const arrangement Arrange,
                                          const unsigned    NT,
                                          const unsigned    Batches,
                                          T*                rmem);

            template<class T>
            __device__ void partial_warp_driver(T*                A,
                                                const unsigned    lda,
                                                int*              ipiv,
                                                int*              info,
                                                const unsigned    thread_id,
                                                const unsigned    M,
                                                const unsigned    N,
                                                const arrangement Arrange,
                                                const unsigned    NT,
                                                const unsigned    Batches,
                                                const unsigned    p,
                                                const unsigned    q,
                                                T*                A_col,
                                                T*                A_row);

            template<class T>
            __device__ void warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              ipiv,
                                        int*              info,
                                        const unsigned    thread_id,
                                        const unsigned    M,
                                        const unsigned    N,
                                        const arrangement Arrange,
                                        const unsigned    NT,
                                        const unsigned    Batches,
                                        const unsigned    p,
                                        const unsigned    q,
                                        T*                A_col,
                                        T*                A_row);

            template<class T>
            __device__ void cta_driver(T*                A,
                                       const unsigned    lda,
                                       int*              ipiv,
                                       int*              info,
                                       const unsigned    thread_id,
                                       const unsigned    M,
                                       const unsigned    N,
                                       const arrangement Arrange,
                                       const unsigned    NT,
                                       const unsigned    Batches,
                                       const unsigned    p,
                                       const unsigned    q,
                                       T*                A_col,
                                       T*                A_row);

            // TODO tune size thresholds.  Current values are just guesses
            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches>
            constexpr inline __device__ __host__ bool use_thread_per_batch() {
                return M * N < 8 * 8;
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

                return M * N <= 24 * 24;
            }

            template<class T, unsigned M, unsigned N, unsigned NT, unsigned Batches>
            constexpr inline __device__ __host__ bool use_warp_per_batch() {
                constexpr bool even_warps = NT % 32 == 0;

                if (NT == 32) {
                    return true;
                }

                return Batches > 1 && even_warps && NT >= 64 && M * N <= 64 * 64;
            }
        } // namespace lu_pp

        template<class T, unsigned M, unsigned N, int Arch>
        constexpr inline __device__ __host__ unsigned lu_pp_suggested_batches_per_warp() {
            constexpr auto thresholds = lu_pp::suggested_bpb_size_thresholds<T, Arch>();
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

        template<class T, unsigned M, unsigned N, int Arch>
        constexpr inline __device__ __host__ unsigned lu_pp_suggested_batches() {
            auto per_warp = lu_pp_suggested_batches_per_warp<T, M, N, Arch>();
            constexpr bool have_large_smem = (Arch == 800) || (Arch == 870) || (Arch == 900) || (Arch == 1000);
            if (per_warp != 1 && have_large_smem) {
                return 2 * per_warp;
            } else {
                return per_warp;
            }
        }

        template<class T, unsigned M, unsigned N, unsigned Batches, int Arch>
        constexpr inline __device__ __host__ dim3 lu_pp_suggested_block_dim() {
            // Targets throughput bound cases

            if constexpr (Batches > 1) {
                constexpr unsigned ideal_batches_per_warp = lu_pp_suggested_batches_per_warp<T, M, N, Arch>();
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
                    return lu_pp_suggested_block_dim<T, M, N, 1, Arch>();
                }
            } else {
                constexpr auto thresholds = lu_pp::suggested_block_dim_size_thresholds<T, Arch>();
                if (M * N <= thresholds[0]) {
                    return 32;
                } else if (M * N <= thresholds[1]) {
                    return 64;
                } else if (M * N <= thresholds[2]) {
                    return 128;
                } else {
                    return 256;
                }
            }
        }


        template<class T, unsigned M, unsigned N, arrangement Arrange, unsigned NT, unsigned Batches>
        inline __device__ void lu_pp_dispatch(T* A, const unsigned lda, int* ipiv, int* info, const unsigned thread_id) {

            if constexpr (lu_pp::use_thread_per_batch<T, M, N, NT, Batches>()) {
                T rmem[N];
                lu_pp::thread_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, Batches, rmem);
                __syncthreads();

            } else if constexpr (lu_pp::use_partial_warp_per_batch<T, M, N, NT, Batches>()) {
                constexpr unsigned threads_per_batch = NT / Batches;
                constexpr auto     pq                = pq_selector(M, N, threads_per_batch);
                constexpr unsigned p                 = pq.p;
                constexpr unsigned q                 = pq.q;
                static_assert(p * q == threads_per_batch);

                constexpr unsigned BatchesPerWarp = 32 / threads_per_batch;
                static_assert(BatchesPerWarp != 0);
                static_assert(BatchesPerWarp < 32);
                static_assert(32 % BatchesPerWarp == 0);

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                T A_col_rmem[nrows];
                T A_row_rmem[ncols];

                lu_pp::partial_warp_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, Batches, p, q, A_col_rmem, A_row_rmem);
                __syncthreads();

            } else if constexpr (lu_pp::use_warp_per_batch<T, M, N, NT, Batches>()) {
                constexpr auto     pq = pq_selector(M, N, 32);
                constexpr unsigned p  = pq.p;
                constexpr unsigned q  = pq.q;
                static_assert(p * q == 32);

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                T A_col_rmem[nrows];
                T A_row_rmem[ncols];

                lu_pp::warp_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, Batches, p, q, A_col_rmem, A_row_rmem);
                __syncthreads();

            } else {
                constexpr auto     pq = pq_selector(M, N, NT);
                constexpr unsigned p  = pq.p;
                constexpr unsigned q  = pq.q;
                static_assert(p * q == NT);

                constexpr unsigned nrows = (M + p - 1) / p;
                constexpr unsigned ncols = (N + q - 1) / q;

                T A_col_rmem[nrows];
                T A_row_rmem[ncols];

                lu_pp::cta_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, Batches, p, q, A_col_rmem, A_row_rmem);
            }
        }
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_LU_PP_CUH
