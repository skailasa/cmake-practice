#ifndef CUSOLVERDX_DATABASE_UNMQR_CUH
#define CUSOLVERDX_DATABASE_UNMQR_CUH


namespace cusolverdx {
    namespace detail {

        namespace unmqr {
            // The Trans argument is the transposition of the mathematical Q requested by the user
            // The TransA argument is the transposition of the physical accesses of A.  This exists to handle QR and LQ with the same implementation

            template<class T>
            __device__ void thread_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned Batches, T* rmem);
            template<class T>
            __device__ void partial_warp_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned Batches, T* rmem);
            template<class T>
            __device__ void warp_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned Batches, T* rmem);


            template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, transpose TransA, arrangement ArrangeB, unsigned NT, unsigned Batches=1>
            inline __device__ void actual_dispatch(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {

                constexpr unsigned L = (Side == side::left) ? M : N; // vector length, used for dispatch logic

                constexpr bool even_warps = NT % 32 == 0;
                constexpr unsigned  num_warps  = NT / 32;

                constexpr unsigned BM = M;
                constexpr unsigned BN = N;
                constexpr unsigned AM = (Side == side::left) ? M : N;

                constexpr unsigned threads_per_batch = NT / Batches;

                if constexpr (Batches > 1) {
                    // TODO need to work on these
                    if constexpr (Batches * K >= NT || L < 8) {
                        // At least 1 RHS per thread or tiny matrices
                        T rmem[AM + BM * BN];
                        thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                    } else if constexpr (even_warps && Batches > num_warps && NT % Batches == 0 && 32 % threads_per_batch == 0) {
                        T rmem[BM * BN + BM + BN];
                        partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                    } else if constexpr (even_warps) {
                        // At least 1 RHS per warp
                        static_assert(NT % 32 == 0);
                        T rmem[BM * BN + BM + BN];
                        warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                    } else {
                        // Irregular NT, only thread driver is reliable here
                        T rmem[AM + BM * BN];
                        thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                    }
                } else if constexpr (even_warps && K < NT && L >= 8) {
                    // Have at least a warp, more threads than RHSs, and the matrix isn't tiny
                    // Split the problem across warps
                    if constexpr (even_warps && K % num_warps == 0 && K > num_warps) {
                        if constexpr (NT % K == 0) {
                        // parallelizes across K per warp
                            T rmem[BM * BN + BM + BN];
                            partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                        } else {
                            static_assert(NT % 32 == 0);
                            T rmem[BM * BN + BM + BN];
                            warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, 32, Batches, rmem);
                        }
                    } else {
                        // Just use a single warp
                        if (thread_id < 32) {
                            if constexpr (NT % K == 0) {
                                T rmem[BM * BN + BM + BN];
                                partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, 32, Batches, rmem);
                            } else {
                                static_assert(NT % 32 == 0);
                                T rmem[BM * BN + BM + BN];
                                warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, 32, Batches, rmem);
                            }
                        }
                    }
                } else {
                    // Large number of RHSs or tiny matrix
                    T rmem[AM + BM * BN];
                    thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, Batches, rmem);
                }
                __syncthreads();
            }

        } // namespace unmqr

        template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned Batches=1>
        inline __device__ void unmqr_dispatch(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {
            return unmqr::actual_dispatch<T, M, N, K, Side, Trans, ArrangeA, transpose::non_transposed, ArrangeB, NT, Batches>(A, lda, tau, B, ldb, thread_id);
        }
        template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned Batches=1>
        inline __device__ void unmlq_dispatch(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {
            constexpr auto NewTrans = Trans == transpose::non_transposed ? transpose::conj_transposed : transpose::non_transposed;
            return unmqr::actual_dispatch<T, M, N, K, Side, NewTrans, ArrangeA, transpose::conj_transposed, ArrangeB, NT, Batches>(A, lda, tau, B, ldb, thread_id);
        }
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UNMQR_CUH
