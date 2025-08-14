#ifndef CUSOLVERDX_DATABASE_BATCH_POTRS_CUH
#define CUSOLVERDX_DATABASE_BATCH_POTRS_CUH

#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/database/indexing.cuh"

namespace cusolverdx {
    namespace detail {

        template<class T, unsigned N, unsigned K, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned Batches>
        inline __device__ void potrs_dispatch(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {

            constexpr transpose trans1 = (Fill == fill_mode::lower) ? transpose::non_transposed : transpose::conj_transposed;
            constexpr transpose trans2 = (Fill == fill_mode::lower) ? transpose::conj_transposed : transpose::non_transposed;

            trsm_dispatch<T, N, K, side::left, diag::non_unit, trans1, Fill, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
            trsm_dispatch<T, N, K, side::left, diag::non_unit, trans2, Fill, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
        }

        template<class T, unsigned N, unsigned K, arrangement ArrangeA, arrangement ArrangeB, transpose Trans, unsigned NT, unsigned Batches>
        inline __device__ void getrs_no_pivot_dispatch(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {

            if constexpr (Trans == non_trans) {
                trsm_dispatch<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
                trsm_dispatch<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
            } else {
                trsm_dispatch<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
                trsm_dispatch<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
            }
        }

        template<class T, unsigned N, unsigned K, arrangement ArrangeB, bool forward, unsigned NT, unsigned Batches>
        inline __device__ void laswp_dispatch(T* B, const unsigned ldb, const int* ipiv, const unsigned thread_id) {
            // 1 thread per RHS
            for (int j = thread_id; j < Batches * K; j += NT) {
                unsigned batch = j / K;
                unsigned RHS   = j % K;

                auto Bj = B + batch * (ArrangeB == arrangement::col_major ? ldb * K : N * ldb);
                auto bj = &index(Bj, ldb, 0, RHS, ArrangeB);

                constexpr int i_start = forward ? 0 : N - 1;
                constexpr int i_inc   = forward ? 1 : -1;
                for (int i = i_start; (forward ? i < N : i >= 0); i += i_inc) {
                    int piv = ipiv[i + batch * N] - 1;
                    if (i != piv) {
                        T temp                           = index(bj, ldb, i, 0, ArrangeB);
                        index(bj, ldb, i, 0, ArrangeB)   = index(bj, ldb, piv, 0, ArrangeB);
                        index(bj, ldb, piv, 0, ArrangeB) = temp;
                    }
                }
            }
        }

        template<class T, unsigned N, unsigned K, arrangement ArrangeA, arrangement ArrangeB, transpose Trans, unsigned NT, unsigned Batches>
        inline __device__ void getrs_partial_pivot_dispatch(const T* A, const unsigned lda, const int* ipiv, T* B, const unsigned ldb, const unsigned thread_id) {

            if constexpr (Trans == non_trans) {
                laswp_dispatch<T, N, K, ArrangeB, true, NT, Batches>(B, ldb, ipiv, thread_id);
                __syncthreads();    
            }
            
            if constexpr (Trans == non_trans) {
                trsm_dispatch<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
                trsm_dispatch<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
            } else {
                trsm_dispatch<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
                trsm_dispatch<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, Batches>(A, lda, B, ldb, thread_id);
            }
            
            if constexpr (Trans != non_trans) {
                __syncthreads();
                laswp_dispatch<T, N, K, ArrangeB, false, NT, Batches>(B, ldb, ipiv, thread_id);
            }
        }
    } // namespace detail
} // namespace cusolverdx

#endif
