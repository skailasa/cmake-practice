#ifndef CUSOLVERDX_DATABASE_TRSM_CUH
#define CUSOLVERDX_DATABASE_TRSM_CUH

#include "cusolverdx/database/indexing.cuh"

namespace cusolverdx {
    namespace detail {
        namespace trsm {

            template<class T>
            __device__ void thread_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, T* rmem, const unsigned M, const unsigned N, const bool unit_diag, const bool trans_a, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);
            // Designed for small N
            template<class T>
            __device__ void warp_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, unsigned thread_id, T* rmem1, T* rmem2, T* rmem3, const unsigned NT, const unsigned M, const unsigned N, const bool unit_diag, const bool trans_a, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);
            // Designed for small N
            template<class T>
            __device__ void cta_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, unsigned thread_id, T* rmem1, T* rmem2, const unsigned M, const unsigned N, const unsigned NT, const bool unit_diag, const bool trans_a, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);

        } // namespace trsm

        // B is MxN
        // if side left, A is MxM, else A is NxN
        // Transpose is evaluated after Fill
        template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned Batches>
        inline __device__ void trsm_dispatch(const T* A, const unsigned lda, const unsigned strideA, T* B, const unsigned ldb, const unsigned strideB, const unsigned thread_id) {
            constexpr bool is_left = (Side == side::left);
            constexpr bool unit_diag = (Diag == diag::unit);

            constexpr bool even_warps = NT % 32 == 0;
            constexpr int  num_warps  = NT / 32;

            // If right side requested, transpose both matrices
            constexpr bool            trans_a = !is_left;
            constexpr bool            conj_a  = Transpose == transpose::conj_transposed;
            constexpr bool            trans_b = !is_left;
            constexpr unsigned impl_M  = is_left ? M : N;
            constexpr unsigned impl_N  = is_left ? N : M;

            // left, non-transposed, lower uses forward substitution
            // toggling any property toggles the type of substitution
            // Thus, we use XOR to check if we need backward substitution
            // != is equivalent to XOR for bools
            constexpr bool fwd = is_left != (Transpose == transpose::non_transposed) != (Fill == fill_mode::lower);

            if (Batches > 1) {
                // Handle batched dispatch separately from non-batched

                // TODO try to do multiple RHSs together where possible
                if (Batches * impl_N >= NT || impl_M < 8) {
                    // At least 1 RHS per thread or tiny matrices
                    for (int j = thread_id; j < Batches * impl_N; j += NT) {
                        unsigned batch = j / impl_N;
                        unsigned RHS   = j % impl_N;

                        auto Aj = A + batch * strideA;
                        auto Bj = B + batch * strideB;
                        auto bj = &index<T>(Bj, ldb, 0, RHS, trans_b, ArrangeB);

                        T rmem[impl_M];

                        trsm::thread_impl<T>(Aj, lda, bj, ldb, rmem, impl_M, 1, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                    }
                } else if constexpr (even_warps && Batches * impl_N >= num_warps) {
                    // At least 1 RHS per warp
                    unsigned warp_id = thread_id / 32;
                    unsigned lane_id = thread_id % 32;
                    for (int j = warp_id; j < Batches * impl_N; j += num_warps) {
                        unsigned batch = j / impl_N;
                        unsigned RHS   = j % impl_N;

                        auto Aj = A + batch * strideA;
                        auto Bj = B + batch * strideB;
                        auto bj = &index<T>(Bj, ldb, 0, RHS, trans_b, ArrangeB);

                        constexpr unsigned nrows = (impl_M + 31) / 32;

                        T rmem1[nrows];
                        T rmem2[1];
                        T rmem3[impl_M];
                        trsm::warp_impl<T>(Aj, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, 32, impl_M, 1, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                    }
                } else {
                    // Just use the full CTA
                    for (int j = 0; j < Batches; ++j) {
                        auto Aj = A + j * strideA;
                        auto Bj = B + j * strideB;

                        constexpr unsigned nrows = (impl_M + NT - 1) / NT;

                        T rmem1[nrows * impl_N];
                        T rmem2[impl_N];
                        trsm::cta_impl<T>(Aj, lda, Bj, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                    }
                }

            } else if constexpr (NT / impl_N > 40 && impl_M > 32) {
                // Have many threads per RHS and more rows than a single warp's threads.
                // Use full CTA synchronously

                constexpr unsigned nrows = (impl_M + NT - 1) / NT;

                T rmem1[nrows * impl_N];
                T rmem2[impl_N];
                trsm::cta_impl<T>(A, lda, B, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);

            } else if constexpr (NT >= 32 && impl_N < NT && impl_M >= 8) {
                // Have at least a warp, fewer threads than RHSs, and the matrix isn't tiny
                // Split the problem across warps
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;
                if constexpr (even_warps && impl_N % num_warps == 0 && impl_N >= num_warps) {
                    // Can split the RHS's nicely

                    // NB, because impl_N < NT, we have N_per_war < 32
                    constexpr unsigned N_per_warp   = impl_N / num_warps;
                    // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps
                    constexpr unsigned num_sub_warp = N_per_warp & (~(N_per_warp - 1));
                    static_assert(num_sub_warp == 1 || num_sub_warp == 2 || num_sub_warp == 4 || num_sub_warp == 8 || num_sub_warp == 16 || num_sub_warp == 32);
                    static_assert(N_per_warp % num_sub_warp == 0);
                    constexpr unsigned sub_warp_NT  = 32 / num_sub_warp;
                    constexpr unsigned sub_warp_N   = N_per_warp / num_sub_warp;

                    unsigned j = sub_warp_N * (thread_id / sub_warp_NT);
                    T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                        constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;

                    T rmem1[nrows * sub_warp_N];
                    T rmem2[sub_warp_N];
                    T rmem3[impl_M];
                    trsm::warp_impl<T>(A, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, sub_warp_NT, impl_M, sub_warp_N, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                } else {
                    // Just use a single warp
                    if (warp_id == 0) {
                        // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps
                        constexpr unsigned num_sub_warp = impl_N & (~(impl_N - 1));
                        constexpr unsigned sub_warp_NT  = 32 / num_sub_warp;
                        constexpr unsigned sub_warp_N   = impl_N / num_sub_warp;


                        unsigned j = sub_warp_N * (lane_id / sub_warp_NT);
                        T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                        constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;

                        T rmem1[nrows * sub_warp_N];
                        T rmem2[sub_warp_N];
                        T rmem3[impl_M];
                        trsm::warp_impl<T>(A, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, sub_warp_NT, impl_M, sub_warp_N, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                    }
                }

            } else if constexpr (2 * impl_N >= NT || impl_M < 8) {
                // Large number of RHSs or tiny matrix
                if constexpr (impl_N % NT == 0) {
                    // Can evenly divide RHSs to threads
                    // This has less smem IO than the next branch if impl_N > NT
                    constexpr unsigned thread_N = impl_N / NT;
                    unsigned           j        = thread_N * thread_id;

                    T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                    T rmem[impl_M * thread_N];
                    trsm::thread_impl<T>(A, lda, bj, ldb, rmem, impl_M, thread_N, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                } else {
                    // Just distribute RHSs round robin
                    for (int j = thread_id; j < impl_N; j += NT) {
                        T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                        T rmem[impl_M];
                        trsm::thread_impl<T>(A, lda, bj, ldb, rmem, impl_M, 1, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                    }
                }

            } else {
                // If other heuristics fail, just use the whole CTA synchronously
                constexpr unsigned nrows = (impl_M + NT - 1) / NT;

                T rmem1[nrows * impl_N];
                T rmem2[impl_N];
                trsm::cta_impl<T>(A, lda, B, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, trans_a, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
            }
        }

        // Wrapper for when batches are packed tightly
        template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned Batches>
        inline __device__ void trsm_dispatch(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {
            const unsigned strideA = (Side == side::left) ? M*lda : N*lda;
            const unsigned strideB = (ArrangeB == arrangement::col_major) ? ldb * N : M * ldb;
            trsm_dispatch<T, M, N, Side, Diag, Transpose, Fill, ArrangeA, ArrangeB, NT, Batches>(
                A, lda, strideA, B, ldb, strideB, thread_id);
        }


    } // namespace detail
} // namespace cusolverdx


#endif // CUSOLVERDX_DATABASE_TRSM_CUH
