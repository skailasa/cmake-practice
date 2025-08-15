#ifndef CUSOLVERDX_DATABASE_INDEXING_CUH
#define CUSOLVERDX_DATABASE_INDEXING_CUH

namespace cusolverdx {
    namespace detail {

        template<class T>
        __device__ inline T& index_impl(T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            if (!transpose == (Arrange == col_major)) {
                return A[i + j * lda];
            } else {
                return A[i * lda + j];
            }
        }
        template<class T>
        __device__ inline const T& index_impl(const T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            if (!transpose == (Arrange == col_major)) {
                return A[i + j * lda];
            } else {
                return A[i * lda + j];
            }
        }

        template<class T>
        __device__ inline T& index(T* A, unsigned lda, int i, int j, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, false, Arrange);
        }

        template<class T>
        __device__ inline T& index(T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, transpose, Arrange);
        }

        // Index an lower-triangle element from a triangular or symmetric matrix
        template<class T>
        __device__ inline T& index_tri_lo(T* A, unsigned lda, int i, int j, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (Fill != fill_mode::lower), Arrange);
        }

        // Index an upper-triangle element from a triangular or symmetric matrix
        template<class T>
        __device__ inline T& index_tri_up(T* A, unsigned lda, int i, int j, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (Fill != fill_mode::upper), Arrange);
        }

        // Index an lower-triangle element from a triangular or symmetric matrix
        template<class T>
        __device__ inline T& index_tri_lo(T* A, unsigned lda, int i, int j, const bool transpose, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (transpose != (Fill != fill_mode::lower)), Arrange);
        }
        template<class T>
        __device__ inline const T& index_tri_lo(const T* A, unsigned lda, int i, int j, const bool transpose, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (transpose != (Fill != fill_mode::lower)), Arrange);
        }

        // Index an upper-triangle element from a triangular or symmetric matrix
        template<class T>
        __device__ inline T& index_tri_up(T* A, unsigned lda, int i, int j, const bool transpose, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (transpose != (Fill != fill_mode::upper)), Arrange);
        }
        template<class T>
        __device__ inline const T& index_tri_up(const T* A, unsigned lda, int i, int j, const bool transpose, const fill_mode Fill, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, (transpose != (Fill != fill_mode::upper)), Arrange);
        }

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_INDEXING_CUH
