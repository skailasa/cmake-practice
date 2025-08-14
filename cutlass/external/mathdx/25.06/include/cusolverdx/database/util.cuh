#ifndef CUSOLVERDX_DATABASE_UTIL_CUH
#define CUSOLVERDX_DATABASE_UTIL_CUH

namespace cusolverdx {
    namespace detail {

        template<class T>
        struct is_real {
            static constexpr bool value = true;
        };
        template<>
        struct is_real<commondx::detail::complex<float>> {
            static constexpr bool value = false;
        };
        template<>
        struct is_real<commondx::detail::complex<double>> {
            static constexpr bool value = false;
        };
        template<>
        struct is_real<cuComplex> {
            static constexpr bool value = false;
        };
        template<>
        struct is_real<cuDoubleComplex> {
            static constexpr bool value = false;
        };
        template<class T>
        static constexpr bool is_real_v = is_real<T>::value;

        template<class T>
        struct real_type {
            using value_type = T;
        };
        template<>
        struct real_type<commondx::complex<float>> {
            using value_type = float;
        };
        template<>
        struct real_type<commondx::complex<double>> {
            using value_type = double;
        };
        template<>
        struct real_type<cuComplex> {
            using value_type = float;
        };
        template<>
        struct real_type<cuDoubleComplex> {
            using value_type = double;
        };

        template<class T>
        using real_type_t = typename real_type<T>::value_type;

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UTIL_CUH
