#ifndef CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP
#define CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP

#include "cusolverdx/detail/solver_execution.hpp"

namespace cusolverdx {
    namespace detail {
        template<class... Operators>
        struct make_description {
        private:
            using operator_wrapper_type                  = solver_operator_wrapper<Operators...>;
            static constexpr bool has_block_operator     = has_operator<operator_type::block, operator_wrapper_type>::value;
            static constexpr bool has_execution_operator = has_block_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called blas_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before Solver description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using execution_type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_complete_solver<operator_wrapper_type>::value,
                                                             block_execution<Operators...>,
                                                             solver_execution<Operators...>>::type;
#else
            using execution_type = block_execution<Operators...>;
#endif
            using description_type = solver_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                               detail::make_description_t<Operator1, Operator2>> {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::solver_description<Operators1...>&,
                                                       const Operator2&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::is_operator_expression<Operator2>::value,
                                               detail::make_description_t<Operators1..., Operator2>> {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::solver_description<Operators2...>&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::is_operator_expression<Operator1>::value,
                                               detail::make_description_t<Operator1, Operators2...>> {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::solver_description<Operators1...>&,
                                                       const detail::solver_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cusolverdx
#endif // CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP
