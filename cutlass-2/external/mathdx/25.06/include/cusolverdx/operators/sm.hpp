#ifndef CUSOLVERDX_OPERATORS_SM_HPP
#define CUSOLVERDX_OPERATORS_SM_HPP

#include "commondx/operators/sm.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    // Import SM operator from commonDx
    template <unsigned int Architecture>
    using SM = commondx::SM<Architecture>;
} // namespace cusolverdx

namespace commondx::detail {
    // Thread specializations
    template<unsigned int Architecture>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::sm, cusolverdx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int Architecture>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::SM<Architecture>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::sm;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_SM_HPP
