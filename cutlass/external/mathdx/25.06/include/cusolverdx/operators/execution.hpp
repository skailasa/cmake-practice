#ifndef CUSOLVERDX_OPERATORS_EXECUTION_HPP
#define CUSOLVERDX_OPERATORS_EXECUTION_HPP

#include "commondx/operators/execution_operators.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    // Import Execution operators from commonDx
    struct Block: public commondx::Block {
    };

} // namespace cusolverdx

namespace commondx::detail {
    // Block specializations
    template<>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::block, cusolverdx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Block> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::block;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_EXECUTION_HPP
