/// Project that uses both the latest version of cutlass, and can build targets relying on the version shipped with cublasDX
#pragma once
#include <iostream>
#include <cutlass/cutlass.h>
#include <cute/layout.hpp>
#include <cutlass/version.h>

using namespace cute;

void lib_func() {

    auto shape = make_shape(_2{}, _2{});

    printf("cutlass\n");
}