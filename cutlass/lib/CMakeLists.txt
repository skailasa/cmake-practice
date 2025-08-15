add_library(Kernels STATIC kernels.cu)

# Toggle: build DX kernels or CUTLASS kernels
option(USE_DX "Use cuBLASDx kernels" OFF)

if (USE_DX)
  target_compile_definitions(Kernels PRIVATE USE_DX)

  add_library(CublasDx STATIC cublas_dx.cu)

  # Find MathDx (cuBLASDx). Point PATHS to the folder that has lib/cmake/mathdx/mathdxConfig.cmake
  find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG
               PATHS "${PROJECT_SOURCE_DIR}/external/mathdx/25.06")

  target_link_libraries(CublasDx PUBLIC mathdx::cublasdx)
  target_link_libraries(Kernels  PUBLIC CublasDx)

else()
  # CUTLASS is header-only: export its *include dir* via an INTERFACE target,
  # and link that target to Kernels so it inherits the include path.
  add_library(Cutlass INTERFACE)
  target_include_directories(Cutlass INTERFACE
    "${CMAKE_SOURCE_DIR}/external/cutlass/include"
  )

  target_link_libraries(Kernels PUBLIC Cutlass)
endif()

# Note don't need to explicitly link header only libraries, as no object files
# that's why only need to link cublasdx library as currently specified