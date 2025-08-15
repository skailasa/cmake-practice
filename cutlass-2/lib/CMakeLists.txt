add_library(KernelsDx STATIC kernels_dx.cu)
add_library(KernelsCutlass STATIC kernels_cutlass.cu)

# Point PATHS to the directory that contains lib/cmake/mathdx/mathdxConfig.cmake
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG
              PATHS "${PROJECT_SOURCE_DIR}/external/mathdx/25.06")
target_link_libraries(KernelsDx PRIVATE mathdx::cublasdx)

# Your CUTLASS-4.x (header-only) usage as an interface target
add_library(cutlass4 INTERFACE IMPORTED)
set_property(TARGET cutlass4 PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/external/cutlass/include")


# Usage requirements: link the right backend privately
target_link_libraries(KernelsCutlass PRIVATE cutlass4)

# export headers
target_include_directories(KernelsDx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(KernelsCutlass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)