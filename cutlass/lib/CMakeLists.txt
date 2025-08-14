add_library(Kernels STATIC kernels.cu)

# add option to compile either the cublasdx kernls OR the cutlass kernels
option(USE_DX "Option to toggle CuBLASdx kernels" OFF)

if (USE_DX)
  # Propagate whetehr or not cublas DX is being used as a precompiled definition to source files
  # in kernels
  target_compile_definitions(Kernels PRIVATE "USE_DX")
endif()

if (USE_DX)
  add_library(CublasDx STATIC cublas_dx.cu) # static library, need to specify sources
else()
  add_library(Cutlass INTERFACE)
endif()

if (USE_DX)
  target_link_libraries(Kernels PRIVATE CublasDx)
endif()

# Note don't need to explicitly link header only libraries, as no object files
# that's why only need to link cublasdx library as currently specified