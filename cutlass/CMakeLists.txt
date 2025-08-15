cmake_minimum_required(VERSION 3.15)


# Function to locate spack packages on system
function(find_spack_package name out_path)
  execute_process(COMMAND spack location -i ${name} OUTPUT_VARIABLE _path OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${out_path} "${_path}" PARENT_SCOPE)
endfunction()

project(CutlassCublasDx LANGUAGES CXX CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 89) # required for cuda projects
enable_language(CUDA) #  required for cuda prjects

find_spack_package(cuda CUDA_PATH)
if(CUDA_PATH)
    message(STATUS "Using Spack CUDA at ${CUDA_PATH}")
    set(CUDAToolkit_ROOT "${CUDA_PATH}")
elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    message(STATUS "Using system CUDA at /usr/local/cuda")
    set(CUDAToolkit_ROOT "/usr/local/cuda")
else()
    message(FATAL_ERROR "Could not find CUDA Toolkit via Spack or /usr/local/cuda")
endif()

# Add to search path for find_package
list(PREPEND CMAKE_PREFIX_PATH "${CUDAToolkit_ROOT}")

# Only set compiler if we found nvcc
if(EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
endif()

add_subdirectory(lib) # recurses through here first
add_executable(main main.cpp)

# Have two compile targets, one for cutlass and one that uses it from cublasdx
target_link_libraries(main PUBLIC Kernels)

target_include_directories(main PUBLIC "${PROJECT_SOURCE_DIR}/lib/")
