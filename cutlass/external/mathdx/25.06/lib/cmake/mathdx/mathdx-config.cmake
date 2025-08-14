# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mathdx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET mathdx::mathdx)
    # CUTLASS
    set_and_check(mathdx_included_CUTLASS_ROOT ${PACKAGE_PREFIX_DIR}/external/cutlass)
    set_and_check(mathdx_included_CUTLASS_INCLUDE_DIR ${PACKAGE_PREFIX_DIR}/external/cutlass/include)

    set(mathdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    # Finds CUTLASS/CuTe and sets mathdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
    #
    # CUTLASS root directory is found by checking following variables in this order:
    # 1. find_package(NvidiaCutlass) (NvidiaCutlasss_ROOT)
    # 2. mathdx_CUTLASS_ROOT
    # 3. ENV{mathdx_CUTLASS_ROOT}
    # 4. find_package(NvidiaCutlass PATHS ${mathdx_included_CUTLASS_ROOT})
    set(mathdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(mathdx_CUTLASS_MIN_VERSION 3.9.0)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${mathdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${mathdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(mathdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS (NvidiaCutlass) dependency: ${mathdx_NvidiaCutlass_include_dir}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED mathdx_CUTLASS_ROOT)
        get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE ${mathdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(mathdx_cutlass_INCLUDE_DIR  "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS dependency via mathdx_CUTLASS_ROOT: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{mathdx_CUTLASS_ROOT})
        get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE $ENV{mathdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS dependency via ENV{mathdx_CUTLASS_ROOT}: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    else()
        find_package(NvidiaCutlass QUIET PATHS ${mathdx_included_CUTLASS_ROOT})
        if(${NvidiaCutlass_FOUND})
            if(${NvidiaCutlass_VERSION} VERSION_LESS ${mathdx_CUTLASS_MIN_VERSION})
                message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${mathdx_CUTLASS_MIN_VERSION}")
            endif()
            get_property(mathdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_NvidiaCutlass_include_dir}")
            if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                message(STATUS "mathdx: Found CUTLASS (NvidiaCutlass) dependency: ${mathdx_NvidiaCutlass_include_dir}")
            endif()
            set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${mathdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via mathdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # commondx
    find_package(commondx
        REQUIRED
        QUIET
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/commondx/"
        NO_DEFAULT_PATH
    )

    set(mathdx_VERSION "25.06.0")
    # mathDx include directories
    set_and_check(mathdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
    set_and_check(mathdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
    # build: 26
    include("${CMAKE_CURRENT_LIST_DIR}/mathdx-targets.cmake")
endif()

list(TRANSFORM mathdx_FIND_COMPONENTS TOLOWER)
# Populate components list when blank or ALL is provided
if(NOT mathdx_FIND_COMPONENTS OR "ALL" IN_LIST mathdx_FIND_COMPONENTS)
    if(NOT mathdx_FIND_COMPONENTS AND mathdx_FIND_REQUIRED)
        set(mathdx_FIND_REQUIRED_ALL TRUE)
    endif()
    set(mathdx_ALL_COMPONENTS TRUE)
    set(mathdx_FIND_COMPONENTS "")

    foreach(comp IN ITEMS cufftdx cublasdx curanddx cusolverdx nvcompdx)
        list(APPEND mathdx_FIND_COMPONENTS ${comp})
        if(mathdx_FIND_REQUIRED_ALL)
            set(mathdx_FIND_REQUIRED_${comp} TRUE)
        endif()
    endforeach()
endif()

# cuFFTDx
if("cufftdx" IN_LIST mathdx_FIND_COMPONENTS)
    set(cufftdx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_cufftdx)
        set(cufftdx_REQUIRED "REQUIRED")
    endif()
    set(cufftdx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(cufftdx_QUIETLY "QUIET")
    endif()
    find_package(cufftdx
        ${cufftdx_REQUIRED}
        ${cufftdx_QUIETLY}
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/cufftdx/"
        NO_DEFAULT_PATH
    )
    if(NOT TARGET mathdx::cufftdx)
        if(cufftdx_FOUND)
            set(mathdx_cufftdx_FOUND TRUE)
            add_library(mathdx::cufftdx ALIAS cufftdx::cufftdx)
            if(NOT TARGET mathdx::cufftdx_separate_twiddles_lut)
                add_library(mathdx::cufftdx_separate_twiddles_lut ALIAS cufftdx_separate_twiddles_lut)
            endif()
            set(cufftdx_LIBRARIES mathdx::cufftdx)
            if(NOT mathdx_FIND_QUIETLY)
                message(STATUS "mathDx: cuFFTDx found: ${cufftdx_INCLUDE_DIRS}")
            endif()
        else()
            set(mathdx_cufftdx_FOUND FALSE)
        endif()
    endif() # NOT TARGET mathdx::<lib>
endif() # IN_LIST mathdx_FIND_COMPONENTS

# cuBLASDx
if("cublasdx" IN_LIST mathdx_FIND_COMPONENTS)
    set(cublasdx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_cublasdx)
        set(cublasdx_REQUIRED "REQUIRED")
    endif()
    set(cublasdx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(cublasdx_QUIETLY "QUIET")
    endif()
    find_package(cublasdx
        ${cublasdx_REQUIRED}
        ${cublasdx_QUIETLY}
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/cublasdx/"
        NO_DEFAULT_PATH
    )
    if(NOT TARGET mathdx::cublasdx)
        if(cublasdx_FOUND)
            set(mathdx_cublasdx_FOUND TRUE)
            add_library(mathdx::cublasdx ALIAS cublasdx::cublasdx)
            set(cublasdx_LIBRARIES mathdx::cublasdx)
            if(NOT mathdx_FIND_QUIETLY)
                message(STATUS "mathDx: cublasdx found: ${cublasdx_INCLUDE_DIRS}")
            endif()
        else()
            set(mathdx_cublasdx_FOUND FALSE)
        endif()
    endif() # NOT TARGET mathdx::<lib>
endif() # IN_LIST mathdx_FIND_COMPONENTS

# cuRANDDx
if("curanddx" IN_LIST mathdx_FIND_COMPONENTS)
    set(curanddx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_curanddx)
        set(curanddx_REQUIRED "REQUIRED")
    endif()
    set(curanddx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(curanddx_QUIETLY "QUIET")
    endif()
    find_package(curanddx
        ${curanddx_REQUIRED}
        ${curanddx_QUIETLY}
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/curanddx/"
        NO_DEFAULT_PATH
    )
    if(NOT TARGET mathdx::curanddx)
        if(curanddx_FOUND)
            set(mathdx_curanddx_FOUND TRUE)
            add_library(mathdx::curanddx ALIAS curanddx::curanddx)
            set(curanddx_LIBRARIES mathdx::curanddx)
            if(NOT mathdx_FIND_QUIETLY)
                message(STATUS "mathDx: curanddx found: ${curanddx_INCLUDE_DIRS}")
            endif()
        else()
            set(mathdx_curanddx_FOUND FALSE)
        endif()
    endif() # NOT TARGET mathdx::<lib>
endif() # IN_LIST mathdx_FIND_COMPONENTS

# cuSolverDx
if("cusolverdx" IN_LIST mathdx_FIND_COMPONENTS)
    set(cusolverdx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_cusolverdx)
        set(cusolverdx_REQUIRED "REQUIRED")
    endif()
    set(cusolverdx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(cusolverdx_QUIETLY "QUIET")
    endif()
    find_package(cusolverdx
        ${cusolverdx_REQUIRED}
        ${cusolverdx_QUIETLY}
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/cusolverdx/"
        NO_DEFAULT_PATH
    )
    if(NOT TARGET mathdx::cusolverdx)
        if(cusolverdx_FOUND)
            set(mathdx_cusolverdx_FOUND TRUE)
            add_library(mathdx::cusolverdx ALIAS cusolverdx::cusolverdx)
            if(NOT TARGET mathdx::cusolverdx_fatbin AND TARGET cusolverdx_fatbin)
                add_library(mathdx::cusolverdx_fatbin ALIAS cusolverdx_fatbin)
            endif()
            set(cusolverdx_LIBRARIES mathdx::cusolverdx)
            if(NOT mathdx_FIND_QUIETLY)
                message(STATUS "mathDx: cusolverdx found: ${cusolverdx_INCLUDE_DIRS}")
            endif()
        else()
            set(mathdx_cusolverdx_FOUND FALSE)
        endif()
    endif() # NOT TARGET mathdx::<lib>
endif() # IN_LIST mathdx_FIND_COMPONENTS

# nvCOMPDx
if("nvcompdx" IN_LIST mathdx_FIND_COMPONENTS)
    set(nvcompdx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_nvcompdx)
        set(nvcompdx_REQUIRED "REQUIRED")
    endif()
    set(nvcompdx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(nvcompdx_QUIETLY "QUIET")
    endif()
    find_package(nvcompdx
        ${nvcompdx_REQUIRED}
        ${nvcompdx_QUIETLY}
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/nvcompdx/"
        NO_DEFAULT_PATH
    )
    if(NOT TARGET mathdx::nvcompdx)
        if(nvcompdx_FOUND)
            set(mathdx_nvcompdx_FOUND TRUE)
            add_library(mathdx::nvcompdx ALIAS nvcompdx::nvcompdx)
            if(NOT TARGET mathdx::nvcompdx_fatbin AND TARGET nvcompdx_fatbin)
                add_library(mathdx::nvcompdx_fatbin ALIAS nvcompdx_fatbin)
            endif()
            set(nvcompdx_LIBRARIES mathdx::nvcompdx)
            if(NOT mathdx_FIND_QUIETLY)
                message(STATUS "mathDx: nvcompdx found: ${nvcompdx_INCLUDE_DIRS}")
            endif()
        else()
            set(mathdx_nvcompdx_FOUND FALSE)
        endif()
    endif() # NOT TARGET mathdx::<lib>
endif() # IN_LIST mathdx_FIND_COMPONENTS

# Check all components
check_required_components(mathdx)
