# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was curanddx-config.cmake.in                            ########

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

if(NOT TARGET curanddx::curanddx)
    # cuRANDDx doesn't include "commondx/complex_types.hpp" which means it doesn't have
    # the part of commonDx that depend on CUTLASS. No direct dependency neither.
    set(curanddx_DEPENDENCY_CUTLASS_DISABLED TRUE)
    if(${mathdx_curanddx_DISABLE_CUTLASS})
        set(curanddx_DEPENDENCY_CUTLASS_DISABLED TRUE)
    endif()
    if(${curanddx_DISABLE_CUTLASS})
        set(curanddx_DEPENDENCY_CUTLASS_DISABLED TRUE)
    endif()

    set(curanddx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    if(NOT ${curanddx_DEPENDENCY_CUTLASS_DISABLED})
        # Finds CUTLASS/CuTe and sets curanddx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
        #
        # CUTLASS root directory is found by checking following variables in this order:
        # 1. Root directory of NvidiaCutlass package
        # 2. curanddx_CUTLASS_ROOT
        # 3. ENV{curanddx_CUTLASS_ROOT}
        set(curanddx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
        set(curanddx_CUTLASS_MIN_VERSION 3.6.0)
        find_package(NvidiaCutlass QUIET)
        if(${NvidiaCutlass_FOUND})
            if(${NvidiaCutlass_VERSION} VERSION_LESS ${curanddx_CUTLASS_MIN_VERSION})
                message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${curanddx_CUTLASS_MIN_VERSION}")
            endif()
            get_property(curanddx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            set_and_check(curanddx_cutlass_INCLUDE_DIR "${curanddx_NvidiaCutlass_include_dir}")
            if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                message(STATUS "curanddx: Found CUTLASS (NvidiaCutlass) dependency: ${curanddx_NvidiaCutlass_include_dir}")
            endif()
            set(curanddx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        elseif(DEFINED curanddx_CUTLASS_ROOT)
            get_filename_component(curanddx_CUTLASS_ROOT_ABSOLUTE ${curanddx_CUTLASS_ROOT} ABSOLUTE)
            set_and_check(curanddx_cutlass_INCLUDE_DIR  "${curanddx_CUTLASS_ROOT_ABSOLUTE}/include")
            if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                message(STATUS "curanddx: Found CUTLASS dependency via curanddx_CUTLASS_ROOT: ${curanddx_CUTLASS_ROOT_ABSOLUTE}")
            endif()
            set(curanddx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        elseif(DEFINED ENV{curanddx_CUTLASS_ROOT})
            get_filename_component(curanddx_CUTLASS_ROOT_ABSOLUTE $ENV{curanddx_CUTLASS_ROOT} ABSOLUTE)
            set_and_check(curanddx_cutlass_INCLUDE_DIR "${curanddx_CUTLASS_ROOT_ABSOLUTE}/include")
            if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                message(STATUS "curanddx: Found CUTLASS dependency via ENV{curanddx_CUTLASS_ROOT}: ${curanddx_CUTLASS_ROOT_ABSOLUTE}")
            endif()
            set(curanddx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        endif()
        if(NOT ${curanddx_DEPENDENCY_CUTLASS_RESOLVED})
            set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
            if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
                message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                    "    Missing CUTLASS dependency.\n"
                                    "    You can set it via curanddx_CUTLASS_ROOT variable or by providing\n"
                                    "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
            endif()
        endif()
    else()
        set(curanddx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    endif()

    # Find commondx
    set(curanddx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(curanddx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(curanddx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${curanddx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(curanddx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET curanddx_commondx_include_dirs 0 curanddx_commondx_include_dir)
        set_and_check(curanddx_commondx_INCLUDE_DIR "${curanddx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "curanddx: Found commondx dependency")
        endif()
    endif()

    if(${curanddx_DEPENDENCY_COMMONDX_RESOLVED} AND ${curanddx_DEPENDENCY_CUTLASS_RESOLVED})
        set(curanddx_VERSION "0.2.0")
        # build: 26
        include("${CMAKE_CURRENT_LIST_DIR}/curanddx-targets.cmake")

        # Resolve dependencies:
        # 1) CUTLASS
        if(NOT ${curanddx_DEPENDENCY_CUTLASS_DISABLED})
            if(${NvidiaCutlass_FOUND})
                target_link_libraries(curanddx::curanddx INTERFACE nvidia::cutlass::cutlass)
            elseif(DEFINED curanddx_CUTLASS_ROOT)
                target_include_directories(curanddx::curanddx INTERFACE ${curanddx_cutlass_INCLUDE_DIR})
            elseif(DEFINED ENV{curanddx_CUTLASS_ROOT})
                target_include_directories(curanddx::curanddx INTERFACE ${curanddx_cutlass_INCLUDE_DIR})
            endif()
        else()
            target_compile_definitions(curanddx::curanddx INTERFACE CURANDDX_DISABLE_CUTLASS_DEPENDENCY)
        endif()
        # 2) commondx
        target_link_libraries(curanddx::curanddx INTERFACE commondx::commondx)

        set_and_check(curanddx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(curanddx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set(curanddx_LIBRARIES curanddx::curanddx)
        check_required_components(curanddx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found curanddx: (Version: 0.2.0, Include dirs: ${curanddx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
