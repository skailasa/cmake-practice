# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was nvcompdx-config.cmake.in                            ########

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

if(NOT TARGET nvcompdx::nvcompdx)
    # Finds CUTLASS/CuTe and sets nvcompdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
    #
    # CUTLASS root directory is found by checking following variables in this order:
    # 1. Root directory of NvidiaCutlass package
    # 2. nvcompdx_CUTLASS_ROOT
    # 3. ENV{nvcompdx_CUTLASS_ROOT}
    # 4. nvcompdx_CUTLASS_ROOT
    set(nvcompdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(nvcompdx_CUTLASS_MIN_VERSION 3.9.0)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${nvcompdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${nvcompdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(nvcompdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(nvcompdx_cutlass_INCLUDE_DIR "${nvcompdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "nvcompdx: Found CUTLASS (NvidiaCutlass) dependency: ${nvcompdx_NvidiaCutlass_include_dir}")
        endif()
        set(nvcompdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED nvcompdx_CUTLASS_ROOT)
        get_filename_component(nvcompdx_CUTLASS_ROOT_ABSOLUTE ${nvcompdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(nvcompdx_cutlass_INCLUDE_DIR  "${nvcompdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "nvcompdx: Found CUTLASS dependency via nvcompdx_CUTLASS_ROOT: ${nvcompdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(nvcompdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{nvcompdx_CUTLASS_ROOT})
        get_filename_component(nvcompdx_CUTLASS_ROOT_ABSOLUTE $ENV{nvcompdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(nvcompdx_cutlass_INCLUDE_DIR "${nvcompdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "nvcompdx: Found CUTLASS dependency via ENV{nvcompdx_CUTLASS_ROOT}: ${nvcompdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(nvcompdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    endif()
    if(NOT ${nvcompdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via nvcompdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # Find commondx
    set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${nvcompdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(nvcompdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET nvcompdx_commondx_include_dirs 0 nvcompdx_commondx_include_dir)
        set_and_check(nvcompdx_commondx_INCLUDE_DIR "${nvcompdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "nvcompdx: Found commondx dependency")
        endif()
    endif()

    if(${nvcompdx_DEPENDENCY_COMMONDX_RESOLVED} AND ${nvcompdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(nvcompdx_VERSION "0.1.0")
        # build: 26
        include("${CMAKE_CURRENT_LIST_DIR}/nvcompdx-targets.cmake")

        # Wrapper for the fatbin library
        if(NOT TARGET nvcompdx_fatbin AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8))
            add_library(nvcompdx_fatbin INTERFACE)
            add_library(nvcompdx::nvcompdx_fatbin ALIAS nvcompdx_fatbin)
            set_and_check(nvcompdx_FATBIN "${PACKAGE_PREFIX_DIR}/lib/libnvcompdx.fatbin")
            target_link_options(nvcompdx_fatbin INTERFACE $<DEVICE_LINK:${nvcompdx_FATBIN}>)
        endif()

        # Resolve dependencies:
        # 1) CUTLASS
        if(${NvidiaCutlass_FOUND})
            target_link_libraries(nvcompdx::nvcompdx INTERFACE nvidia::cutlass::cutlass)
            if(TARGET nvcompdx_fatbin)
                target_link_libraries(nvcompdx_fatbin INTERFACE nvidia::cutlass::cutlass)
            endif()
        elseif(DEFINED nvcompdx_CUTLASS_ROOT)
            target_include_directories(nvcompdx::nvcompdx INTERFACE ${nvcompdx_cutlass_INCLUDE_DIR})
            if(TARGET nvcompdx_fatbin)
                target_include_directories(nvcompdx_fatbin INTERFACE ${nvcompdx_cutlass_INCLUDE_DIR})
            endif()
        elseif(DEFINED ENV{nvcompdx_CUTLASS_ROOT})
            target_include_directories(nvcompdx::nvcompdx INTERFACE ${nvcompdx_cutlass_INCLUDE_DIR})
            if(TARGET nvcompdx_fatbin)
                target_include_directories(nvcompdx_fatbin INTERFACE ${nvcompdx_cutlass_INCLUDE_DIR})
            endif()
        endif()

        # 2) commondx
        target_link_libraries(nvcompdx::nvcompdx INTERFACE commondx::commondx)
        if(TARGET nvcompdx_fatbin)
            target_link_libraries(nvcompdx_fatbin INTERFACE commondx::commondx)
        endif()

        set_and_check(nvcompdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(nvcompdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(nvcompdx_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib")
        set(nvcompdx_LIBRARIES nvcompdx::nvcompdx)
        check_required_components(nvcompdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found nvcompdx: (Version: 0.1.0, Include dirs: ${nvcompdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
