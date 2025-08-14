# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cublasdx-config.cmake.in                            ########

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

if(NOT TARGET cublasdx::cublasdx)
    # Finds CUTLASS/CuTe and sets cublasdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
    #
    # CUTLASS root directory is found by checking following variables in this order:
    # 1. Root directory of NvidiaCutlass package
    # 2. cublasdx_CUTLASS_ROOT
    # 3. ENV{cublasdx_CUTLASS_ROOT}
    # 4. cublasdx_CUTLASS_ROOT
    set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(cublasdx_CUTLASS_MIN_VERSION 3.9.0)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${cublasdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${cublasdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(cublasdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR "${cublasdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS (NvidiaCutlass) dependency: ${cublasdx_NvidiaCutlass_include_dir}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED cublasdx_CUTLASS_ROOT)
        get_filename_component(cublasdx_CUTLASS_ROOT_ABSOLUTE ${cublasdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR  "${cublasdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS dependency via cublasdx_CUTLASS_ROOT: ${cublasdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{cublasdx_CUTLASS_ROOT})
        get_filename_component(cublasdx_CUTLASS_ROOT_ABSOLUTE $ENV{cublasdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR "${cublasdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS dependency via ENV{cublasdx_CUTLASS_ROOT}: ${cublasdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    endif()
    if(NOT ${cublasdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via cublasdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # Find commondx
    set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${cublasdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(cublasdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET cublasdx_commondx_include_dirs 0 cublasdx_commondx_include_dir)
        set_and_check(cublasdx_commondx_INCLUDE_DIR "${cublasdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found commondx dependency")
        endif()
    endif()

    if(${cublasdx_DEPENDENCY_COMMONDX_RESOLVED} AND ${cublasdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(cublasdx_VERSION "0.4.0")
        # build: 26
        include("${CMAKE_CURRENT_LIST_DIR}/cublasdx-targets.cmake")

        # Resolve dependencies:
        # 1) CUTLASS
        if(${NvidiaCutlass_FOUND})
            target_link_libraries(cublasdx::cublasdx INTERFACE nvidia::cutlass::cutlass)
        elseif(DEFINED cublasdx_CUTLASS_ROOT)
            target_include_directories(cublasdx::cublasdx INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
        elseif(DEFINED ENV{cublasdx_CUTLASS_ROOT})
            target_include_directories(cublasdx::cublasdx INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
        endif()
        # 2) commondx
        target_link_libraries(cublasdx::cublasdx INTERFACE commondx::commondx)

        set_and_check(cublasdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cublasdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set(cublasdx_LIBRARIES cublasdx::cublasdx)
        check_required_components(cublasdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found cublasdx: (Version: 0.4.0, Include dirs: ${cublasdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
