# Copyright 2019, 2023 Peter Dimov
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at http://boost.org/LICENSE_1_0.txt)

# This CMake configuration file provides support for find_package(Boost).

if(Boost_VERBOSE OR Boost_DEBUG)

  message(STATUS "Found Boost ${Boost_VERSION} at ${Boost_DIR}")

  # Output requested configuration (f.ex. "REQUIRED COMPONENTS filesystem")

  if(Boost_FIND_QUIETLY)
    set(_BOOST_CONFIG "${_BOOST_CONFIG} QUIET")
  endif()

  if(Boost_FIND_REQUIRED)
    set(_BOOST_CONFIG "${_BOOST_CONFIG} REQUIRED")
  endif()

  foreach(__boost_comp IN LISTS Boost_FIND_COMPONENTS)
    if(${Boost_FIND_REQUIRED_${__boost_comp}})
      list(APPEND _BOOST_COMPONENTS ${__boost_comp})
    else()
      list(APPEND _BOOST_OPTIONAL_COMPONENTS ${__boost_comp})
    endif()
  endforeach()

  if(_BOOST_COMPONENTS)
    set(_BOOST_CONFIG "${_BOOST_CONFIG} COMPONENTS ${_BOOST_COMPONENTS}")
  endif()

  if(_BOOST_OPTIONAL_COMPONENTS)
    set(_BOOST_CONFIG "${_BOOST_CONFIG} OPTIONAL_COMPONENTS ${_BOOST_OPTIONAL_COMPONENTS}")
  endif()

  if(_BOOST_CONFIG)
    message(STATUS "  Requested configuration:${_BOOST_CONFIG}")
  endif()

  unset(_BOOST_CONFIG)
  unset(_BOOST_COMPONENTS)
  unset(_BOOST_OPTIONAL_COMPONENTS)

endif()

macro(boostcfg_find_component comp required quiet)

  set(_BOOST_QUIET)
  if(Boost_FIND_QUIETLY OR ${quiet})
    set(_BOOST_QUIET QUIET)
  endif()

  set(_BOOST_REQUIRED)
  if(${required} AND Boost_FIND_REQUIRED)
    set(_BOOST_REQUIRED REQUIRED)
  endif()

  set(__boost_comp_nv "${comp}")

  get_filename_component(_BOOST_CMAKEDIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

  if(Boost_DEBUG)
    message(STATUS "BoostConfig: find_package(boost_${__boost_comp_nv} ${Boost_VERSION} EXACT CONFIG ${_BOOST_REQUIRED} ${_BOOST_QUIET} HINTS ${_BOOST_CMAKEDIR})")
  endif()

  find_package(boost_${__boost_comp_nv} ${Boost_VERSION} EXACT CONFIG ${_BOOST_REQUIRED} ${_BOOST_QUIET} HINTS ${_BOOST_CMAKEDIR})

  set(__boost_comp_found ${boost_${__boost_comp_nv}_FOUND})

  # FindPackageHandleStandardArgs expects <package>_<component>_FOUND
  set(Boost_${comp}_FOUND ${__boost_comp_found})

  # FindBoost sets Boost_<COMPONENT>_FOUND
  string(TOUPPER ${comp} _BOOST_COMP)
  set(Boost_${_BOOST_COMP}_FOUND ${__boost_comp_found})

  # FindBoost compatibility variables: Boost_LIBRARIES, Boost_<C>_LIBRARY
  if(__boost_comp_found)

    list(APPEND Boost_LIBRARIES Boost::${__boost_comp_nv})
    set(Boost_${_BOOST_COMP}_LIBRARY Boost::${__boost_comp_nv})

  endif()

  unset(_BOOST_REQUIRED)
  unset(_BOOST_QUIET)
  unset(_BOOST_CMAKEDIR)
  unset(__boost_comp_nv)
  unset(__boost_comp_found)
  unset(_BOOST_COMP)

endmacro()

# Find boost_headers

boostcfg_find_component(headers 1 0)

if(NOT boost_headers_FOUND)

  set(Boost_FOUND 0)
  set(Boost_NOT_FOUND_MESSAGE "A required dependency, boost_headers, has not been found.")

  return()

endif()

# Compatibility variables

set(Boost_MAJOR_VERSION ${Boost_VERSION_MAJOR})
set(Boost_MINOR_VERSION ${Boost_VERSION_MINOR})
set(Boost_SUBMINOR_VERSION ${Boost_VERSION_PATCH})

set(Boost_VERSION_STRING ${Boost_VERSION})
set(Boost_VERSION_MACRO ${Boost_VERSION_MAJOR}0${Boost_VERSION_MINOR}0${Boost_VERSION_PATCH})

get_target_property(Boost_INCLUDE_DIRS Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
set(Boost_LIBRARIES "")

# Save project's policies
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW) # if IN_LIST

# Find components

foreach(__boost_comp IN LISTS Boost_FIND_COMPONENTS)

  boostcfg_find_component(${__boost_comp} ${Boost_FIND_REQUIRED_${__boost_comp}} 0)

endforeach()

# Compatibility targets

if(NOT TARGET Boost::boost)

  add_library(Boost::boost INTERFACE IMPORTED)
  set_property(TARGET Boost::boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES Boost::headers)

  add_library(Boost::diagnostic_definitions INTERFACE IMPORTED)
  add_library(Boost::disable_autolinking INTERFACE IMPORTED)
  add_library(Boost::dynamic_linking INTERFACE IMPORTED)

  if(WIN32)

    set_property(TARGET Boost::diagnostic_definitions PROPERTY INTERFACE_COMPILE_DEFINITIONS "BOOST_LIB_DIAGNOSTIC")
    set_property(TARGET Boost::disable_autolinking PROPERTY INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")
    set_property(TARGET Boost::dynamic_linking PROPERTY INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_DYN_LINK")

  endif()

endif()

# Restore project's policies
cmake_policy(POP)
