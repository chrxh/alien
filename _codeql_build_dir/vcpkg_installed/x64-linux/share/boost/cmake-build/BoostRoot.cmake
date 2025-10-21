# Copyright 2019-2023 Peter Dimov
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

if(CMAKE_SOURCE_DIR STREQUAL Boost_SOURCE_DIR AND WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

    set(CMAKE_INSTALL_PREFIX "C:/Boost" CACHE PATH "Installation path prefix, prepended to installation directories" FORCE)

endif()

include(BoostMessage)
include(BoostInstall)

#

boost_message(VERBOSE "Boost: using CMake ${CMAKE_VERSION}")

#

set(__boost_incompatible_libraries "")

# Define cache variables if root project

if(1)

  # --with-<library>
  set(BOOST_INCLUDE_LIBRARIES "" CACHE STRING
    "List of libraries to build (default: all but excluded and incompatible)")

  # --without-<library>
  set(BOOST_EXCLUDE_LIBRARIES "" CACHE STRING
    "List of libraries to exclude from build")

  set(BOOST_INCOMPATIBLE_LIBRARIES
    "${__boost_incompatible_libraries}"
    CACHE STRING
    "List of libraries with incompatible CMakeLists.txt files")

  option(BOOST_ENABLE_MPI
    "Build and enable installation of Boost.MPI and its dependents (requires MPI, CMake 3.9)")

  option(BOOST_ENABLE_PYTHON
    "Build and enable installation of Boost.Python and its dependents (requires Python, CMake 3.14)")

  # --layout, --libdir, --cmakedir, --includedir in BoostInstall

  # runtime-link=static|shared

  set(BOOST_RUNTIME_LINK shared CACHE STRING "Runtime library selection for the MS ABI (shared or static)")
  set_property(CACHE BOOST_RUNTIME_LINK PROPERTY STRINGS shared static)

  option(BUILD_TESTING "Build the tests." OFF)
  include(CTest)

  if(NOT TARGET tests)
    add_custom_target(tests)
  endif()

  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -C $<CONFIG>)
  add_dependencies(check tests)

  if(NOT TARGET tests-quick)
    add_custom_target(tests-quick)
  endif()

  add_custom_target(check-quick COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -C $<CONFIG> -R quick)
  add_dependencies(check-quick tests-quick)

  # link=static|shared
  option(BUILD_SHARED_LIBS "Build shared libraries")

  # --stagedir
  set(BOOST_STAGEDIR "${CMAKE_CURRENT_BINARY_DIR}/stage" CACHE STRING "Build output directory")

  if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BOOST_STAGEDIR}/bin")
  endif()

  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${BOOST_STAGEDIR}/lib")
  endif()

  if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${BOOST_STAGEDIR}/lib")
  endif()

  # Set default visibility to hidden to match b2

  set(CMAKE_C_VISIBILITY_PRESET hidden CACHE STRING "Symbol visibility for C")
  set_property(CACHE CMAKE_C_VISIBILITY_PRESET PROPERTY STRINGS default hidden protected internal)

  set(CMAKE_CXX_VISIBILITY_PRESET hidden CACHE STRING "Symbol visibility for C++")
  set_property(CACHE CMAKE_CXX_VISIBILITY_PRESET PROPERTY STRINGS default hidden protected internal)

  option(CMAKE_VISIBILITY_INLINES_HIDDEN "Inline function have hidden visibility" ON)

  # Enable IDE folders for Visual Studio

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

else()

  # add_subdirectory use

  if(NOT DEFINED BOOST_INCOMPATIBLE_LIBRARIES)
    set(BOOST_INCOMPATIBLE_LIBRARIES ${__boost_incompatible_libraries})
  endif()

  if(NOT DEFINED BOOST_ENABLE_MPI)
    set(BOOST_ENABLE_MPI OFF)
  endif()

  if(NOT DEFINED BOOST_ENABLE_PYTHON)
    set(BOOST_ENABLE_PYTHON OFF)
  endif()

  if(NOT DEFINED BOOST_SKIP_INSTALL_RULES)
    set(BOOST_SKIP_INSTALL_RULES ON)
  endif()

  set(BUILD_TESTING OFF)

endif()

if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)

  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  if(NOT BOOST_RUNTIME_LINK STREQUAL "static")
    string(APPEND CMAKE_MSVC_RUNTIME_LIBRARY "DLL")
  endif()

endif()

# Output configuration status lines

set(_msg "")

if(NOT CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE)
  string(APPEND _msg "${CMAKE_BUILD_TYPE} build, ")
endif()

if(BUILD_SHARED_LIBS)
  string(APPEND _msg "shared libraries, ")
else()
  string(APPEND _msg "static libraries, ")
endif()

if(MSVC)
  if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    string(APPEND _msg "static runtime, ")
  elseif(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    string(APPEND _msg "shared runtime, ")
  endif()
endif()

string(APPEND _msg "MPI ${BOOST_ENABLE_MPI}, Python ${BOOST_ENABLE_PYTHON}, testing ${BUILD_TESTING}")

message(STATUS "Boost: ${_msg}")

unset(_msg)

if(BOOST_INCLUDE_LIBRARIES)
  message(STATUS "Boost: libraries included: ${BOOST_INCLUDE_LIBRARIES}")
endif()

if(BOOST_EXCLUDE_LIBRARIES)
  message(STATUS "Boost: libraries excluded: ${BOOST_EXCLUDE_LIBRARIES}")
endif()

# Define functions

function(__boost_auto_install __boost_lib)
  if(NOT CMAKE_VERSION VERSION_LESS 3.13)

    string(MAKE_C_IDENTIFIER "${__boost_lib}" __boost_lib_target)

    if(TARGET "Boost::${__boost_lib_target}" AND TARGET "boost_${__boost_lib_target}")

      get_target_property(__boost_lib_incdir "boost_${__boost_lib_target}" INTERFACE_INCLUDE_DIRECTORIES)

      set(incdir "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${__boost_lib}/include")

      if("${__boost_lib_incdir}" STREQUAL "${incdir}" OR "${__boost_lib_incdir}" STREQUAL "$<BUILD_INTERFACE:${incdir}>")

        boost_message(DEBUG "Enabling installation for '${__boost_lib}'")
        boost_install(TARGETS "boost_${__boost_lib_target}" VERSION "${BOOST_SUPERPROJECT_VERSION}" HEADER_DIRECTORY "${incdir}")

      else()
        boost_message(DEBUG "Not enabling installation for '${__boost_lib}'; interface include directory '${__boost_lib_incdir}' does not equal '${incdir}' or '$<BUILD_INTERFACE:${incdir}>'")
      endif()

    else()
      boost_message(DEBUG "Not enabling installation for '${__boost_lib}'; targets 'Boost::${__boost_lib_target}' and 'boost_${__boost_lib_target}' weren't found")
    endif()

  endif()
endfunction()

function(__boost_scan_dependencies lib var)

  set(result "")
  set(required_components "")
  set(optional_components "")

  if(EXISTS "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${lib}/CMakeLists.txt")

    file(STRINGS "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${lib}/CMakeLists.txt" data)

    foreach(line IN LISTS data)

      if(line MATCHES "^[ ]*Boost::([A-Za-z0-9_]+)[ ]*$")

        list(APPEND required_components ${CMAKE_MATCH_1})
        string(REGEX REPLACE "^numeric_" "numeric/" dep ${CMAKE_MATCH_1})
        list(APPEND result ${dep})

      elseif(line MATCHES "^[ ]*$<TARGET_NAME_IF_EXISTS:Boost::([A-Za-z0-9_]+)>[ ]*$")

        list(APPEND optional_components ${CMAKE_MATCH_1})
        string(REGEX REPLACE "^numeric_" "numeric/" dep ${CMAKE_MATCH_1})
        list(APPEND result ${dep})

      endif()

    endforeach()

  endif()

  list(REMOVE_DUPLICATES required_components)
  list(REMOVE_DUPLICATES optional_components)
  list(REMOVE_ITEM required_components boost ${lib}) # due to property_tree and python
  if(required_components OR optional_components)
    find_package(Boost COMPONENTS ${required_components} OPTIONAL_COMPONENTS ${optional_components} REQUIRED CONFIG)
  endif()
  set(${var} ${result} PARENT_SCOPE)

endfunction()

#

file(GLOB __boost_libraries RELATIVE "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs" "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/*/CMakeLists.txt" "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/numeric/*/CMakeLists.txt")

# Check for mistakes in BOOST_INCLUDE_LIBRARIES

foreach(__boost_included_lib IN LISTS BOOST_INCLUDE_LIBRARIES)

  if(NOT "${__boost_included_lib}/CMakeLists.txt" IN_LIST __boost_libraries)

    message(WARNING "Library '${__boost_included_lib}' given in BOOST_INCLUDE_LIBRARIES has not been found.")

  endif()

endforeach()

# Scan for dependencies

set(__boost_include_libraries ${BOOST_INCLUDE_LIBRARIES})

if(__boost_include_libraries)
  list(REMOVE_DUPLICATES __boost_include_libraries)
endif()

set(__boost_libs_to_scan ${__boost_include_libraries})

while(__boost_libs_to_scan)

  boost_message(DEBUG "Scanning dependencies: ${__boost_libs_to_scan}")

  set(__boost_dependencies "")

  foreach(__boost_lib IN LISTS __boost_libs_to_scan)

    __boost_scan_dependencies(${__boost_lib} __boost_deps)
    list(APPEND __boost_dependencies ${__boost_deps})

  endforeach()

  list(REMOVE_DUPLICATES __boost_dependencies)


  if(__boost_libs_to_scan)
    list(REMOVE_ITEM __boost_libs_to_scan ${__boost_include_libraries})
    list(REMOVE_ITEM __boost_libs_to_scan ${__boost_lib})
  endif()

  list(APPEND __boost_include_libraries ${__boost_libs_to_scan})

endwhile()

# Installing targets created in other directories requires CMake 3.13
if(CMAKE_VERSION VERSION_LESS 3.13)

  boost_message(STATUS "Boost installation support requires CMake 3.13 (have ${CMAKE_VERSION})")

endif()

set(__boost_mpi_libs mpi graph_parallel property_map_parallel)
set(__boost_python_libs python parameter_python)

foreach(__boost_lib_cml IN LISTS __boost_libraries)

  get_filename_component(__boost_lib "${__boost_lib_cml}" DIRECTORY)

  if(__boost_lib IN_LIST BOOST_INCOMPATIBLE_LIBRARIES)

    boost_message(DEBUG "Skipping incompatible Boost library ${__boost_lib}")

  elseif(__boost_lib IN_LIST BOOST_EXCLUDE_LIBRARIES)

    boost_message(DEBUG "Skipping excluded Boost library ${__boost_lib}")

  elseif(NOT BOOST_ENABLE_MPI AND __boost_lib IN_LIST __boost_mpi_libs)

    boost_message(DEBUG "Skipping Boost library ${__boost_lib}, BOOST_ENABLE_MPI is OFF")

  elseif(NOT BOOST_ENABLE_PYTHON AND __boost_lib IN_LIST __boost_python_libs)

    boost_message(DEBUG "Skipping Boost library ${__boost_lib}, BOOST_ENABLE_PYTHON is OFF")

  elseif(NOT BOOST_INCLUDE_LIBRARIES OR __boost_lib IN_LIST BOOST_INCLUDE_LIBRARIES)

    boost_message(VERBOSE "Adding Boost library ${__boost_lib}")
    add_subdirectory(libs/${__boost_lib})

    __boost_auto_install(${__boost_lib})

  elseif(__boost_lib IN_LIST __boost_include_libraries OR __boost_lib STREQUAL "headers")

    # Disable tests for dependencies

    set(__boost_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF) # hide cache variable

    set(__boost_cmake_folder ${CMAKE_FOLDER})

    if("${CMAKE_FOLDER}" STREQUAL "")
      set(CMAKE_FOLDER "Dependencies")
    endif()

    boost_message(VERBOSE "Adding Boost dependency ${__boost_lib}")
    add_subdirectory(libs/${__boost_lib})

    __boost_auto_install(${__boost_lib})

    set(BUILD_TESTING ${__boost_build_testing})
    set(CMAKE_FOLDER ${__boost_cmake_folder})

  elseif(BUILD_TESTING)

    # Disable tests and installation for libraries neither included nor dependencies

    set(__boost_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF) # hide cache variable

    set(__boost_skip_install ${BOOST_SKIP_INSTALL_RULES})
    set(BOOST_SKIP_INSTALL_RULES ON)

    set(__boost_cmake_folder ${CMAKE_FOLDER})

    if("${CMAKE_FOLDER}" STREQUAL "")
      set(CMAKE_FOLDER "Dependencies")
    endif()

    boost_message(DEBUG "Adding Boost library ${__boost_lib} with EXCLUDE_FROM_ALL")
    add_subdirectory(libs/${__boost_lib} EXCLUDE_FROM_ALL)

    set(BUILD_TESTING ${__boost_build_testing})
    set(BOOST_SKIP_INSTALL_RULES ${__boost_skip_install})
    set(CMAKE_FOLDER ${__boost_cmake_folder})

  endif()

endforeach()

# Install BoostConfig.cmake

if(BOOST_SKIP_INSTALL_RULES)

  boost_message(DEBUG "Boost: not installing BoostConfig.cmake due to BOOST_SKIP_INSTALL_RULES=${BOOST_SKIP_INSTALL_RULES}")
  return()

endif()

if(CMAKE_SKIP_INSTALL_RULES)

  boost_message(DEBUG "Boost: not installing BoostConfig.cmake due to CMAKE_SKIP_INSTALL_RULES=${CMAKE_SKIP_INSTALL_RULES}")
  return()

endif()

if(0)
set(CONFIG_INSTALL_DIR "${BOOST_INSTALL_CMAKEDIR}/Boost-${BOOST_SUPERPROJECT_VERSION}")
set(CONFIG_FILE_NAME "${CMAKE_CURRENT_LIST_DIR}/../config/BoostConfig.cmake")

install(FILES "${CONFIG_FILE_NAME}" DESTINATION "${CONFIG_INSTALL_DIR}")

set(CONFIG_VERSION_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/tmpinst/BoostConfigVersion.cmake")

if(NOT CMAKE_VERSION VERSION_LESS 3.14)

  write_basic_package_version_file("${CONFIG_VERSION_FILE_NAME}" COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

else()

  set(OLD_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
  set(CMAKE_SIZEOF_VOID_P "")

  write_basic_package_version_file("${CONFIG_VERSION_FILE_NAME}" COMPATIBILITY SameMajorVersion)

  set(CMAKE_SIZEOF_VOID_P ${OLD_CMAKE_SIZEOF_VOID_P})

endif()

install(FILES "${CONFIG_VERSION_FILE_NAME}" DESTINATION "${CONFIG_INSTALL_DIR}")
endif()