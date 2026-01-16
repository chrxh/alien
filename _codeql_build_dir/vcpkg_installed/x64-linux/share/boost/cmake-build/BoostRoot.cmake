# Copyright 2019-2023 Peter Dimov
# Copyright 2025 Alexander Grund
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

if(CMAKE_SOURCE_DIR STREQUAL Boost_SOURCE_DIR AND WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

    set(CMAKE_INSTALL_PREFIX "C:/Boost" CACHE PATH "Installation path prefix, prepended to installation directories" FORCE)

endif()

include(BoostMessage)
include(BoostInstall)

#

if(CMAKE_SOURCE_DIR STREQUAL Boost_SOURCE_DIR)
  boost_message(STATUS "Boost: using CMake ${CMAKE_VERSION}")
else()
  boost_message(VERBOSE "Boost: using CMake ${CMAKE_VERSION}")
endif()

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
    "Build and enable installation of Boost.MPI and its dependents (requires MPI, CMake 3.10)")

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

  add_custom_target(check VERBATIM COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -C $<CONFIG>)
  add_dependencies(check tests)

  if(NOT TARGET tests-quick)
    add_custom_target(tests-quick)
  endif()

  add_custom_target(check-quick VERBATIM COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -C $<CONFIG> -R quick)
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

      get_target_property(is_installed "boost_${__boost_lib_target}" _boost_is_installed)
      if(is_installed)
        return() # Ignore libraries for which boost_install was already called
      endif()

      get_target_property(__boost_lib_incdir "boost_${__boost_lib_target}" INTERFACE_INCLUDE_DIRECTORIES)

      set(incdir "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${__boost_lib}/include")

      set(extradir "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${__boost_lib}/extra")
      if(NOT EXISTS "${extradir}")
        set(extradir "")
      endif()

      if("${__boost_lib_incdir}" STREQUAL "${incdir}" OR "${__boost_lib_incdir}" STREQUAL "$<BUILD_INTERFACE:${incdir}>")

        boost_message(DEBUG "Enabling installation for '${__boost_lib}'")
        boost_install(TARGETS "boost_${__boost_lib_target}" VERSION "${BOOST_SUPERPROJECT_VERSION}" HEADER_DIRECTORY "${incdir}" EXTRA_DIRECTORY "${extradir}")

      else()
        boost_message(DEBUG "Not enabling installation for '${__boost_lib}'; interface include directory '${__boost_lib_incdir}' does not equal '${incdir}' or '$<BUILD_INTERFACE:${incdir}>'")
      endif()

    else()
      boost_message(DEBUG "Not enabling installation for '${__boost_lib}'; targets 'Boost::${__boost_lib_target}' and 'boost_${__boost_lib_target}' weren't found")
    endif()

  endif()
endfunction()

function(__boost_scan_dependencies lib var sub_folder)

  # Libraries that define at least one library with a name like "<prefix>_"
  set(prefix_names "asio" "dll" "fiber" "log" "regex" "stacktrace")
  set(result "")
  set(required_components "")
  set(optional_components "")
  set(exclude_components "")

  set(cml_files "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/${lib}")
  if(sub_folder)
    file(GLOB_RECURSE cml_files "${cml_files}/${sub_folder}/CMakeLists.txt")
  else()
    string(APPEND cml_files "/CMakeLists.txt")
  endif()

  foreach(cml_file IN LISTS cml_files)
    if(NOT EXISTS "${cml_file}")
      CONTINUE()
    endif()
    set(libs_to_exclude "")

    file(STRINGS "${cml_file}" data)

    foreach(line IN LISTS data)
      if(line MATCHES "^ *# *Boost-(Include|Exclude):? *(.*)$")
        set(type ${CMAKE_MATCH_1})
        set(line ${CMAKE_MATCH_2})
      else()
        set(type "Include")
      endif()
      if(line MATCHES "^([^#]*Boost::[A-Za-z0-9_]+[^#]*)(#.*)?$")
        string(REGEX MATCHALL "Boost::[A-Za-z0-9_]+" libs "${CMAKE_MATCH_1}")

        foreach(dep IN LISTS libs)
          string(REGEX REPLACE "^Boost::" "" dep ${dep})
          # Check if this dependency is optional in the original line
          set(_is_optional OFF)
          if(line MATCHES "\\$<TARGET_NAME_IF_EXISTS:Boost::${dep}>")
            set(_is_optional ON)
          endif()
          if(dep STREQUAL "headers" OR dep STREQUAL "boost" OR dep MATCHES "linking")
            continue()
          endif()
          if(dep MATCHES "(included_)?(unit_test_framework|prg_exec_monitor|test_exec_monitor)")
            set(dep "test")
          elseif(dep STREQUAL "numpy")
            set(dep "python")
          elseif(dep MATCHES "serialization")
            set(dep "serialization")
          else()
            string(REGEX REPLACE "^numeric_" "numeric/" dep ${dep})
            foreach(prefix IN LISTS prefix_names)
              if(dep MATCHES "^${prefix}_")
                set(dep ${prefix})
                break()
              endif()
            endforeach()
          endif()

          set(component_name ${dep})
          string(REGEX REPLACE "^numeric/" "numeric_" component_name ${component_name})
          
          if(NOT dep STREQUAL lib)
            if(type STREQUAL "Exclude")
              list(APPEND libs_to_exclude ${dep})
              list(APPEND exclude_components ${component_name})
            else()
              list(APPEND result ${dep})
              if(_is_optional)
                list(APPEND optional_components ${component_name})
              else()
                list(APPEND required_components ${component_name})
              endif()
            endif()
          endif()
        endforeach()

      endif()

    endforeach()

  endforeach()

  if(libs_to_exclude)
    list(REMOVE_ITEM result ${libs_to_exclude})
    list(REMOVE_ITEM optional_components ${exclude_components})
    list(REMOVE_ITEM required_components ${exclude_components})
  endif()

  list(REMOVE_DUPLICATES required_components)
  list(REMOVE_DUPLICATES optional_components)
  list(REMOVE_ITEM required_components boost ${lib})
  if(required_components OR optional_components)
    message(STATUS "boost required: " ${required_components})
    message(STATUS "boost optional: " ${optional_components})
    find_package(Boost COMPONENTS ${required_components} OPTIONAL_COMPONENTS ${optional_components} REQUIRED CONFIG)
  endif()

  list(REMOVE_DUPLICATES result)
  set(${var} ${result} PARENT_SCOPE)

endfunction()

macro(__boost_add_header_only lib)

  if(TARGET "boost_${lib}" AND TARGET "Boost::${lib}")

    get_target_property(__boost_lib_type "boost_${lib}" TYPE)

    if(__boost_lib_type STREQUAL "INTERFACE_LIBRARY")

      list(APPEND __boost_header_only "Boost::${lib}")

    endif()

    set(__boost_lib_type)
  endif()

endmacro()

#

file(GLOB __boost_libraries RELATIVE "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs" "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/*/CMakeLists.txt" "${BOOST_SUPERPROJECT_SOURCE_DIR}/libs/numeric/*/CMakeLists.txt")

# Check for mistakes in BOOST_INCLUDE_LIBRARIES
if(BOOST_INCLUDE_LIBRARIES)

  set(__boost_any_library_found OFF)

  foreach(__boost_included_lib IN LISTS BOOST_INCLUDE_LIBRARIES)

    if(NOT "${__boost_included_lib}/CMakeLists.txt" IN_LIST __boost_libraries)

      message(WARNING "Library '${__boost_included_lib}' given in BOOST_INCLUDE_LIBRARIES has not been found.")

    else()

      set(__boost_any_library_found ON)

    endif()

  endforeach()

  if(NOT __boost_any_library_found)

    message(FATAL_ERROR "None of the libraries given in BOOST_INCLUDE_LIBRARIES has been found so no libraries would be built. Verify BOOST_INCLUDE_LIBRARIES ('${BOOST_INCLUDE_LIBRARIES}')")

  endif()

endif()

# Scan for dependencies

function(__boost_gather_dependencies var input_list with_test)
  set(result "")

  set(libs_to_scan ${input_list})
  while(libs_to_scan)

    boost_message(DEBUG "Scanning dependencies: ${libs_to_scan}")

    set(cur_dependencies "")

    foreach(lib IN LISTS libs_to_scan)

      __boost_scan_dependencies(${lib} new_deps "")
      list(APPEND cur_dependencies ${new_deps})
      # Only consider test dependencies of the input libraries, not transitively as those tests aren't build
      if(with_test AND lib IN_LIST input_list)
          __boost_scan_dependencies(${lib} new_deps "test")
          list(APPEND cur_dependencies ${new_deps})
          __boost_scan_dependencies(${lib} new_deps "example")
          list(APPEND cur_dependencies ${new_deps})
      endif()

    endforeach()

    list(REMOVE_DUPLICATES cur_dependencies)

    if(cur_dependencies)
      list(REMOVE_ITEM cur_dependencies ${libs_to_scan} ${result})
      list(APPEND result ${cur_dependencies})
    endif()

    set(libs_to_scan ${cur_dependencies})

  endwhile()

  list(REMOVE_ITEM result ${input_list})
  set(${var} ${result} PARENT_SCOPE)
endfunction()

if(BOOST_INCLUDE_LIBRARIES)
  list(REMOVE_DUPLICATES BOOST_INCLUDE_LIBRARIES)
  __boost_gather_dependencies(__boost_dependencies "${BOOST_INCLUDE_LIBRARIES}" OFF)
  if(BUILD_TESTING)
    __boost_gather_dependencies(__boost_test_dependencies "${BOOST_INCLUDE_LIBRARIES}" ON)
    if(__boost_dependencies)
      list(REMOVE_ITEM __boost_test_dependencies ${__boost_dependencies})
    endif()
  endif()
else()
  set(__boost_dependencies "")
  set(__boost_test_dependencies "")
endif()

# Installing targets created in other directories requires CMake 3.13
if(CMAKE_VERSION VERSION_LESS 3.13)

  boost_message(STATUS "Boost installation support requires CMake 3.13 (have ${CMAKE_VERSION})")

endif()

set(__boost_mpi_libs mpi graph_parallel property_map_parallel)
set(__boost_python_libs python parameter_python)

set(__boost_header_only "")

foreach(__boost_lib_cml IN LISTS __boost_libraries)

  get_filename_component(__boost_lib "${__boost_lib_cml}" DIRECTORY)

  if(__boost_lib IN_LIST BOOST_INCOMPATIBLE_LIBRARIES)

    boost_message(DEBUG "Skipping incompatible Boost library '${__boost_lib}'")

  elseif(__boost_lib IN_LIST BOOST_EXCLUDE_LIBRARIES)

    boost_message(DEBUG "Skipping excluded Boost library '${__boost_lib}'")

  elseif(NOT BOOST_ENABLE_MPI AND __boost_lib IN_LIST __boost_mpi_libs)

    if(__boost_lib IN_LIST BOOST_INCLUDE_LIBRARIES)

      message(SEND_ERROR "Boost library '${__boost_lib}' has been explicitly requested, but BOOST_ENABLE_MPI is OFF. Set BOOST_ENABLE_MPI to ON.")

    elseif(NOT BOOST_INCLUDE_LIBRARIES)

      message(STATUS "Skipping Boost library '${__boost_lib}', BOOST_ENABLE_MPI is OFF")

    else()

      boost_message(DEBUG "Skipping Boost library '${__boost_lib}', BOOST_ENABLE_MPI is OFF")

    endif()

  elseif(NOT BOOST_ENABLE_PYTHON AND __boost_lib IN_LIST __boost_python_libs)

    if(__boost_lib IN_LIST BOOST_INCLUDE_LIBRARIES)

      message(SEND_ERROR "Boost library '${__boost_lib}' has been explicitly requested, but BOOST_ENABLE_PYTHON is OFF. Set BOOST_ENABLE_PYTHON to ON.")

    elseif(NOT BOOST_INCLUDE_LIBRARIES)

      message(STATUS "Skipping Boost library '${__boost_lib}', BOOST_ENABLE_PYTHON is OFF")

    else()

      boost_message(DEBUG "Skipping Boost library '${__boost_lib}', BOOST_ENABLE_PYTHON is OFF")

    endif()

  elseif(NOT BOOST_INCLUDE_LIBRARIES OR __boost_lib IN_LIST BOOST_INCLUDE_LIBRARIES)

    boost_message(VERBOSE "Adding Boost library '${__boost_lib}'")
    add_subdirectory(libs/${__boost_lib})

    __boost_auto_install(${__boost_lib})
    __boost_add_header_only(${__boost_lib})

  elseif(__boost_lib IN_LIST __boost_dependencies OR __boost_lib STREQUAL "headers")

    # Disable tests for dependencies

    set(__boost_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF) # hide cache variable

    set(__boost_cmake_folder ${CMAKE_FOLDER})

    if("${CMAKE_FOLDER}" STREQUAL "")
      set(CMAKE_FOLDER "Dependencies")
    endif()

    boost_message(VERBOSE "Adding Boost dependency '${__boost_lib}'")
    add_subdirectory(libs/${__boost_lib})

    __boost_auto_install(${__boost_lib})
    __boost_add_header_only(${__boost_lib})

    set(BUILD_TESTING ${__boost_build_testing})
    set(CMAKE_FOLDER ${__boost_cmake_folder})

  elseif(__boost_lib IN_LIST __boost_test_dependencies)

    # Disable tests and installation for libraries not included but used as test dependencies

    set(__boost_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF) # hide cache variable

    set(__boost_skip_install ${BOOST_SKIP_INSTALL_RULES})
    set(BOOST_SKIP_INSTALL_RULES ON)

    set(__boost_cmake_folder ${CMAKE_FOLDER})

    if("${CMAKE_FOLDER}" STREQUAL "")
      set(CMAKE_FOLDER "Test Dependencies")
    endif()

    boost_message(DEBUG "Adding Boost test dependency '${__boost_lib}' with EXCLUDE_FROM_ALL")
    add_subdirectory(libs/${__boost_lib} EXCLUDE_FROM_ALL)

    set(BUILD_TESTING ${__boost_build_testing})
    set(BOOST_SKIP_INSTALL_RULES ${__boost_skip_install})
    set(CMAKE_FOLDER ${__boost_cmake_folder})

  endif()

endforeach()

# Compatibility targets for use with add_subdirectory/FetchContent

if(BOOST_ENABLE_COMPATIBILITY_TARGETS)

  # Boost::headers

  list(REMOVE_ITEM __boost_header_only Boost::headers)
  target_link_libraries(boost_headers INTERFACE ${__boost_header_only})

  # Boost::boost

  add_library(boost_comptarget_boost INTERFACE)
  add_library(Boost::boost ALIAS boost_comptarget_boost)
  target_link_libraries(boost_comptarget_boost INTERFACE Boost::headers)

  # Boost::diagnostic_definitions

  add_library(boost_comptarget_diagnostic_definitions INTERFACE)
  add_library(Boost::diagnostic_definitions ALIAS boost_comptarget_diagnostic_definitions)
  target_compile_definitions(boost_comptarget_diagnostic_definitions INTERFACE BOOST_LIB_DIAGNOSTIC)

  # Boost::disable_autolinking

  add_library(boost_comptarget_disable_autolinking INTERFACE)
  add_library(Boost::disable_autolinking ALIAS boost_comptarget_disable_autolinking)
  target_compile_definitions(boost_comptarget_disable_autolinking INTERFACE BOOST_ALL_NO_LIB)

  # Boost::dynamic_linking

  add_library(boost_comptarget_dynamic_linking INTERFACE)
  add_library(Boost::dynamic_linking ALIAS boost_comptarget_dynamic_linking)
  target_compile_definitions(boost_comptarget_dynamic_linking INTERFACE BOOST_ALL_DYN_LINK)

endif()

# Install BoostConfig.cmake

if(BOOST_SKIP_INSTALL_RULES)

  boost_message(DEBUG "Boost: not installing BoostConfig.cmake due to BOOST_SKIP_INSTALL_RULES=${BOOST_SKIP_INSTALL_RULES}")
  return()

endif()

if(CMAKE_SKIP_INSTALL_RULES)

  boost_message(DEBUG "Boost: not installing BoostConfig.cmake due to CMAKE_SKIP_INSTALL_RULES=${CMAKE_SKIP_INSTALL_RULES}")
  return()

endif()
