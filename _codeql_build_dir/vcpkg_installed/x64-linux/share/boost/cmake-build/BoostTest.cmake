# Copyright 2018-2023 Peter Dimov
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

# Clear global variables on each `include(BoostTest)`

set(BOOST_TEST_LINK_LIBRARIES "")
set(BOOST_TEST_COMPILE_DEFINITIONS "")
set(BOOST_TEST_COMPILE_OPTIONS "")
set(BOOST_TEST_COMPILE_FEATURES "")
set(BOOST_TEST_INCLUDE_DIRECTORIES "")
set(BOOST_TEST_SOURCES "")
set(BOOST_TEST_WORKING_DIRECTORY "")
set(BOOST_TEST_PREFIX "")

# Include guard

if(NOT CMAKE_VERSION VERSION_LESS 3.10)
  include_guard()
endif()

include(BoostMessage)

# Private helper functions

function(__boost_test_list_replace list what with)

  set(result "")

  foreach(x IN LISTS ${list})

    if(x STREQUAL what)
      set(x ${with})
    endif()

    list(APPEND result ${x})

  endforeach()

  set(${list} ${result} PARENT_SCOPE)

endfunction()

# boost_test( [TYPE type] [PREFIX prefix] [NAME name] [WORKING_DIRECTORY wd] [IGNORE_TEST_GLOBALS]
#   SOURCES sources...
#   ARGUMENTS args...
#   LINK_LIBRARIES libs...
#   COMPILE_DEFINITIONS defs...
#   COMPILE_OPTIONS opts...
#   COMPILE_FEATURES features...
#   INCLUDE_DIRECTORIES dirs...
# )

function(boost_test)

  cmake_parse_arguments(_
    "IGNORE_TEST_GLOBALS"
    "TYPE;PREFIX;NAME;WORKING_DIRECTORY"
    "SOURCES;ARGUMENTS;LIBRARIES;LINK_LIBRARIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS;COMPILE_FEATURES;INCLUDE_DIRECTORIES"
    ${ARGN})

  if(NOT __TYPE)
    set(__TYPE run)
  endif()

  if(NOT __NAME)
    list(GET __SOURCES 0 __NAME)
    get_filename_component(__NAME ${__NAME} NAME_WE)
    string(MAKE_C_IDENTIFIER ${__NAME} __NAME)
  endif()

  if(__UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING "Extra arguments for test '${__NAME}' ignored: ${__UNPARSED_ARGUMENTS}")
  endif()

  if(__LIBRARIES)
    boost_message(VERBOSE "Test '${__NAME}' uses deprecated parameter LIBRARIES; use LINK_LIBRARIES")
  endif()

  if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    return()
  endif()

  if(__IGNORE_TEST_GLOBALS)

    set(BOOST_TEST_LINK_LIBRARIES "")
    set(BOOST_TEST_COMPILE_DEFINITIONS "")
    set(BOOST_TEST_COMPILE_OPTIONS "")
    set(BOOST_TEST_COMPILE_FEATURES "")
    set(BOOST_TEST_INCLUDE_DIRECTORIES "")
    set(BOOST_TEST_SOURCES "")
    set(BOOST_TEST_WORKING_DIRECTORY "")
    set(BOOST_TEST_PREFIX "")

  endif()

  list(APPEND BOOST_TEST_LINK_LIBRARIES ${__LIBRARIES} ${__LINK_LIBRARIES})
  list(APPEND BOOST_TEST_COMPILE_DEFINITIONS ${__COMPILE_DEFINITIONS})
  list(APPEND BOOST_TEST_COMPILE_OPTIONS ${__COMPILE_OPTIONS})
  list(APPEND BOOST_TEST_COMPILE_FEATURES ${__COMPILE_FEATURES})
  list(APPEND BOOST_TEST_INCLUDE_DIRECTORIES ${__INCLUDE_DIRECTORIES})
  list(APPEND BOOST_TEST_SOURCES ${__SOURCES})

  if(__WORKING_DIRECTORY)
    set(BOOST_TEST_WORKING_DIRECTORY ${__WORKING_DIRECTORY})
  endif()

  if(__PREFIX)
    set(BOOST_TEST_PREFIX ${__PREFIX})
  endif()

  if(NOT BOOST_TEST_PREFIX)
    set(BOOST_TEST_PREFIX ${PROJECT_NAME})
  endif()

  set(__NAME ${BOOST_TEST_PREFIX}-${__NAME})

  if(MSVC)

    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-fexceptions" "/GX")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-fno-exceptions" "/GX-")

    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-frtti" "/GR")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-fno-rtti" "/GR-")

    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-w" "/W0")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-Wall" "/W4")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-Wextra" "")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-pedantic" "")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-Wpedantic" "")

    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-Werror" "/WX")
    __boost_test_list_replace(BOOST_TEST_COMPILE_OPTIONS "-Wno-error" "/WX-")

  endif()

  foreach(feature IN LISTS BOOST_TEST_COMPILE_FEATURES)
    if(NOT feature IN_LIST CMAKE_CXX_COMPILE_FEATURES)

      boost_message(VERBOSE "Test '${__NAME}' skipped, '${feature}' is not supported")
      return()

    endif()
  endforeach()

  foreach(library IN LISTS BOOST_TEST_LINK_LIBRARIES)

    if(TARGET ${library})
      get_target_property(features ${library} INTERFACE_COMPILE_FEATURES)

      if(features) # need to check because features-NOTFOUND is a valid list
        foreach(feature IN LISTS features)
          if(NOT feature IN_LIST CMAKE_CXX_COMPILE_FEATURES)

            boost_message(VERBOSE "Test '${__NAME}' skipped, '${feature}' required by '${library}' is not supported")
            return()

          endif()
        endforeach()
      endif()
    endif()
  endforeach()

  if(NOT TARGET tests)
    add_custom_target(tests)
  endif()

  if(NOT TARGET tests-quick)
    add_custom_target(tests-quick)
  endif()

  if(__TYPE STREQUAL "compile")

    add_library(${__NAME} STATIC EXCLUDE_FROM_ALL ${BOOST_TEST_SOURCES})
    target_link_libraries(${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_dependencies(tests ${__NAME})

    if("${__NAME}" MATCHES "quick")
      add_dependencies(tests-quick ${__NAME})
    endif()

  elseif(__TYPE STREQUAL "compile-fail")

    add_library(${__NAME} STATIC EXCLUDE_FROM_ALL ${BOOST_TEST_SOURCES})
    target_link_libraries(${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_test(NAME ${__TYPE}-${__NAME} COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${__NAME} --config $<CONFIG>)

    set_tests_properties(${__TYPE}-${__NAME} PROPERTIES WILL_FAIL TRUE RUN_SERIAL TRUE)

  elseif(__TYPE STREQUAL "link")

    add_executable(${__NAME} EXCLUDE_FROM_ALL ${BOOST_TEST_SOURCES})
    target_link_libraries(${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_dependencies(tests ${__NAME})

    if("${__NAME}" MATCHES "quick")
      add_dependencies(tests-quick ${__NAME})
    endif()

  elseif(__TYPE STREQUAL "link-fail")

    add_library(compile-${__NAME} OBJECT EXCLUDE_FROM_ALL ${BOOST_TEST_SOURCES})
    target_link_libraries(compile-${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(compile-${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(compile-${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(compile-${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(compile-${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_dependencies(tests compile-${__NAME})

    if("${__NAME}" MATCHES "quick")
      add_dependencies(tests-quick compile-${__NAME})
    endif()

    add_executable(${__NAME} EXCLUDE_FROM_ALL $<TARGET_OBJECTS:compile-${__NAME}>)
    target_link_libraries(${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_test(NAME ${__TYPE}-${__NAME} COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${__NAME} --config $<CONFIG>)
    set_tests_properties(${__TYPE}-${__NAME} PROPERTIES WILL_FAIL TRUE RUN_SERIAL TRUE)

  elseif(__TYPE STREQUAL "run" OR __TYPE STREQUAL "run-fail")

    add_executable(${__NAME} EXCLUDE_FROM_ALL ${BOOST_TEST_SOURCES})
    target_link_libraries(${__NAME} ${BOOST_TEST_LINK_LIBRARIES})
    target_compile_definitions(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_DEFINITIONS})
    target_compile_options(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_OPTIONS})
    target_compile_features(${__NAME} PRIVATE ${BOOST_TEST_COMPILE_FEATURES})
    target_include_directories(${__NAME} PRIVATE ${BOOST_TEST_INCLUDE_DIRECTORIES})

    add_dependencies(tests ${__NAME})

    if("${__NAME}" MATCHES "quick")
      add_dependencies(tests-quick ${__NAME})
    endif()

    add_test(NAME ${__TYPE}-${__NAME} COMMAND ${__NAME} ${__ARGUMENTS})

    if(__TYPE STREQUAL "run-fail")
      set_tests_properties(${__TYPE}-${__NAME} PROPERTIES WILL_FAIL TRUE)
    endif()

    if(BOOST_TEST_WORKING_DIRECTORY)
      set_target_properties(${__NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${BOOST_TEST_WORKING_DIRECTORY}")
      set_tests_properties(${__TYPE}-${__NAME} PROPERTIES WORKING_DIRECTORY "${BOOST_TEST_WORKING_DIRECTORY}")
    endif()

  else()

    message(AUTHOR_WARNING "Unknown test type '${__TYPE}' for test '${__NAME}'")

  endif()

endfunction(boost_test)
