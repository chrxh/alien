# Copyright 2018, 2019 Peter Dimov
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

# Include BoostTest outside the include guard for it to clear its global variables
include(BoostTest)

if(NOT CMAKE_VERSION VERSION_LESS 3.10)
  include_guard()
endif()

if(BUILD_TESTING AND CMAKE_VERSION VERSION_LESS 3.9)
  message(AUTHOR_WARNING "BoostTestJamfile requires CMake 3.9") # CMAKE_MATCH_x
endif()

include(BoostMessage)

# boost_test_jamfile( FILE jamfile [PREFIX prefix]
#   LINK_LIBRARIES libs...
#   COMPILE_DEFINITIONS defs...
#   COMPILE_OPTIONS opts...
#   COMPILE_FEATURES features...
#   INCLUDE_DIRECTORIES dirs...
# )

function(boost_test_jamfile)

  cmake_parse_arguments(_
    ""
    "FILE;PREFIX"
    "LIBRARIES;LINK_LIBRARIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS;COMPILE_FEATURES;INCLUDE_DIRECTORIES"
    ${ARGN})

  if(__UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING "boost_test_jamfile: extra arguments ignored: ${__UNPARSED_ARGUMENTS}")
  endif()

  if(__LIBRARIES)
    message(AUTHOR_WARNING "boost_test_jamfile: LIBRARIES is deprecated, use LINK_LIBRARIES")
  endif()

  if(NOT __FILE)
    message(AUTHOR_WARNING "boost_test_jamfile: required argument FILE is missing")
    return()
  endif()

  if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    return()
  endif()

  file(STRINGS "${__FILE}" data)

  set(types "compile|compile-fail|link|link-fail|run|run-fail")

  foreach(line IN LISTS data)
    # Extract type and remaining part, (silently) ignore any other line
    if(line MATCHES "^[ \t]*(${types})([ \t].*|$)")
      set(type ${CMAKE_MATCH_1})
      set(args ${CMAKE_MATCH_2}) # This starts with a space

      if(args MATCHES "^[ \t]+([^ \t]+)[ \t]*(;[ \t]*)?$")
        # Single source, e.g. 'run foo.c ;'
        # Semicolon is optional to support e.g. 'run mytest.cpp\n : : : something ;' (ignore 'something')
        set(sources ${CMAKE_MATCH_1})
      elseif(args MATCHES "^(([ \t]+[a-zA-Z0-9_]+\.cpp)+)[ \t]*(;[ \t]*)?$")
        # Multiple sources with restricted names to avoid false positives, e.g. 'run foo.cpp bar.cpp ;'
        # Again with optional semicolon
        string(STRIP "${CMAKE_MATCH_1}" sources)
        # Convert space-separated list into CMake list
        string(REGEX REPLACE "\.cpp[ \t]+" ".cpp;" sources "${sources}")
      else()
        boost_message(VERBOSE "boost_test_jamfile: Jamfile line ignored: ${line}")
        continue()
      endif()

      boost_test(PREFIX "${__PREFIX}" TYPE "${type}"
        SOURCES ${sources}
        LINK_LIBRARIES ${__LIBRARIES} ${__LINK_LIBRARIES}
        COMPILE_DEFINITIONS ${__COMPILE_DEFINITIONS}
        COMPILE_OPTIONS ${__COMPILE_OPTIONS}
        COMPILE_FEATURES ${__COMPILE_FEATURES}
        INCLUDE_DIRECTORIES ${__INCLUDE_DIRECTORIES}
      )
    endif()

  endforeach()
endfunction()
