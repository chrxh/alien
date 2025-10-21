# This config-module creates the following import libraries:
#
# - GLEW::glew shared lib
# - GLEW::glew_s static lib
#
# Additionally GLEW::GLEW will be created as an
# copy of either the shared (default) or the static libs.
#
# Dependending on the setting of BUILD_SHARED_LIBS at GLEW build time
# either the static or shared versions may not be available.
#
# Set GLEW_USE_STATIC_LIBS to OFF or ON to force using the shared
# or static lib for GLEW::GLEW 
#

include(${CMAKE_CURRENT_LIST_DIR}/glew-targets.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/CopyImportedTargetProperties.cmake)

# decide which import library (glew/glew_s)
# needs to be copied to GLEW::GLEW
set(_glew_target_postfix "")
set(_glew_target_type SHARED)
if(DEFINED GLEW_USE_STATIC_LIBS)
  # if defined, use only static or shared
  if(GLEW_USE_STATIC_LIBS)
    set(_glew_target_postfix "_s")
  endif()
  # else use static only if no shared
elseif(NOT TARGET GLEW::glew AND TARGET GLEW::glew_s)
  set(_glew_target_postfix "_s")
endif()
if(_glew_target_postfix STREQUAL "")
  set(_glew_target_type SHARED)
else()
  set(_glew_target_type STATIC)
endif()

# CMake doesn't allow creating ALIAS lib for an IMPORTED lib
# so create imported ones and copy the properties
foreach(_glew_target glew)
  set(_glew_src_target "GLEW::${_glew_target}${_glew_target_postfix}")
  string(TOUPPER "GLEW::${_glew_target}" _glew_dest_target)
  if(TARGET ${_glew_dest_target})
    get_target_property(_glew_previous_src_target ${_glew_dest_target}
      _GLEW_SRC_TARGET)
    if(NOT _glew_previous_src_target STREQUAL _glew_src_target)
      message(FATAL_ERROR "find_package(GLEW) was called the second time with "
        "different GLEW_USE_STATIC_LIBS setting. Previously, "
        "`glew-config.cmake` created ${_glew_dest_target} as a copy of "
        "${_glew_previous_src_target}. Now it attempted to copy it from "
        "${_glew_src_target}. ")
    endif()
  else()
    add_library(${_glew_dest_target} ${_glew_target_type} IMPORTED)
    # message(STATUS "add_library(${_glew_dest_target} ${_glew_target_type} IMPORTED)")
    copy_imported_target_properties(${_glew_src_target} ${_glew_dest_target})
    set_target_properties(${_glew_dest_target} PROPERTIES
      _GLEW_SRC_TARGET ${_glew_src_target})
  endif()
endforeach()
