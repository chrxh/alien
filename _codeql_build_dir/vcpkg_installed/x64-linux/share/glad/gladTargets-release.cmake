#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "glad::glad" for configuration "Release"
set_property(TARGET glad::glad APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glad::glad PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libglad.a"
  )

list(APPEND _cmake_import_check_targets glad::glad )
list(APPEND _cmake_import_check_files_for_glad::glad "${_IMPORT_PREFIX}/lib/libglad.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
