#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "freetype" for configuration "Debug"
set_property(TARGET freetype APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(freetype PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/libfreetyped.a"
  )

list(APPEND _cmake_import_check_targets freetype )
list(APPEND _cmake_import_check_files_for_freetype "${_IMPORT_PREFIX}/debug/lib/libfreetyped.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
