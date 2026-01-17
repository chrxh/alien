#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "implot::implot" for configuration "Debug"
set_property(TARGET implot::implot APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(implot::implot PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/libimplotd.a"
  )

list(APPEND _cmake_import_check_targets implot::implot )
list(APPEND _cmake_import_check_files_for_implot::implot "${_IMPORT_PREFIX}/debug/lib/libimplotd.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
