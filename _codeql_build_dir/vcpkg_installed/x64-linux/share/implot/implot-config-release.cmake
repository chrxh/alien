#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "implot::implot" for configuration "Release"
set_property(TARGET implot::implot APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(implot::implot PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libimplot.a"
  )

list(APPEND _cmake_import_check_targets implot::implot )
list(APPEND _cmake_import_check_files_for_implot::implot "${_IMPORT_PREFIX}/lib/libimplot.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
