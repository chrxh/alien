#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CLI11::CLI11" for configuration "Debug"
set_property(TARGET CLI11::CLI11 APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(CLI11::CLI11 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/libCLI11.a"
  )

list(APPEND _cmake_import_check_targets CLI11::CLI11 )
list(APPEND _cmake_import_check_files_for_CLI11::CLI11 "${_IMPORT_PREFIX}/debug/lib/libCLI11.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
