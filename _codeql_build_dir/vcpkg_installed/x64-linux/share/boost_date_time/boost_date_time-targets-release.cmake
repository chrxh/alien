#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Boost::date_time" for configuration "Release"
set_property(TARGET Boost::date_time APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Boost::date_time PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libboost_date_time.a"
  )

list(APPEND _cmake_import_check_targets Boost::date_time )
list(APPEND _cmake_import_check_files_for_Boost::date_time "${_IMPORT_PREFIX}/lib/libboost_date_time.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
