#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Boost::wserialization" for configuration "Release"
set_property(TARGET Boost::wserialization APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Boost::wserialization PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libboost_wserialization.a"
  )

list(APPEND _cmake_import_check_targets Boost::wserialization )
list(APPEND _cmake_import_check_files_for_Boost::wserialization "${_IMPORT_PREFIX}/lib/libboost_wserialization.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
