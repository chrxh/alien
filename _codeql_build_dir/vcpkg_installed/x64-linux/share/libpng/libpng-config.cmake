
  include(CMakeFindDependencyMacro)
  find_dependency(ZLIB)
  include("${CMAKE_CURRENT_LIST_DIR}/libpng16.cmake")
  