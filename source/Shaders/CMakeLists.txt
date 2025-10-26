cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(Shaders)

# This is a header-only library
add_library(Shaders INTERFACE)

target_include_directories(Shaders INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
