cmake_minimum_required (VERSION 2.6)
project (PythiaGen)

# The version number
set (PythiaGen_VERSION_MAJOR 1)
set (PythiaGen_VERSION_MINOR 0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# source code
add_subdirectory(src)
