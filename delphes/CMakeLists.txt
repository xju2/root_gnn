cmake_minimum_required (VERSION 3.16)

# must be set before project(...) call; version module is needed before
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project (DelphesAnalyzer)

# The version number
set (DelphesAnalyzer_VERSION_MAJOR 1)
set (DelphesAnalyzer_VERSION_MINOR 0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


if(CMAKE_BUILD_TYPE MATCHES Debug)
	message(STATUS "In Debug Mode")
	add_compile_options(-g)
	#add_compile_options(-DDebug)
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wfatal-errors)
add_compile_options(-Wno-deprecated)
add_compile_options(-DGZIPSUPPORT)


if (NOT DEFINED Delphes_DIR)
    set(Delphes_DIR /global/cfs/cdirs/atlas/xju/software/DelphesInstall)
endif()

add_subdirectory(libDelphesUtils)

add_executable(AnaDelphes AnaDelphes.cxx)
set_target_properties(AnaDelphes PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(AnaDelphes DelphesUtils)