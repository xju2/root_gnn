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
# add_compile_options(-std=c++11)
add_compile_options(-DGZIPSUPPORT)

# include_directories(${HEPMC_DIR}/include)
# link_directories   (${HEPMC_DIR}/lib)

if (NOT DEFINED Delphes_DIR)
    set(Delphes_DIR /global/cfs/cdirs/atlas/xju/software/DelphesInstall)
endif()

find_package(ROOT 6.18 REQUIRED COMPONENTS Core Hist Tree TreePlayer)
find_package(Delphes)

# add executables
# message(INFO, ${ROOT_INCLUDE_DIRS})
# include_directories(
# 	${ROOT_INCLUDE_DIRS}
# 	${Delphes_INCLUDE_DIR}
# )

add_executable(delphesAna delphesAna.cxx)
set_target_properties(delphesAna PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(delphesAna PUBLIC ${ROOT_INCLUDE_DIRS} ${Delphes_INCLUDE_DIR})
target_link_libraries(delphesAna Delphes ROOT::Core ROOT::Tree ROOT::TreePlayer)