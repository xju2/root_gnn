# add pythia path
if (NOT DEFINED PYTHIA8_DIR)
	set(PYTHIA8_DIR /usr/local/include/Pythia8)
endif ()

include_directories(${PYTHIA8_DIR}/include)
link_directories(${PYTHIA8_DIR}/lib)

if (NOT DEFINED HEPMC_DIR)
	set(HEPMC_DIR  /usr/local/include/HepMC2)
endif()

include_directories(${HEPMC_DIR}/include)
link_directories   (${HEPMC_DIR}/lib)

# add pythia executables
add_executable(pythia pythia.cxx)
set_target_properties(pythia PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pythia pythia8 HepMC)