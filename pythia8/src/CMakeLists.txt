if(CMAKE_BUILD_TYPE MATCHES Debug)
	message(STATUS "In Debug Mode")
	add_compile_options(-g)
	#add_compile_options(-DDebug)
endif()

add_compile_options(-Wfatal-errors)
add_compile_options(-Wno-deprecated)
add_compile_options(-std=c++11)
add_compile_options(-DGZIPSUPPORT)

# add executables
add_subdirectory(util)