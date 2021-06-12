find_package(ROOT 6.18 REQUIRED COMPONENTS Core Hist Tree TreePlayer)
find_package(Delphes)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

file(GLOB SrcFiles *.cxx)
add_library(DelphesUtils SHARED ${SrcFiles})
target_include_directories(DelphesUtils 
    PUBLIC ${ROOT_INCLUDE_DIRS} ${Delphes_INCLUDE_DIR}
    INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(DelphesUtils
    Delphes
    ROOT::Core ROOT::Tree 
    ROOT::TreePlayer ROOT::Physics
)

install(TARGETS DelphesUtils 
        DESTINATION lib
        EXPORT DelphesUtilsTargets)

file(GLOB HeaderFiles include/DelphesUtils/*.hpp)
install(FILES ${HeaderFiles}
    DESTINATION include/DelphesUtils)