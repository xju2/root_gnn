# Find the Delphes includes and libraries.
#
# This module defines the `Delphes` imported target that encodes all
# necessary information in its target properties.
include(FindPackageHandleStandardArgs)

find_library(Delphes_LIBRARY
  NAMES libDelphes.so libDelphesDisplay.so
  HINTS ENV Delphes_DIR Delphes
  PATHS /opt/Delphes /usr/local
  DOC "The Delphes library")

find_path(Delphes_INCLUDE_DIR
  NAMES classes/DelphesClasses.h 
  HINTS ENV Delphes_DIR Delphes
  PATHS /opt/Delphes /usr/local
  DOC "The Delphes include directory")

find_package_handle_standard_args(Delphes
  REQUIRED_VARS Delphes_LIBRARY Delphes_INCLUDE_DIR)

add_library(Delphes SHARED IMPORTED)
set_property(TARGET Delphes PROPERTY IMPORTED_LOCATION ${Delphes_LIBRARY})
set_property(TARGET Delphes PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Delphes_INCLUDE_DIR})

mark_as_advanced(Delphes_FOUND Delphes_INCLUDE_DIR Delphes_LIBRARY)