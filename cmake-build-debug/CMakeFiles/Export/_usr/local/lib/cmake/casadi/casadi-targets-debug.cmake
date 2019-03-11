#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "casadi" for configuration "Debug"
set_property(TARGET casadi APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(casadi PROPERTIES
  IMPORTED_LOCATION_DEBUG "/usr/local/lib/libcasadi.3.5.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libcasadi.3.5.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS casadi )
list(APPEND _IMPORT_CHECK_FILES_FOR_casadi "/usr/local/lib/libcasadi.3.5.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
