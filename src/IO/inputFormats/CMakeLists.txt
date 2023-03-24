target_include_directories(${MAIN_BINARY} PUBLIC include/IO/inputFormats)
if (Matlab_FOUND)
    target_include_directories(${MEX_LIB} PUBLIC include/IO/inputFormats)
endif()
