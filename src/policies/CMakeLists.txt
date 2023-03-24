target_include_directories(${MAIN_BINARY} PUBLIC include/policies)

if (Matlab_FOUND)
    target_include_directories(${MEX_LIB} PUBLIC include/policies)
endif()