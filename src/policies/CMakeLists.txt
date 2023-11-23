target_include_directories(${MAIN_BINARY} PUBLIC include/policies)

if (Matlab_FOUND)
    target_include_directories(${MEX_LIB} PUBLIC include/policies)
endif()

if (pybind11_FOUND)
    target_include_directories(${PY_LIB} PUBLIC include/policies)
endif()