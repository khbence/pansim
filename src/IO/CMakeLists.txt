include(${CMAKE_CURRENT_LIST_DIR}/dataProviders/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/outputs/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/inputFormats/CMakeLists.txt)

target_include_directories(${MAIN_BINARY} PUBLIC include/IO)
if (Matlab_FOUND)
    target_include_directories(${MEX_LIB} PUBLIC include/IO)
endif()

if (pybind11_FOUND)
    target_include_directories(${PY_LIB} PUBLIC include/IO)
endif()
