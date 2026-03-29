add_library(
    panSimRuntime
    OBJECT
        ${CMAKE_CURRENT_LIST_DIR}/programOptions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/simulationRuntime.cpp)
target_include_directories(
    panSimRuntime
    PUBLIC
        include
        include/agentData
        include/tools
        include/tools/progressionMatrices
        include/policies
        include/IO
        include/IO/dataProviders
        include/IO/outputs
        include/IO/inputFormats
        cmake/config/out
        rapidjson/include
        cxxopts/include)
target_link_libraries(panSimRuntime PUBLIC OpenMP::OpenMP_CXX)

target_sources(${MAIN_BINARY} PRIVATE $<TARGET_OBJECTS:panSimRuntime>)

if (Matlab_FOUND)
    target_sources(${MEX_LIB} PRIVATE $<TARGET_OBJECTS:panSimRuntime>)
endif()

if (pybind11_FOUND)
    target_sources(${PY_LIB} PRIVATE $<TARGET_OBJECTS:panSimRuntime>)
endif()
