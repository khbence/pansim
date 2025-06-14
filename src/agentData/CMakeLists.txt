target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentMeta.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStat.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentType.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dynamicPPState.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/globalStates.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/PPStateTypes.cpp)

target_include_directories(${MAIN_BINARY} PUBLIC include/agentData)

if (Matlab_FOUND)
    target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentMeta.cpp)
    target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStat.cpp)
    target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentType.cpp)
    target_sources(${MEX_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dynamicPPState.cpp)
    target_sources(${MEX_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/globalStates.cpp)
    target_sources(${MEX_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/PPStateTypes.cpp)

    target_include_directories(${MEX_LIB} PUBLIC include/agentData)
endif()


if (pybind11_FOUND)
    target_sources(${PY_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentMeta.cpp)
    target_sources(${PY_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStat.cpp)
    target_sources(${PY_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentType.cpp)
    target_sources(${PY_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dynamicPPState.cpp)
    target_sources(${PY_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/globalStates.cpp)
    target_sources(${PY_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/PPStateTypes.cpp)

    target_include_directories(${PY_LIB} PUBLIC include/agentData)
endif()