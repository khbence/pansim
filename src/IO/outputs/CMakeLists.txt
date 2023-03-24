target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStatOutput.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/JSONWriter.cpp)

target_include_directories(${MAIN_BINARY} PUBLIC include/IO/outputs)

if (Matlab_FOUND)
    target_sources(${MEX_LIB}
                PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStatOutput.cpp)
    target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/JSONWriter.cpp)

    target_include_directories(${MEX_LIB} PUBLIC include/IO/outputs)
endif()
