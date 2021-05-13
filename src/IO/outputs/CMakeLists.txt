target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStatOutput.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/JSONWriter.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/IO/outputs)
