target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentMeta.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStat.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentType.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dynamicPPState.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/globalStates.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/PPStateTypes.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/agentData)
