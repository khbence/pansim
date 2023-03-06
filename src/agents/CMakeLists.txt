target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentHandler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentMeta.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentStat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentType.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dynamicPPState.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/PPStateTypes.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/agents)

include(${CMAKE_CURRENT_LIST_DIR}/progressionMatrices/CMakeLists.txt)