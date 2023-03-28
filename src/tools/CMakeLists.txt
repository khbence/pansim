target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeDay.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeHandler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/logger.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/smallTools.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/randomGenerator.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/globalStates.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/tools)