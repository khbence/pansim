target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentsFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/agentTypesFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/closureFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/configRandomFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/locationsFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/locationTypesFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/parametersFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/progressionConfigFormat.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/progressionMatrixFormat.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/io/JSONReaders/inputFormats)