target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/policy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/closurePolicy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/infectionPolicy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/movementPolicy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/testingPolicy.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/policies)