target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/abstractPolicy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/movementPolicies.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/policies)