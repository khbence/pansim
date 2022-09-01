include(${CMAKE_CURRENT_LIST_DIR}/utility/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/io/CMakeLists.txt)

target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})
