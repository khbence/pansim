include(${CMAKE_CURRENT_LIST_DIR}/agentData/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/IO/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/policies/CMakeLists.txt)

add_executable(${MAIN_BINARY})
target_include_directories(${SRC_LIBRARY} PUBLIC include)
target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
