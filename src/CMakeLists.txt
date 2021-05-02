include(${CMAKE_CURRENT_LIST_DIR}/example/CMakeLists.txt)

add_executable(${MAIN_BINARY})
target_include_directories(${SRC_LIBRARY} PRIVATE include)
target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
