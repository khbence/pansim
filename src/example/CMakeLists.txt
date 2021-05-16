target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/example.cpp)
# set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/example.cpp PROPERTIES
# LANGUAGE CUDA)

target_include_directories(${MAIN_BINARY} PUBLIC include/example)
