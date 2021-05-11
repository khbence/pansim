target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/example.cpp)
#set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/example.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(${SRC_LIBRARY} PUBLIC include/example)
