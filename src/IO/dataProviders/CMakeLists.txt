target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dataProvider.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/progressionType.cpp)

target_include_directories(${MAIN_BINARY} PUBLIC include/IO/dataProviders)
