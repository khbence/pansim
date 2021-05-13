target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dataProvider.cpp)
target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/progressionType.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/IO/dataProviders)
