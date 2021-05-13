target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/basicLengthAbstract.cpp)
target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/multiBadMatrix.cpp)
target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/singleBadMatrix.cpp)

target_include_directories(${SRC_LIBRARY}
                           PUBLIC include/tools/progressionMatrices)
