target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/basicLengthAbstract.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/multiBadMatrix.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/singleBadMatrix.cpp)

target_include_directories(${MAIN_BINARY}
                           PUBLIC include/tools/progressionMatrices)
