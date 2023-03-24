include(${CMAKE_CURRENT_LIST_DIR}/progressionMatrices/CMakeLists.txt)

target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/operators.cpp)
target_sources(${MAIN_BINARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/randomGenerator.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/smallTools.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeDay.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeHandler.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timing.cpp)
target_sources(${MAIN_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/util.cpp)

target_include_directories(${MAIN_BINARY} PUBLIC include/tools)


target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/operators.cpp)
target_sources(${MEX_LIB}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/randomGenerator.cpp)
target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/smallTools.cpp)
target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeDay.cpp)
target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeHandler.cpp)
target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timing.cpp)
target_sources(${MEX_LIB} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/util.cpp)

target_include_directories(${MEX_LIB} PUBLIC include/tools)
