include(${CMAKE_CURRENT_LIST_DIR}/progressionMatrices/CMakeLists.txt)

target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/operators.cpp)
target_sources(${SRC_LIBRARY}
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/randomGenerator.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/smallTools.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeDay.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timeHandler.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/timing.cpp)
target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/util.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/tools)
