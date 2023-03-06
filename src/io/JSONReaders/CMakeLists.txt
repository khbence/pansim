target_sources(
  ${SRC_LIBRARY}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/JSONReader.cpp)

target_include_directories(${SRC_LIBRARY} PUBLIC include/io/JSONReaders)

include(${CMAKE_CURRENT_LIST_DIR}/inputFormats/CMakeLists.txt)