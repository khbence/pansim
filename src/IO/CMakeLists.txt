include(${CMAKE_CURRENT_LIST_DIR}/dataProviders/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/outputs/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/inputFormats/CMakeLists.txt)

target_include_directories(${SRC_LIBRARY} PUBLIC include/IO)
