include(FetchContent)

find_package(GTest)

add_executable(${TEST_BINARY} main.cpp)

target_link_libraries(${TEST_BINARY} ${SRC_LIBRARY} gtest gtest_main)

enable_testing()

add_test(NAME google_test COMMAND $<TARGET_FILE:${TEST_BINARY}>)
