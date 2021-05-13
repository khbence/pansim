include(FetchContent)

find_package(GTest QUIET)

if(NOT GTEST_FOUND)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0)

    FetchContent_MakeAvailable(googletest)
endif()

# setup rapidcheck

find_package(rapidcheck QUIET)

if(NOT rapidcheck_FOUND)
    FetchContent_Declare(rcheck GIT_REPOSITORY https://github.com/emil-e/rapidcheck)

    FetchContent_MakeAvailable(rcheck)
endif()

add_executable(${TEST_BINARY})
target_sources(${TEST_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

function(target_link_libraries_system target)
    set(libs ${ARGN})
    foreach(lib ${libs})
        get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
        target_link_libraries(${target} ${scope} ${lib})
    endforeach(lib)
endfunction(target_link_libraries_system)

# set(RC_ENABLE_TESTS ON CACHE INTERNAL "To load the GTEST variant of rapidcheck")

target_link_libraries(${TEST_BINARY} gtest gtest_main)

target_link_libraries_system(${TEST_BINARY} rapidcheck rapidcheck_gtest)

enable_testing()

add_test(NAME google_test COMMAND $<TARGET_FILE:${TEST_BINARY}>)

# set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/subprojects) include(ExternalProject) set(STAGED_INSTALL_PREFIX
# ${CMAKE_BINARY_DIR}/stage) message(STATUS "${PFOJECT_NAME} staged install: ${STAGED_INSTALL_PREFIX}")
