option(ENABLE_GTEST_TESTING "Enable GoogleTest/RapidCheck-based C++ tests" OFF)

add_test(
    NAME deterministic_cpu_reference
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_determinism_reference.sh
        $<TARGET_FILE:${MAIN_BINARY}>
        ${CMAKE_CURRENT_SOURCE_DIR}/reference
)

if(CUDAToolkit_FOUND AND USE_GPU AND NOT GPU_ATOMICS)
    add_test(
        NAME deterministic_gpu_smoke
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/run_determinism_smoke.sh
            $<TARGET_FILE:${MAIN_BINARY}>
    )
endif()

if(NOT ENABLE_GTEST_TESTING)
    return()
endif()

include(FetchContent)

find_package(GTest QUIET)

if(NOT GTEST_FOUND)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0)

    FetchContent_MakeAvailable(googletest)
endif()

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

target_link_libraries(${TEST_BINARY} gtest gtest_main)
target_link_libraries_system(${TEST_BINARY} rapidcheck rapidcheck_gtest)

add_test(NAME google_test COMMAND $<TARGET_FILE:${TEST_BINARY}>)
