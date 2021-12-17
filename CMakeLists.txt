cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(pansimProject CXX)
include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAIN_BINARY "panSim")
set(TEST_BINARY "testPanSim")
# set(CUB_DIR, "${CMAKE_CURRENT_LIST_DIR}/cub")

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
# target_compile_features(project_options INTERFACE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
    if(ENABLE_BUILD_WITH_TIME_TRACE)
        add_compile_definitions(project_options INTERFACE -ftime-trace)
    endif()
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" OFF)

# Git version information (build time)
set(GIT_VERSION "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "Current program version is ${GIT_VERSION}")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/in/version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/out/version.h
               @ONLY)

# Preset simulation template types
set(MovementPolicy
    "RealMovement"
    CACHE STRING "Type of the movement policy.")
set_property(CACHE MovementPolicy PROPERTY STRINGS RealMovement DummyMovement NoMovement)
set(TestingPolicy
    "DetailedTesting"
    CACHE STRING "Type of the testing policy.")
set_property(CACHE TestingPolicy PROPERTY STRINGS NoTesting DetailedTesting)

set(ClosurePolicy
    "RuleClosure"
    CACHE STRING "Type of the closure policy.")
set_property(CACHE ClosurePolicy PROPERTY STRINGS NoClosure RuleClosure)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/in/configTypes.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/out/configTypes.h @ONLY)

option(USE_GPU "Using CUDA-GPU if available" ON)

if(USE_GPU)
    find_package(CUDAToolkit)
endif(USE_GPU)

if(CUDAToolkit_FOUND AND USE_GPU)
    include(cmake/mycudautils.cmake)
    set(ARCHITECTURE
        "AUTO"
        CACHE STRING "Your target CUDA architecture")
    select_cuda_architectures("${ARCHITECTURE}" ARCH_LIST)
    message(STATUS "CUDA_ARCH: ${ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES
        "${ARCH_LIST}"
        CACHE STRING "CUDA architectures" FORCE)
    # include(FindCUDA/select_compute_arch)
    # CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    # string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    # string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    # string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    # SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

    add_executable(${MAIN_BINARY} src/main.cpp)

    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(
            old_flags
            TARGET ${EXISTING_TARGET}
            PROPERTY INTERFACE_COMPILE_OPTIONS)
        if(NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            set_property(
                TARGET ${EXISTING_TARGET}
                PROPERTY
                    INTERFACE_COMPILE_OPTIONS
                    "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
        endif()
    endfunction()

    message(STATUS "Using CUDA-GPU")
    enable_language(CUDA)
    cuda_convert_flags(project_warnings)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
    set_target_properties(${MAIN_BINARY} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${MAIN_BINARY} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${MAIN_BINARY} PUBLIC CUDA::cudart)
else()
    add_executable(${MAIN_BINARY} src/main.cpp)
    if(USE_GPU AND NOT CUDA_FOUND)
        message(INFO "CUDA does not exists")
    endif()
    message(STATUS "Using CPU only")
endif()

# find_package(Thrust REQUIRED CONFIG)
find_package(OpenMP)

#if(CUDAToolkit_FOUND AND USE_GPU)
#    thrust_create_target(Thrust HOST CPP DEVICE CUDA)
#else()
#    thrust_create_target(Thrust HOST CPP DEVICE OMP)
#endif()

add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_include_directories(${MAIN_BINARY} PUBLIC rapidjson/include cxxopts/include cmake/config/out)
target_link_libraries(${MAIN_BINARY} PUBLIC project_options ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX CUDA::curand)
# target_include_directories(${MAIN_BINARY} SYSTEM PRIVATE $<TARGET_PROPERTY:Thrust,INTERFACE_INCLUDE_DIRECTORIES>)

include(src/CMakeLists.txt)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
