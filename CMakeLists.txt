cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(pansim CXX)
include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_LIBRARY "panSimLibrary")
set(MAIN_BINARY "panSim")
set(TEST_BINARY "testPanSim")
#set(CUB_DIR, "${CMAKE_CURRENT_LIST_DIR}/cub")

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
option(ENABLE_FUZZING "Enable Fuzzing Builds" OFF)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if(ENABLE_PCH)
    # This sets a global PCH parameter, each project will build its own PCH, which is a good idea if any #define's
    # change
    #
    # consider breaking this out per project as necessary
    target_precompile_headers(project_options INTERFACE <iostream>)
endif()

if(ENABLE_FUZZING)
    message("Building Fuzz Tests, using fuzzing sanitizer https://www.llvm.org/docs/LibFuzzer.html")
    add_subdirectory(fuzz_test)
endif()

option(ENABLE_UNITY "Enable Unity builds of projects" OFF)
if(ENABLE_UNITY)
    # Add for any project you want to apply unity builds for
    set_target_properties(intro PROPERTIES UNITY_BUILD ON)
endif()

# find_package(OpenMP)
# if(OPENMP_FOUND)
#   target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
#   set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP)
# endif()

option(USE_GPU, "Using CUDA-GPU if available" ON)

if(USE_GPU)
    find_package(CUDAToolkit)    
endif(USE_GPU)

#add_subdirectory(cub)
#add_subdirectory(thrust)

find_package(Thrust REQUIRED PATHS "thrust/thrust/cmake/")
find_package(CUB REQUIRED PATHS "cub/cub/cmake/")

add_library(${SRC_LIBRARY})

if(CUDA_FOUND AND USE_GPU)
    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
        if(NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
                "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
                )
        endif()
    endfunction()

    message(STATUS "Using CUDA-GPU")
    enable_language(CUDA)
    thrust_create_target(Thrust HOST CPP DEVICE CUDA)
    CUDA_CONVERT_FLAGS(project_warnings)
    target_link_libraries(${SRC_LIBRARY} PUBLIC CUDA::cudart)
else()
    if(USE_GPU AND NOT CUDA_FOUND)
        message(WARNING "CUDA does not exists")
    endif()
    message(STATUS "Using CPU only")
    thrust_create_target(Thrust HOST CPP DEVICE OMP)
endif()

function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    message(${lib_include_dirs})
    target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
    target_link_libraries(${target} ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)

set_target_properties(${SRC_LIBRARY} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${SRC_LIBRARY} PUBLIC project_options Thrust ${CMAKE_THREAD_LIBS_INIT})
#target_include_directories(${SRC_LIBRARY} SYSTEM PRIVATE $<TARGET_PROPERTY:Thrust,INTERFACE_INCLUDE_DIRECTORIES>)

include(src/CMakeLists.txt)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
