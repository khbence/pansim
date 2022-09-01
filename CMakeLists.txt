cmake_minimum_required(VERSION 3.15)

set(SRC_LIBRARY "pansimLibrary")
set(MAIN_BINARY "pansim")
set(TEST_BINARY "testPansim")

project(coreProject CXX)
include(cmake/StandardProjectSettings.cmake)
# include(cmake/PreventInSourceBuilds.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
# target_compile_features(project_options INTERFACE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE
         "Enable -ftime-trace to generate time tracing .json files on clang"
         OFF)
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
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  #
  # consider breaking this out per project as necessary
  target_precompile_headers(project_options INTERFACE <iostream> <vector>
                            <string>)
endif()

if(ENABLE_FUZZING)
  message(
    "Building Fuzz Tests, using fuzzing sanitizer https://www.llvm.org/docs/LibFuzzer.html"
  )
  add_subdirectory(fuzz_test)
endif()

option(ENABLE_UNITY "Enable Unity builds of projects" OFF)
if(ENABLE_UNITY)
  # Add for any project you want to apply unity builds for
  set_target_properties(intro PROPERTIES UNITY_BUILD ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_subdirectory(json)

add_library(${SRC_LIBRARY})

add_executable(${MAIN_BINARY})

include(src/CMakeLists.txt)

find_package(PkgConfig REQUIRED)
pkg_check_modules (JEMALLOC jemalloc)

pkg_search_module(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})

target_link_libraries(${SRC_LIBRARY} PUBLIC project_options project_warnings ${FILE_FORMAT_LIB} OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json ${JEMALLOC_LIBRARIES})
target_link_libraries(${MAIN_BINARY} PUBLIC project_options project_warnings ${JEMALLOC_LIBRARIES})

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
