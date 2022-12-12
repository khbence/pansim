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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_subdirectory(json)

option(USE_GPU "Set to use thrust CUDA GPU" OFF)

find_package(Thrust REQUIRED CONFIG)
set(THRUST_CPP_DIALECT 17)

if(USE_GPU)
  thrust_create_target(Thrust HOST OMP DEVICE CUDA)
else()
  thrust_create_target(Thrust HOST OMP DEVICE OMP)
endif()

add_library(${SRC_LIBRARY})
add_executable(${MAIN_BINARY})

include(src/CMakeLists.txt)

include(FetchContent)
FetchContent_Declare(
    fmtfetch
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 9.1.0
)

FetchContent_MakeAvailable(fmtfetch)

target_link_libraries(${SRC_LIBRARY} PUBLIC project_options project_warnings ${FILE_FORMAT_LIB} Thrust nlohmann_json::nlohmann_json fmt::fmt)
target_include_directories(${SRC_LIBRARY} PUBLIC cxxopts/include)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
