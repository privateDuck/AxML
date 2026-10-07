cmake_minimum_required(VERSION 3.20)

project(AxML VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(AxML INTERFACE)

target_include_directories(AxML
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
)

# Development translation unit
add_executable(AxML_dev
        test/main.cpp
)

target_link_libraries(AxML_dev
        PRIVATE
        AxML
)

# Tests
add_subdirectory(test)
add_subdirectory(cuda)