cmake_minimum_required(VERSION 3.12)

project(dnage VERSION 0.0.1)

set(target_name                 dnage)
set(CMAKE_CXX_FLAGS             /std:c++17)

file(GLOB src
    src/*.cpp
    src/**/*.cpp
    src/**/**/*.cpp
    src/**/**/**/*.cpp
    src/**/**/**/**/*.cpp
)

add_library(${target_name} ${src})
target_include_directories(${target_name} PUBLIC include/dnage)