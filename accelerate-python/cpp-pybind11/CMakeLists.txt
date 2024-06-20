cmake_minimum_required(VERSION 3.12)
project(example)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find packages
set(Python3_FIND_VIRTUALENV ONLY)
find_package (Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)

# Create the Python module
pybind11_add_module(lib lib.cpp)
target_link_libraries(lib PRIVATE spdlog::spdlog)
target_compile_options(lib PRIVATE -O3)
