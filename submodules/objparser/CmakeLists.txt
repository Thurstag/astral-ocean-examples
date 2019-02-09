cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

# Define variables
set(PROJECT_NAME "astral-ocean-examples")
set(PROJECT_VERSION "0.0.0")

# Define project
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION})

# Define output folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Create library
add_library(objparser objparser.cpp objparser.h)