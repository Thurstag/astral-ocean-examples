cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

# Define variables
set(AO_EX_PROJECT_NAME "astral-ocean-examples")
set(AO_EX_PROJECT_VERSION "0.0.0")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(AO_EX_SUBMODULES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules")

# Define dependencies
set(DEPENDENCIES GLM)
set(SUBMODULES stb glfw astral-ocean)

# Define include directories
set(glfw_INCLUDE_DIR "${AO_EX_SUBMODULES_DIR}/glfw/include")
set(stb_INCLUDE_DIR "${AO_EX_SUBMODULES_DIR}/stb")
set(astral-ocean_INCLUDE_DIR "${AO_EX_SUBMODULES_DIR}/astral-ocean/include")

# Disable tests for astral-ocean
set(AO_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Disable GLFW unwanted builds
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Cmake settings
set_property(GLOBAL PROPERTY USE_FOLDERS TRUE)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Define output folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Force c++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project name
project(${AO_EX_PROJECT_NAME} VERSION ${AO_EX_PROJECT_VERSION})

# Include headers
include_directories("${AO_EX_SUBMODULES_DIR}/astral-ocean/submodules/")
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# Try to find libraries
if(NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	foreach(DEPENDENCY ${DEPENDENCIES})
		find_package(${DEPENDENCY} REQUIRED)
	endforeach()
endif()

# Check all libs
foreach(DEPENDENCY ${DEPENDENCIES})
	if(NOT ${${DEPENDENCY}_FOUND})
		message(FATAL_ERROR "Could not find ${DEPENDENCY} library!")
	else()
		set(LIB_PATH "${${DEPENDENCY}_INCLUDE_DIR}")
		if(LIB_PATH STREQUAL "")
			message("${DEPENDENCY} library found")
		else()
			message("${DEPENDENCY} library found: '${LIB_PATH}', auto-include it")
			include_directories(${LIB_PATH})
		endif()
	endif()
endforeach()

# Add submodules
foreach(SUBMODULE ${SUBMODULES})
	message("Include ${SUBMODULE} submodule")
	
	if(EXISTS "${AO_EX_SUBMODULES_DIR}/${SUBMODULE}/CMakeLists.txt")
		add_subdirectory("${AO_EX_SUBMODULES_DIR}/${SUBMODULE}")
	endif()
	if(EXISTS ${${SUBMODULE}_INCLUDE_DIR})
		include_directories(${${SUBMODULE}_INCLUDE_DIR})
	else()
		message(WARNING "Unable to include ${SUBMODULE} headers (path: ${${SUBMODULE}_INCLUDE_DIR})")
	endif()
endforeach()

# Include sub-projects
add_subdirectory("src")
