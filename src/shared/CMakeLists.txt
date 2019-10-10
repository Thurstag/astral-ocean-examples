file(GLOB_RECURSE SHARED_SRC "*.cpp")
file(GLOB_RECURSE SHARED_HEADERS "*.h" "*.hpp")

# Define library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.1")
        create_library("ao-shared" ${CMAKE_CURRENT_SOURCE_DIR} "ao-vulkan;tbb")
    else()
        create_library("ao-shared" ${CMAKE_CURRENT_SOURCE_DIR} "ao-vulkan;tbb")
    endif()
else()
    create_library("ao-shared" ${CMAKE_CURRENT_SOURCE_DIR} "ao-vulkan")
endif()

# Re-create sub-directories
create_sub_directories(SHARED_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} "Header Files")
create_sub_directories(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR} "Source Files")
