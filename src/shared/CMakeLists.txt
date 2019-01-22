file(GLOB_RECURSE SHARED_SRC "*.cpp")
file(GLOB_RECURSE SHARED_HEADERS "*.h" "*.hpp")

# Define library
create_library("ao-shared" ${CMAKE_CURRENT_SOURCE_DIR} "ao-vulkan")

# Re-create sub-directories
create_sub_directories(SHARED_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} "Header Files")
create_sub_directories(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR} "Source Files")
