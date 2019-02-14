set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("mipmap" ${MAIN_CPP} ${SOURCE})
set_target_properties("mipmap" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("mipmap" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY};objparser;meshoptimizer")
