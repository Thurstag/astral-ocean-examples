set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("texture-array" ${MAIN_CPP} ${SOURCE})
set_target_properties("texture-array" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("texture-array" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
