set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("depth-rectangle" ${MAIN_CPP} ${SOURCE})
set_target_properties("depth-rectangle" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("depth-rectangle" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
