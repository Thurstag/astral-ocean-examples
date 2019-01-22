set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("triangle" ${MAIN_CPP} ${SOURCE})
set_target_properties("triangle" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("triangle" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
