set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("rectangle" ${MAIN_CPP} ${SOURCE})
set_target_properties("rectangle" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("rectangle" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
