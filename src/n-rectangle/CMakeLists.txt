set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("n-rectangle" ${MAIN_CPP} ${SOURCE})
set_target_properties("n-rectangle" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("n-rectangle" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
