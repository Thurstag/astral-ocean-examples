set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Define executable
add_executable("textured-rectangle" ${MAIN_CPP} ${SOURCE})
set_target_properties("textured-rectangle" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("textured-rectangle" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
