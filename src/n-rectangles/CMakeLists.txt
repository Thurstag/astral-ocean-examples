set(MAIN_CPP main.cpp)

# Include *.cpp & *.h
file(GLOB_RECURSE SOURCE *.cpp *.h *.hpp)

# Add resources
if(WIN32)
    set(SOURCE "${SOURCE};../shared/resources/windows/shared.rc")
endif()

# Define executable
add_executable("n-rectangles" ${MAIN_CPP} ${SOURCE})
set_target_properties("n-rectangles" PROPERTIES FOLDER "AO-EXAMPLES")

# Import libraries
target_link_libraries("n-rectangles" "ao-vulkan;ao-shared;glfw;${GLM_LIBRARY}")
