# setting a common place to put all executable files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_executable(device_query device_query.cpp)
target_link_libraries(device_query RenderAndCompare)

add_executable(visualize_dataset visualize_dataset.cpp)
target_link_libraries(visualize_dataset RenderAndCompare)

add_executable(check_cpp_data_layer check_cpp_data_layer.cpp)
target_link_libraries(check_cpp_data_layer RenderAndCompare)

add_executable(check_network_model check_network_model.cpp)
target_link_libraries(check_network_model RenderAndCompare)
