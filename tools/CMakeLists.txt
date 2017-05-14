# setting a common place to put all executable files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(device_query device_query.cpp)
target_link_libraries(device_query ${Caffe_LIBRARIES})