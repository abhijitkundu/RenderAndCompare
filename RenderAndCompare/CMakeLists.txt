file(GLOB _HDRS *.h *.hpp)
file(GLOB _SRCS *.cpp *.cu)

########################### Add Target ################################

cuda_include_directories("${EIGEN_INCLUDE_DIRS}")
cuda_add_library(RenderAndCompare ${_SRCS} ${_HDRS})
target_link_libraries(RenderAndCompare
  CuteGL::Surface
  ${Caffe_LIBRARIES}
  ${Boost_LIBRARIES}
  ${HDF5_LIBRARIES}  
  ${OpenCV_LIBS}
  nlohmann_json
  )

target_include_directories(RenderAndCompare
  PUBLIC
    ${EIGEN_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
 )

target_include_directories(RenderAndCompare
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_SOURCE_DIR}
 )