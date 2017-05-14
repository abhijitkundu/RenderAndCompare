cmake_minimum_required(VERSION 3.0.0)
project(RenderAndCompare)

########################### Misc. Configs ##############################
message(STATUS "=============================================================")
message(STATUS "============= Configuring RenderAndCompare ==================")

# Add custom cmake files folder
set (PROJECT_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
set (CMAKE_MODULE_PATH "${PROJECT_CMAKE_DIR}" "${CMAKE_MODULE_PATH}")

# Add custom Compile Settings and flags
include(CompileSettings)

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message(STATUS "=============================================================")
message(STATUS "================= Look for Dependencies =====================")

# Find Eigen dependencies
find_package(Eigen REQUIRED)

# Find CuteGL Dependencies
find_package(CuteGL REQUIRED)

# Find Caffe Dependencies
find_package(Caffe REQUIRED)

# Find JSON Dependencies
option (WITH_JSON "Use JSON" ON)
if(WITH_JSON)
  find_package(nlohmann_json)
  if(NOT nlohmann_json_FOUND)
    set(WITH_JSON OFF)
    message(STATUS "nlohmann_json Not Found") 
  endif()
endif()

# Add OpenCV dependencies
find_package(OpenCV REQUIRED)
if (${OpenCV_VERSION_MAJOR} EQUAL 2)
  set(REQUIRED_OpenCV_LIBS opencv_core opencv_highgui)
else()
  set(REQUIRED_OpenCV_LIBS opencv_core opencv_highgui opencv_imgcodecs)
endif()

# Add Boost dependencies
find_package(Boost
  REQUIRED
  COMPONENTS
    filesystem
    system
    program_options
    serialization
    timer
  )

# Find HDF5 dependencies
find_package(HDF5 1.8.12
  REQUIRED
  COMPONENTS
    CXX
  )

message(STATUS "=============================================================")
message(STATUS "========== Adding RenderAndCompare Library ==================")

add_subdirectory(RenderAndCompare)

message(STATUS "=============================================================")
message(STATUS "====================== Adding Tools =========================")

add_subdirectory(tools)


message(STATUS "=============================================================")
message(STATUS "================  Configuration Summary  ====================")
message(STATUS "Project Name:       ${PROJECT_NAME}")
message(STATUS "C++ Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:       C++ ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "Build type Flags:   ${CMAKE_BUILD_TYPE_FLAGS}")
message(STATUS "C++ compile flags:  ${CMAKE_CXX_FLAGS}")
message(STATUS "Install Path:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Eigen Version:      ${EIGEN_VERSION}")
message(STATUS "Caffe:              CUDA: ${Caffe_HAVE_CUDA}  CUDNN: ${Caffe_HAVE_CUDNN}  ${Caffe_DEFINITIONS}")
message(STATUS "OpenCV Version:     ${OpenCV_VERSION}")
message(STATUS "HDF5 Version:       ${HDF5_VERSION}")
message(STATUS "Boost Version:      ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
if (WITH_JSON)
message(STATUS "Json:               YES (${JSON_INCLUDE_DIR})")
else()
message(STATUS "Json:               NO")
endif()
message(STATUS "=============================================================")
