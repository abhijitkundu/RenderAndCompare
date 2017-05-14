######################### COMPILE SETTINGS ################################
message(STATUS "===============================================================")
message(STATUS "============ Configuring CompileSettings  =====================")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE
    STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
 set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif(NOT CMAKE_BUILD_TYPE)


if(UNIX)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")

  option (USE_PEDANTIC_FLAGS "Use Pedantic Flags" ON)
  if(USE_PEDANTIC_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
  endif()
  
  option (USE_DEBUG_SYMBOLS "Use Debug Symbols" OFF)
  if(USE_DEBUG_SYMBOLS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -g")
  endif()
endif()

## Enable C++ standard (falls to next avialble)
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED 11)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_BUILD_TYPE_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_BUILD_TYPE_FLAGS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_BUILD_TYPE_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
endif()

option (USE_OpenMP "Use OpenMP" ON)
if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    set(USE_OpenMP OFF)
  endif()
endif()