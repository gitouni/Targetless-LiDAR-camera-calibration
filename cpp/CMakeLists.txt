cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE "Release")
project(cpp_tools)
# set(PCL_DIR "/usr/include/pcl-1.12") # set your PCL Path manually if you met PCL_NOT_FOUND ERROR
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(include
    ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcl_preprocess preprocess.cpp)
target_link_libraries(pcl_preprocess
     ${PCL_LIBRARIES}
     )
add_executable(viewer view.cpp)
target_link_libraries(viewer
     ${PCL_LIBRARIES}
     )
