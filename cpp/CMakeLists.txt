cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_BUILD_TYPE "Debug")
project(cpp_tools)
# set(PCL_DIR "/usr/include/pcl-1.12")
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
add_executable(ransac_fit ransac_fit.cpp)
target_link_libraries(ransac_fit
    ${PCL_LIBRARIES}
    )
# add_executable(viewer view.cpp)
# target_link_libraries(viewer
#     ${PCL_LIBRARIES}
#     )
# add_executable(pcl_bplines curve.cpp)
# target_link_libraries(pcl_bplines
#      ${PCL_LIBRARIES}
#      )