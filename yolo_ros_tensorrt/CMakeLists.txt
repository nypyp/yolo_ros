cmake_minimum_required(VERSION 3.8)
project(yolo_ros_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# TensorRT
find_package(TensorRT REQUIRED)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)

list(APPEND INCLUDE_DIRS
            ${CUDA_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${TensorRT_INCLUDE_DIRS}
            include
)

list(APPEND ALL_LIBS
            ${CUDA_LIBRARIES}
            ${CUDA_LIB_DIR}
            ${OpenCV_LIBRARIES}
            ${TensorRT_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

add_executable(yolo_tensorrt_node 
        src/yolo_tensorrt_node.cpp
)

ament_target_dependencies(yolo_tensorrt_node
        rclcpp
        sensor_msgs
        std_msgs
        cv_bridge
        yolo_msgs
        rclcpp_lifecycle
        lifecycle_msgs
)

link_directories(${ALL_LIBS})
target_link_libraries(yolo_tensorrt_node nvinfer nvinfer_plugin  ${CUDA_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS
        yolo_tensorrt_node
        DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY engines/
        DESTINATION share/${PROJECT_NAME}/engines
)

