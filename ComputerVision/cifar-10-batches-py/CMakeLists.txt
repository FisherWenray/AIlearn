# CIFAR-10 ONNX Runtime C++ CMake Configuration
#
# Install dependencies first:
#   macOS:   brew install opencv onnxruntime
#   Linux:   sudo apt install libopencv-dev onnxruntime-dev
#
# Compile:
#   mkdir build && cd build
#   cmake ..
#   make
#
# Run:
#   ./cifar_infer_onnx ../models/cifar10_resnet18.onnx ../data/test.png

cmake_minimum_required(VERSION 3.10)
project(cifar_infer_onnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

# Find ONNX Runtime
find_package(ONNX REQUIRED)
if(NOT ONNX_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found!")
endif()
message(STATUS "ONNX Runtime found")

# Executable
add_executable(cifar_infer_onnx src/main_onnx.cpp)

# Link libraries
target_link_libraries(cifar_infer_onnx
    ${OpenCV_LIBS}
    ${ONNX_LIBRARIES}
)

# Include directories
target_include_directories(cifar_infer_onnx PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNX_INCLUDE_DIRS}
)

# macOS specific
if(APPLE)
    target_link_libraries(cifar_infer_onnx c++)
endif()

# Copy model files to build directory
set(MODEL_FILES
    "${CMAKE_SOURCE_DIR}/models/cifar10_resnet18.onnx"
)

foreach(f ${MODEL_FILES})
    if(EXISTS ${f})
        get_filename_component(name ${f} NAME)
        add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}
            COMMAND ${CMAKE_COMMAND} -E copy ${f} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}
            COMMENT "Copying ${name}"
        )
        add_dependencies(cifar_infer_onnx ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})
    endif()
endforeach()
