#    Copyright (c) 2021 Vivante Corporation
#
#    Permission is hereby granted, free of charge, to any person obtaining a
#    copy of this software and associated documentation files (the "Software"),
#    to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#    and/or sell copies of the Software, and to permit persons to whom the
#    Software is furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#    DEALINGS IN THE SOFTWARE.
#
cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

project(tflite_vx_delegate)

OPTION(ENABLE_NBG_SUPPORT "enable customized nbg op in tflite" ON)
OPTION(VX_DELEGATE_USE_TFLITE_LIB_FROM_SDK "Use prebuilt TFLite library instead of building the TFLite from source" OFF)

set(CMAKE_CXX_STANDARD 14)
if(ANDROID_TOOLCHAIN)
# bypass warning as error since tensorflow lite can not pass with android ndk r22b
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-c++11-narrowing")
endif()

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

find_package(tensorflow REQUIRED)
find_package(tim-vx REQUIRED)

list(APPEND VX_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/delegate_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/op_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vx_delegate_adaptor.cc
)

add_library(vx_delegate SHARED ${VX_DELEGATES_SRCS})

list(APPEND VX_CUSTOM_OP_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/vsi_npu_custom_op.cc
)
if(ANDROID_TOOLCHAIN)
  list(APPEND VX_DELEGATE_DEPENDENCIES log)
endif()

if (VX_DELEGATE_USE_TFLITE_LIB_FROM_SDK)
  list(APPEND VX_DELEGATE_DEPENDENCIES "-ltensorflow-lite")
else()
  list(APPEND VX_DELEGATE_DEPENDENCIES tensorflow-lite)
endif()

target_link_libraries(vx_delegate ${VX_DELEGATE_DEPENDENCIES})

add_library(vx_custom_op STATIC ${VX_CUSTOM_OP_SRCS})
target_include_directories(vx_custom_op PUBLIC ${PROJECT_SOURCE_DIR})
add_dependencies(vx_custom_op vx_delegate)

if(VX_DELEGATE_USE_TFLITE_LIB_FROM_SDK)
  target_link_libraries(vx_custom_op "-ltensorflow-lite")

  # Add the tensorflow-lite target include directories as it is not linked neither to vx-delegate nor to vx_custom_op.
  # A prebuilt tensoflow-lite library from SDK is linked instead.
  target_include_directories(vx_delegate PUBLIC $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(vx_custom_op PUBLIC $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  target_link_libraries(vx_custom_op tensorflow-lite)
  set_target_properties(benchmark_model PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)
  set_target_properties(label_image PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)
  add_subdirectory(examples/minimal)
endif()

