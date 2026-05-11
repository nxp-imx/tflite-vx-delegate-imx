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

project(tflite_vx_delegate)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TFLITE_ENABLE_MULTI_DEVICE       "Enable multi devices support"          OFF)
option(TFLITE_ENABLE_OPTIMIZE           "Enable optimize tiny yolov4"           OFF)
option(TFLITE_ENABLE_NODE_TRACE         "Enable node trace"                     OFF)
option(BUILD_FOR_LITERT                 "Configure delegate build for LiteRT"   OFF)
OPTION(ENABLE_NBG_SUPPORT               "enable customized nbg op in tflite"    ON)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

if(${BUILD_FOR_LITERT})
  set(LITERT_GIT_REPOSITORY "https://github.com/nxp-imx/litert-imx" CACHE STRING "")
  set(LITERT_GIT_TAG "lf-6.18.20_2.0.0" CACHE STRING "")
  set(TFLITE_SUB_PATH "tflite")
  set(VX_DELEGATE_TARGET "litert_vx_delegate")
  find_package(LiteRT REQUIRED)
else()
  set(TFLITE_GIT_REPOSITORY "https://github.com/nxp-imx/tensorflow-imx" CACHE STRING "")
  set(TFLITE_GIT_TAG "lf-6.18.20_2.0.0" CACHE STRING "")
  set(TFLITE_SUB_PATH "tensorflow/lite")
  set(VX_DELEGATE_TARGET "vx_delegate")
  find_package(tensorflow REQUIRED)
endif()

if(TFLITE_ENABLE_OPTIMIZE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVSI_FEAT_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS -DENABLE_TENSOR_CACHE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVSI_FEAT_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS -DENABLE_TENSOR_CACHE")
endif()
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(ANDROID_TOOLCHAIN)
# bypass warning as error since tensorflow lite can not pass with android ndk r22b
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-c++11-narrowing")
endif()

find_package(tim-vx REQUIRED)

list(APPEND VX_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/delegate_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/op_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vx_delegate_adaptor.cc
)

if(TFLITE_ENABLE_MULTI_DEVICE)
  ADD_DEFINITIONS(-DMULTI_DEVICE_FEATURE_MODE)
endif()

if(TFLITE_ENABLE_NODE_TRACE)
  ADD_DEFINITIONS(-DNODE_TRACE_DB_MODE)
endif()

add_library(${VX_DELEGATE_TARGET} SHARED ${VX_DELEGATES_SRCS})
include_directories(
    $<TARGET_PROPERTY:TensorFlow::tensorflow-lite,INCLUDE_DIRECTORIES>
)

list(APPEND VX_CUSTOM_OP_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/vsi_npu_custom_op.cc
)
if(ANDROID_TOOLCHAIN)
  list(APPEND VX_DELEGATE_DEPENDENCIES log)
endif()

target_link_libraries(${VX_DELEGATE_TARGET} ${VX_DELEGATE_DEPENDENCIES})
if((NOT DEFINED TIM_VX_INSTALL) AND (NOT DEFINED ANDROID_ROOT))
target_link_libraries(${VX_DELEGATE_TARGET} -Wl,--whole-archive tim-vx)
endif()
add_library(vx_custom_op STATIC ${VX_CUSTOM_OP_SRCS})
target_include_directories(vx_custom_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(vx_custom_op TensorFlow::tensorflow-lite)
add_dependencies(vx_custom_op ${VX_DELEGATE_TARGET})

set_target_properties(benchmark_model PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)
set_target_properties(label_image PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)

add_subdirectory(examples/minimal)
if(TFLITE_ENABLE_MULTI_DEVICE)
  add_subdirectory(examples/multi_device)
endif()
