#
# Copyright 2026 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)
FetchContent_Declare(
  litert
  GIT_REPOSITORY ${LITERT_GIT_REPOSITORY}
  GIT_TAG ${LITERT_GIT_TAG}
  GIT_SHALLOW    TRUE
)

FetchContent_GetProperties(litert)
if(NOT litert_POPULATED)
  FetchContent_Populate(litert)
endif()

message(STATUS "Downloading TensorFlow repository...")
FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG v2.19.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-src
)
FetchContent_GetProperties(tensorflow)
if(NOT tensorflow_POPULATED)
  FetchContent_Populate(tensorflow)
endif()
set(TENSORFLOW_SOURCE_DIR "${tensorflow_SOURCE_DIR}")

add_subdirectory("${litert_SOURCE_DIR}/tflite"
                 "${litert_BINARY_DIR}")
get_target_property(TFLITE_SOURCE_DIR tensorflow-lite SOURCE_DIR)

if(NOT TFLITE_LIB_LOC OR NOT EXISTS ${TFLITE_LIB_LOC})
  add_library(TensorFlow::tensorflow-lite ALIAS tensorflow-lite)
else()
  add_library(TensorFlow::tensorflow-lite UNKNOWN IMPORTED)
  set_target_properties(TensorFlow::tensorflow-lite PROPERTIES
    IMPORTED_LOCATION ${TFLITE_LIB_LOC}
    INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>
  )
  set_target_properties(tensorflow-lite PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

list(APPEND VX_DELEGATE_DEPENDENCIES TensorFlow::tensorflow-lite)
list(APPEND VX_DELEGATES_SRCS ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc)
list(APPEND VX_CUSTOM_OP_SRCS ${TFLITE_SOURCE_DIR}/delegates/external/external_delegate.cc)
