cmake_minimum_required(VERSION 3.1...3.14)

project(iris_mesh VERSION 3.11.5 LANGUAGES CXX)

set(CLFML_IRIS_MESH_MAIN_PROJECT OFF)

# Optionally build the examples
option(CLFML_IRIS_MESH_BUILD_EXAMPLE_PROJECTS "Build example projects" ON)

# Optionally enable ROS2 package build (Requires ROS2 jazzy to be installed!)
option(CLFML_ROS2_PACKAGE_BUILD "Build a ROS2 package" OFF)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17)
  set(CLFML_IRIS_MESH_MAIN_PROJECT ON)
else()
  set(CLFML_IRIS_MESH_BUILD_EXAMPLE_PROJECTS OFF)
endif()

if (POLICY CMP0077)
  # Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory.
  cmake_policy(SET CMP0077 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/src/iris_mesh.cpp)
add_library(CLFML::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/src)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (TARGET tensorflow-lite)
  target_link_libraries(${PROJECT_NAME} tensorflow-lite)
else()

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" OR CLFML_IRIS_MESH_COMPILE_TENSORFLOW_FROM_SOURCE)
    include(FetchContent)
    FetchContent_Declare(
        tensorflow_rel_package
        URL https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.1.zip
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/tensorflow
        SOURCE_SUBDIR tensorflow/lite
    )
    FetchContent_MakeAvailable(tensorflow_rel_package)
  else()
    include(FetchContent)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
      FetchContent_Declare(
    tensorflow_compiled_rel_package
    URL https://github.com/CLFML/TensorFlow_Lite_Compiled/releases/download/v2.16.1/tensorflow_linux_generic_aarch64.zip
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/tensorflow
)
    else()
      FetchContent_Declare(
      tensorflow_compiled_rel_package
      URL https://github.com/CLFML/TensorFlow_Lite_Compiled/releases/download/v2.16.1/tensorflow_linux_generic_x86_64.zip
      SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/tensorflow
  )
    endif()
    FetchContent_MakeAvailable(tensorflow_compiled_rel_package)
    if(CLFML_ROS2_PACKAGE_BUILD)
      install(FILES external/tensorflow/lib/libtensorflowlite.so DESTINATION lib/${PROJECT_NAME})
    endif()
  endif()
  target_link_libraries(${PROJECT_NAME} tensorflow-lite)
endif()

set(IRIS_MESH_CPU_MODEL_PATH ${CMAKE_CURRENT_LIST_DIR}/models/CPU/iris_mesh.tflite)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DCLFML_IRIS_MESH_CPU_MODEL_PATH="${IRIS_MESH_CPU_MODEL_PATH}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


if(CLFML_IRIS_MESH_BUILD_EXAMPLE_PROJECTS)
  FetchContent_Declare(
        face_detector.cpp
        GIT_REPOSITORY https://github.com/CLFML/Face_Detector.Cpp.git
        GIT_TAG main
    )
  FetchContent_Declare(
        face_mesh.cpp
        GIT_REPOSITORY https://github.com/CLFML/Face_Mesh.Cpp.git
        GIT_TAG main
  )
  FetchContent_MakeAvailable(face_detector.cpp)
  FetchContent_MakeAvailable(face_mesh.cpp)
  add_executable(iris_mesh_demo ${CMAKE_CURRENT_LIST_DIR}/example/iris_mesh_demo/demo.cpp)
  target_link_libraries(iris_mesh_demo PUBLIC CLFML::${PROJECT_NAME} CLFML::face_detector CLFML::face_mesh)
endif()
