# Iris_Mesh.Cpp
Face mesh generator using the Iris Mesh Mediapipe model with a CPU delegate written in C++ 

- Plain C/C++ implementation with minimal dependencies (Tensorflow Lite + OpenCV)
- Google MediaPipe models without the MediaPipe framework
- Runs on ARM

## API Features
This library offers support for:
- 3D Iris Landmarking (5x3D points)
- Iris contour and eye-brow landmarks (71 points)

### 3D Iris Landmarking
This is some example code for Iris landmarking:

```cpp
/* Create instance of Iris landmarker */
CLFML::IrisMesh::IrisMesh iris_det;

/* Load image into model and do inference! */
iris_det.load_image(eye_roi_cropped_frame);

 /* Get the iris mesh keypoints from the model inference output */
std::array<cv::Point3f, 5> iris_mesh_keypoints = iris_det.get_iris_mesh_points();
```

## Example code
For a full example showcasing both these API functions see the example code in [example/iris_mesh_demo/demo.cpp](example/iris_mesh_demo/demo.cpp).

## Building with CMake
Before using this library you will need the following packages installed:
- OpenCV
- Working C++ compiler (GCC, Clang, MSVC (2017 or Higher))
- CMake
- Ninja (**Optional**, but preferred)

### Running the examples (CPU)
1. Clone this repo
2. Run:
```bash
cmake . -B build -G Ninja
```
3. Let CMake generate and run:
```bash
cd build && ninja
```
4. After building you can run (linux & mac):
```bash
./iris_mesh_demo
```
or (if using windows)
```bat
iris_mesh_demo.exe
```

### Using it in your project as library
Add this to your top-level CMakeLists file:
```cmake
include(FetchContent)

FetchContent_Declare(
    iris_mesh.cpp
    GIT_REPOSITORY https://github.com/CLFML/Iris_Mesh.Cpp
    GIT_TAG main
    # Put the Iris_Mesh lib into lib/ folder
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/Iris_Mesh.Cpp
)
FetchContent_MakeAvailable(iris_mesh.cpp)
...
target_link_libraries(YOUR_EXECUTABLE CLFML::iris_mesh)
```
Or manually clone this repo and add the library to your project using:
```cmake
add_subdirectory(Iris_Mesh.Cpp)
...
target_link_libraries(YOUR_EXECUTABLE CLFML::iris_mesh)
```
## Aditional documentation
See our [wiki](https://clfml.github.io/Iris_Mesh.Cpp/)...

## Todo
- Add language bindings for Python, C# and Java
- Add support for MakeFiles and Bazel
- Add Unit-tests 
- Add ROS2 package support

## License
This work is licensed under the Apache 2.0 license. 

The [iris_mesh model](https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/preview) is also licensed under the Apache 2.0 license.