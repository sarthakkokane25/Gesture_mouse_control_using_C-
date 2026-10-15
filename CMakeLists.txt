cmake_minimum_required(VERSION 3.20)

project(Gesturemousecontrol)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Gesturemousecontrol
    Gesturemousecontrol.cpp
    Gesturemousecontrol.h
    PalmDetector.cpp
    PalmDetector.h
    HandPose.cpp
    HandPose.h
)

# OpenCV
target_include_directories(Gesturemousecontrol PRIVATE
    "C:/gesture mouse control libraries/opencv/opencv/build/include"
)

target_link_directories(Gesturemousecontrol PRIVATE
    "C:/gesture mouse control libraries/opencv/opencv/build/x64/vc16/lib"
)

target_link_libraries(Gesturemousecontrol PRIVATE
    opencv_world4120d
)

# ONNX Runtime
target_include_directories(Gesturemousecontrol PRIVATE
    "C:/gesture mouse control libraries/onnxruntime/include"
)

target_link_directories(Gesturemousecontrol PRIVATE
    "C:/gesture mouse control libraries/onnxruntime/lib"
)

target_link_libraries(Gesturemousecontrol PRIVATE
    onnxruntime
)

# Windows mouse control
target_link_libraries(Gesturemousecontrol PRIVATE
    user32
)

# Copy required DLLs beside the executable
add_custom_command(TARGET Gesturemousecontrol POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/gesture mouse control libraries/opencv/opencv/build/x64/vc16/bin/opencv_world4120d.dll"
        "$<TARGET_FILE_DIR:Gesturemousecontrol>"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/gesture mouse control libraries/onnxruntime/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:Gesturemousecontrol>"
)