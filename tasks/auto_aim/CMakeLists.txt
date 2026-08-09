cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Optional NVIDIA GPU backend (device: CUDA in the yaml configs) via ONNX
# Runtime's CUDA execution provider. OpenVINO's own GPU plugin is Intel-only,
# so this is a second, independent inference path for YOLOV5 -- see
# JETSON_ORIN.md for how the .onnx models were derived from the .xml/.bin
# OpenVINO IR already used by the OpenVINO path. Silently absent (falls back
# to OpenVINO-only) if onnxruntime isn't installed at /opt/onnxruntime.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /opt/onnxruntime/lib)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(auto_aim io openvino::runtime tinympcstatic)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "onnxruntime found -- device: CUDA (NVIDIA GPU) backend enabled for YOLOV5")
    target_compile_definitions(auto_aim PUBLIC HAVE_ONNXRUNTIME)
    target_include_directories(auto_aim PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(auto_aim ${ONNXRUNTIME_LIB})
else()
    message(STATUS "onnxruntime not found at /opt/onnxruntime -- device: CUDA will not be available (OpenVINO CPU/GPU only)")
endif()