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

# Optional NVIDIA GPU backend #2 (device: TENSORRT) via TensorRT's own C++
# API directly -- not through ONNX Runtime's TensorRT execution provider.
# Substantially faster than device: CUDA above for the same model (TensorRT
# does FP16 + kernel-level autotuning for the exact GPU it's built on), at
# the cost of needing a one-time engine build per device (see yolov5.cpp
# and JETSON_ORIN.md). Silently absent if TensorRT's dev headers/libs
# (libnvinfer-dev, libnvonnxparsers-dev) and the CUDA runtime aren't
# installed. TensorRT's own headers/libs are all under standard system
# paths on Ubuntu (installed via apt, unlike OpenVINO/onnxruntime's
# hand-placed /opt paths), so plain find_path/find_library without an
# explicit PATHS hint is enough.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_NVINFER_LIB nvinfer)
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h PATHS /usr/local/cuda/include)
find_library(CUDA_CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/x86_64-linux/lib)

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

if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB AND CUDA_INCLUDE_DIR AND CUDA_CUDART_LIB)
    message(STATUS "TensorRT found -- device: TENSORRT (NVIDIA GPU) backend enabled for YOLOV5")
    target_compile_definitions(auto_aim PUBLIC HAVE_TENSORRT)
    target_include_directories(auto_aim PUBLIC ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
    target_link_libraries(auto_aim ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB} ${CUDA_CUDART_LIB})
else()
    message(STATUS "TensorRT not found -- device: TENSORRT will not be available")
endif()