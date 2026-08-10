#ifndef AUTO_AIM__TRT_ENGINE_HPP
#define AUTO_AIM__TRT_ENGINE_HPP

#ifdef HAVE_TENSORRT
#include <NvInfer.h>

#include <memory>
#include <string>

namespace auto_aim
{

// Shared by YOLOV5's single-threaded device: TENSORRT path (yolov5.hpp/.cpp)
// and MultiThreadDetector's async TENSORRT path (mt_detector.hpp/.cpp) --
// moved here (no logic changes) so this ~70 lines of TensorRT bring-up
// (builder/parser/config/FP16 flag/serialize-and-cache-to-disk) exists in
// exactly one place instead of drifting between two copies. Must be kept
// alive (as a member, not a stack local) for the lifetime of any
// runtime/engine/context built from it -- TensorRT keeps a reference to the
// ILogger internally.
class TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

// Loads a cached .engine file at engine_path if present (fast path), else
// builds one from onnx_path (FP16 if the platform supports it) and caches it
// to engine_path -- a one-time, slow (can take several minutes) operation
// per device. Fills runtime/engine; both must be empty unique_ptrs on entry.
void trt_build_or_load_engine(
  TRTLogger & logger, std::unique_ptr<nvinfer1::IRuntime> & runtime,
  std::unique_ptr<nvinfer1::ICudaEngine> & engine, const std::string & onnx_path,
  const std::string & engine_path);

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
#endif  // AUTO_AIM__TRT_ENGINE_HPP
