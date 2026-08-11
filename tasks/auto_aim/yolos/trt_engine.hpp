#ifndef AUTO_AIM__TRT_ENGINE_HPP
#define AUTO_AIM__TRT_ENGINE_HPP

#ifdef HAVE_TENSORRT
#include <NvInfer.h>

#include <cstdint>
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

// Names baked into the fused-NMS ONNX graph by scripts/onnx/fuse_nms.py --
// kept as named constants (not string literals scattered across yolov5.cpp/
// mt_detector.cpp) so there's exactly one place to update if that script's
// output names ever change.
inline constexpr const char * kTrtOutputTensorName = "output/sink_port_0";
inline constexpr const char * kTrtSelectedIndicesTensorName = "selected_indices";

// Must match max_output_boxes_per_class in scripts/onnx/fuse_nms.py exactly
// -- this is the worst-case row count TrtNmsOutputAllocator
// (trt_nms_allocator.hpp) pre-sizes its device buffer to, so that
// reallocateOutputAsync() is expected to fire (i.e. actually allocate) at
// most once ever, not once per frame. No compiler links these two
// constants -- if you change one, change the other.
inline constexpr int64_t kMaxNmsOutputBoxes = 64;

struct TrtIONames
{
  std::string input;
  std::string output;            // kTrtOutputTensorName -- 1x25200x22 float, static shape
  std::string selected_indices;  // kTrtSelectedIndicesTensorName -- [-1,3] int64, data-dependent shape
};

// Discovers I/O tensor names by IOMode + exact name match. Replaces the
// naive "first kINPUT wins input, else output" loop this repo used before
// the engine had two outputs -- that loop silently broke once a second
// output existed (it just kept overwriting a single output-name variable
// with whichever output getIOTensorName() enumerated last). No
// backward-compat fallback for a pre-fusion single-output engine/onnx:
// since assets/yolov5.onnx is permanently rewritten by fuse_nms.py going
// forward, a stale local .engine (gitignored, cached from before that
// change) or an accidentally-reverted .onnx should fail loudly here at
// construction time, not silently degrade.
TrtIONames trt_discover_io_names(nvinfer1::ICudaEngine & engine);

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
#endif  // AUTO_AIM__TRT_ENGINE_HPP
