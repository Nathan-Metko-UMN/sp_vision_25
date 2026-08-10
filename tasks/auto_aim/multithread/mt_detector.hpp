#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <memory>

#include "tasks/auto_aim/yolos/trt_engine.hpp"
#endif

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  YOLO yolo_;

  // ---- OpenVINO async path (existing) ----
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};

#ifdef HAVE_TENSORRT
  // ---- TensorRT async path ----
  //
  // YOLOV5's own device: TENSORRT path (yolov5.hpp/.cpp) is single-buffered
  // and fully synchronous (H2D copy, inference, and D2H copy each
  // cudaStreamSynchronize immediately), so the GPU and CPU take turns being
  // idle -- confirmed via tegrastats on real hardware (GPU averaging ~15%
  // utilization). This path instead ring-buffers every per-frame CUDA
  // resource across kTrtRingSize slots, so push() for frame N+1 can start
  // CPU letterbox/preprocess and enqueue its own H2D/infer/D2H while frame
  // N's are still in flight on the stream -- the CPU only ever waits
  // (via cudaEventSynchronize on that *specific* slot's event, never a full
  // cudaStreamSynchronize which would also wait for later, unrelated
  // enqueued frames) when it's about to reuse a slot whose previous
  // occupant's GPU work hasn't finished yet.
  static constexpr int kTrtRingSize = 3;

  bool use_tensorrt_ = false;
  TRTLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  // Single execution context, reused sequentially across slots by
  // re-pointing addresses with setTensorAddress immediately before each
  // enqueueV3 -- confirmed against the installed TensorRT header
  // (NvInferRuntime.h): enqueueV3's contract only requires a slot's memory
  // not be modified/released before that specific enqueue's stream
  // synchronization, which the per-slot cudaEvent below guarantees. Safe
  // here because push() is only ever called from one producer thread (see
  // mt_standard.cpp), so there's no cross-thread race on the context.
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
  cudaStream_t trt_stream_ = nullptr;
  std::string trt_input_name_, trt_output_name_;

  struct TrtSlot
  {
    void * input_device = nullptr;
    void * output_device = nullptr;
    float * input_host = nullptr;   // pinned
    float * output_host = nullptr;  // pinned
    cudaEvent_t d2h_done = nullptr;
    // Bookkeeping only -- the real cross-thread synchronization is always
    // the cudaEvent above (safe to wait on from multiple threads per CUDA's
    // own docs); atomic just removes any ambiguity about this flag itself
    // being read/written from both push_tensorrt() and pop_tensorrt().
    std::atomic<bool> in_flight{false};
  };
  std::array<TrtSlot, kTrtRingSize> trt_slots_;
  int trt_next_slot_ = 0;  // producer-side ring cursor, advanced only in push_tensorrt()

  // {img.clone(), t, slot_index, letterbox_scale} -- FIFO order matches
  // slot-reuse order (the ring is strictly round-robin), so no separate
  // slot->queue-entry lookup is needed.
  tools::ThreadSafeQueue<std::tuple<cv::Mat, std::chrono::steady_clock::time_point, int, double>>
    trt_queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] TRT queue is full!"); }};

  // Persistent letterbox canvas, mirrors YOLOV5::letterbox_canvas_.
  cv::Mat trt_letterbox_canvas_;
  int trt_letterbox_w_ = -1, trt_letterbox_h_ = -1;

  void push_tensorrt(cv::Mat & img, std::chrono::steady_clock::time_point t);
  std::tuple<cv::Mat, std::chrono::steady_clock::time_point, std::list<Armor>> pop_tensorrt();
#endif
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
