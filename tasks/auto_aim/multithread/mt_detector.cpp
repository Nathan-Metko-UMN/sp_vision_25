#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

#ifdef HAVE_TENSORRT
  if (device_ == "TENSORRT") {
    use_tensorrt_ = true;

    auto dot = model_path.find_last_of('.');
    auto onnx_path = (dot == std::string::npos ? model_path : model_path.substr(0, dot)) + ".onnx";
    auto engine_path = (dot == std::string::npos ? model_path : model_path.substr(0, dot)) + ".engine";

    trt_build_or_load_engine(trt_logger_, trt_runtime_, trt_engine_, onnx_path, engine_path);

    trt_context_.reset(trt_engine_->createExecutionContext());
    if (!trt_context_)
      throw std::runtime_error("MultiThreadDetector: failed to create TensorRT execution context");

    if (cudaStreamCreate(&trt_stream_) != cudaSuccess)
      throw std::runtime_error("MultiThreadDetector: cudaStreamCreate failed");

    // Fixed shapes: input 1x3x640x640, output 1x25200x22 -- same shapes
    // YOLOV5's own TensorRT path hardcodes (this .onnx model has no dynamic
    // axes).
    for (int i = 0; i < trt_engine_->getNbIOTensors(); i++) {
      std::string name = trt_engine_->getIOTensorName(i);
      if (trt_engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
        trt_input_name_ = name;
      } else {
        trt_output_name_ = name;
      }
    }
    if (trt_input_name_.empty() || trt_output_name_.empty())
      throw std::runtime_error("MultiThreadDetector: TensorRT engine has unexpected I/O tensor layout");

    for (auto & slot : trt_slots_) {
      if (cudaMalloc(&slot.input_device, 1 * 3 * 640 * 640 * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMalloc (input) failed");
      if (cudaMalloc(&slot.output_device, 1 * 25200 * 22 * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMalloc (output) failed");
      if (cudaMallocHost(reinterpret_cast<void **>(&slot.input_host), 1 * 3 * 640 * 640 * sizeof(float)) !=
          cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMallocHost (input) failed");
      if (cudaMallocHost(
            reinterpret_cast<void **>(&slot.output_host), 1 * 25200 * 22 * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMallocHost (output) failed");
      if (cudaEventCreate(&slot.d2h_done) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaEventCreate failed");
    }

    tools::logger()->info(
      "[MultiThreadDetector] initialized ! using TensorRT backend (ring size {}), engine={}", kTrtRingSize,
      engine_path);
    return;  // skip the OpenVINO compiled_model_ setup below -- unused in TensorRT mode
  }
#endif

  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})  // TODO
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

MultiThreadDetector::~MultiThreadDetector()
{
#ifdef HAVE_TENSORRT
  for (auto & slot : trt_slots_) {
    if (slot.input_device) cudaFree(slot.input_device);
    if (slot.output_device) cudaFree(slot.output_device);
    if (slot.input_host) cudaFreeHost(slot.input_host);
    if (slot.output_host) cudaFreeHost(slot.output_host);
    if (slot.raw_input_device) cudaFree(slot.raw_input_device);
    if (slot.raw_input_host) cudaFreeHost(slot.raw_input_host);
    if (slot.d2h_done) cudaEventDestroy(slot.d2h_done);
  }
  if (trt_stream_) cudaStreamDestroy(trt_stream_);
#endif
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
#ifdef HAVE_TENSORRT
  if (use_tensorrt_) {
    push_tensorrt(img, t);
    return;
  }
#endif

  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // preproces
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  auto input_port = compiled_model_.input();
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  queue_.push({img.clone(), t, std::move(infer_request)});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
#ifdef HAVE_TENSORRT
  if (use_tensorrt_) {
    auto [img, t, armors] = pop_tensorrt();
    return {std::move(armors), t};
  }
#endif

  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
#ifdef HAVE_TENSORRT
  if (use_tensorrt_) {
    auto [img, t, armors] = pop_tensorrt();
    return {img, std::move(armors), t};
  }
#endif

  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

#ifdef HAVE_TENSORRT
void MultiThreadDetector::push_tensorrt(cv::Mat & img, std::chrono::steady_clock::time_point t)
{
  int slot_idx = trt_next_slot_;
  trt_next_slot_ = (trt_next_slot_ + 1) % kTrtRingSize;
  auto & slot = trt_slots_[slot_idx];

  // Backpressure: block until pop_tensorrt() has fully finished reading
  // this slot's *previous* occupant -- GPU completion (d2h_done) alone is
  // not enough to prove that, only that the GPU is done, not that the
  // consumer thread has read the result out yet. See the in_flight/
  // trt_slot_mutex_ comment in mt_detector.hpp for why this distinction is
  // load-bearing here (it wasn't before GPU-side preprocessing made push()
  // fast enough to race ahead of a slower consumer).
  {
    std::unique_lock<std::mutex> lock(trt_slot_mutex_);
    trt_slot_cv_.wait(lock, [&] { return !slot.in_flight.load(); });
  }

  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // Lazily (re)allocate every slot's raw-frame staging buffers together --
  // only changes if the source resolution changes (never, in practice, for
  // a fixed camera/ROI). Mirrors YOLOV5::infer_tensorrt_gpu_preprocess.
  auto t0 = std::chrono::steady_clock::now();
  if (trt_raw_w_ != img.cols || trt_raw_h_ != img.rows) {
    size_t raw_size = static_cast<size_t>(img.cols) * img.rows * 3;
    for (auto & s : trt_slots_) {
      if (s.raw_input_device) cudaFree(s.raw_input_device);
      if (s.raw_input_host) cudaFreeHost(s.raw_input_host);
      if (cudaMalloc(&s.raw_input_device, raw_size) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMalloc (raw input) failed");
      if (cudaMallocHost(reinterpret_cast<void **>(&s.raw_input_host), raw_size) != cudaSuccess)
        throw std::runtime_error("MultiThreadDetector: cudaMallocHost (raw input) failed");
    }
    trt_raw_w_ = img.cols;
    trt_raw_h_ = img.rows;
  }

  // Copy into this slot's tightly-packed pinned staging buffer -- handles
  // both a contiguous frame and a non-contiguous ROI-cropped view.
  cv::Mat host_view(img.rows, img.cols, CV_8UC3, slot.raw_input_host);
  img.copyTo(host_view);
  auto t1 = std::chrono::steady_clock::now();

  // H2D (raw frame), GPU preprocessing kernel (letterbox resize + BGR->RGB
  // + normalize + HWC->CHW, see preprocess_kernel.cu), infer, D2H -- all
  // enqueued async on trt_stream_, no sync in between. This slot's device
  // buffers aren't touched by any other in-flight work (we just confirmed
  // above that any *previous* occupant's D2H has completed), and
  // setTensorAddress is re-pointed immediately before this enqueueV3 call,
  // so this frame's kernels read/write only this slot's addresses.
  size_t raw_size = static_cast<size_t>(img.cols) * img.rows * 3;
  cudaMemcpyAsync(
    slot.raw_input_device, slot.raw_input_host, raw_size, cudaMemcpyHostToDevice, trt_stream_);

  launch_letterbox_preprocess(
    static_cast<const uint8_t *>(slot.raw_input_device), img.cols, img.rows, img.cols * 3,
    static_cast<float *>(slot.input_device), w, h, static_cast<float>(scale), trt_stream_);

  trt_context_->setTensorAddress(trt_input_name_.c_str(), slot.input_device);
  trt_context_->setTensorAddress(trt_output_name_.c_str(), slot.output_device);
  if (!trt_context_->enqueueV3(trt_stream_)) {
    throw std::runtime_error("MultiThreadDetector: TensorRT enqueueV3 failed");
  }

  cudaMemcpyAsync(
    slot.output_host, slot.output_device, 1 * 25200 * 22 * sizeof(float), cudaMemcpyDeviceToHost,
    trt_stream_);
  cudaEventRecord(slot.d2h_done, trt_stream_);
  slot.in_flight.store(true);

  auto t2 = std::chrono::steady_clock::now();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  tools::logger()->info(
    "[MT-TRT-TIMING] memcpy={:.2f}ms dispatch={:.2f}ms slot={}", ms(t0, t1), ms(t1, t2), slot_idx);

  trt_queue_.push({img.clone(), t, slot_idx, scale});
}

std::tuple<cv::Mat, std::chrono::steady_clock::time_point, std::list<Armor>>
MultiThreadDetector::pop_tensorrt()
{
  auto [img, t, slot_idx, scale] = trt_queue_.pop();
  auto & slot = trt_slots_[slot_idx];

  // Wait ONLY for this slot's D2H copy -- not a full stream sync, which
  // would incorrectly also block on later frames' work the producer thread
  // may have already enqueued by now.
  cudaEventSynchronize(slot.d2h_done);

  // View, not clone: consumed synchronously by postprocess() right below,
  // and the slot is provably still ours -- in_flight (and therefore
  // push_tensorrt()'s ability to reuse this slot) isn't cleared until after
  // this line runs.
  cv::Mat output(25200, 22, CV_32F, slot.output_host);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  // Only now is this slot truly free -- wake any push_tensorrt() call
  // blocked waiting to reuse it (see the wait in push_tensorrt() and the
  // in_flight/trt_slot_mutex_ comment in mt_detector.hpp).
  {
    std::lock_guard<std::mutex> lock(trt_slot_mutex_);
    slot.in_flight.store(false);
  }
  trt_slot_cv_.notify_all();

  return {img, t, std::move(armors)};
}
#endif

}  // namespace multithread

}  // namespace auto_aim
