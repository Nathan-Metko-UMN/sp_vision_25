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

  // Backpressure: this slot's buffers are still owned by whatever frame
  // last used it until the GPU has actually finished with them. Wait
  // (NOT cudaStreamSynchronize -- that would also wait for later-enqueued,
  // unrelated frames) only if this specific slot still has work in flight.
  // On a healthy pipeline (ring size > frames-in-flight) this is a
  // near-instant no-op; it only actually stalls the producer if pop() is
  // falling behind, which is the correct, intentional backpressure.
  if (slot.in_flight.load()) {
    cudaEventSynchronize(slot.d2h_done);
    slot.in_flight.store(false);
  }

  // letterbox (persistent-canvas pattern, mirrors YOLOV5::detect())
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  auto t0 = std::chrono::steady_clock::now();
  if (trt_letterbox_canvas_.empty() || trt_letterbox_w_ != w || trt_letterbox_h_ != h) {
    trt_letterbox_canvas_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    trt_letterbox_w_ = w;
    trt_letterbox_h_ = h;
  }
  cv::resize(img, trt_letterbox_canvas_(cv::Rect(0, 0, w, h)), {w, h});
  auto t1 = std::chrono::steady_clock::now();

  // preprocess (BGR->RGB swap, u8->f32 scale, HWC->CHW) directly into this
  // slot's pinned input buffer.
  int blob_shape[4] = {1, 3, 640, 640};
  cv::Mat blob(4, blob_shape, CV_32F, slot.input_host);
  cv::dnn::blobFromImage(
    trt_letterbox_canvas_, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false, CV_32F);
  auto t2 = std::chrono::steady_clock::now();

  // H2D, infer, D2H -- all enqueued async on trt_stream_, no sync here. This
  // slot's device buffers aren't touched by any other in-flight work (we
  // just confirmed above that any *previous* occupant's D2H has completed),
  // and setTensorAddress is re-pointed immediately before this enqueueV3
  // call, so this frame's kernels read/write only this slot's addresses.
  cudaMemcpyAsync(
    slot.input_device, slot.input_host, 1 * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice,
    trt_stream_);

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

  auto t3 = std::chrono::steady_clock::now();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  tools::logger()->info(
    "[MT-TRT-TIMING] letterbox={:.2f}ms preprocess={:.2f}ms enqueue={:.2f}ms slot={}", ms(t0, t1),
    ms(t1, t2), ms(t2, t3), slot_idx);

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
  slot.in_flight.store(false);

  // View, not clone: postprocess() below consumes it synchronously, and
  // this slot can't be reused (push_tensorrt only advances trt_next_slot_
  // round-robin, kTrtRingSize-1 pushes away) before this function returns.
  cv::Mat output(25200, 22, CV_32F, slot.output_host);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, t, std::move(armors)};
}
#endif

}  // namespace multithread

}  // namespace auto_aim
