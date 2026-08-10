#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

#include <memory>
#endif

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#endif

#if defined(HAVE_VPI) && defined(HAVE_TENSORRT)
#include <vpi/Image.h>
#include <vpi/Stream.h>
#endif

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLOV5 : public YOLOBase
{
public:
  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  bool use_cuda_ = false;
#ifdef HAVE_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  std::string ort_input_name_, ort_output_name_;
  cv::Mat infer_cuda(const cv::Mat & input);
#endif

  bool use_tensorrt_ = false;
#ifdef HAVE_TENSORRT
  class TRTLogger : public nvinfer1::ILogger
  {
  public:
    void log(Severity severity, const char * msg) noexcept override;
  };
  TRTLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
  cudaStream_t trt_stream_ = nullptr;
  void * trt_input_device_ = nullptr;
  void * trt_output_device_ = nullptr;
  // Pinned (page-locked) host buffers, allocated once and reused every
  // frame -- pageable host memory (e.g. plain cv::Mat/new[]) forces the CUDA
  // driver to stage through an internal pinned bounce buffer on every
  // cudaMemcpyAsync, roughly doubling H2D/D2H transfer time. Preprocessing
  // writes directly into trt_input_host_ (no separate host buffer + copy).
  float * trt_input_host_ = nullptr;
  float * trt_output_host_ = nullptr;
  std::string trt_input_name_, trt_output_name_;

  void trt_build_or_load_engine(const std::string & onnx_path, const std::string & engine_path);
  // input_is_rgb: true skips blobFromImage's BGR->RGB swap (the VIC
  // preprocessing path below already hands back RGB); false (the plain CPU
  // letterbox path) keeps the original BGR->RGB swap behavior.
  cv::Mat infer_tensorrt(const cv::Mat & input, bool input_is_rgb = false);
#endif

#if defined(HAVE_VPI) && defined(HAVE_TENSORRT)
  // VIC (Tegra's dedicated hardware image processor) offload for the
  // letterbox resize, via NVIDIA VPI. VIC on this hardware only accepts
  // RGBA8 -- not BGR8/U8 -- so the pipeline is: CPU BGR->RGBA convert (full
  // source resolution) -> VIC rescale straight into a view of a persistent,
  // zeroed 640x640 RGBA canvas (replicating the letterbox pad-with-black
  // behavior in one hardware call, same as the CPU path's
  // resize-into-black-canvas) -> CPU RGBA->RGB drop-alpha (small, 640x640
  // only) -> normal blobFromImage normalize+CHW. See yolov5.cpp for why this
  // is not a guaranteed win: converting the full source frame to RGBA before
  // VIC ever sees it may cost as much as the CPU resize it replaces.
  VPIStream vpi_stream_ = nullptr;
  VPIImage vpi_input_ = nullptr;   // wraps rgba_full_
  VPIImage vpi_canvas_ = nullptr;  // wraps rgba_canvas_ (640x640 RGBA8, zeroed once)
  VPIImage vpi_canvas_view_ = nullptr;  // sub-rect view into vpi_canvas_, recreated if w_/h_ change
  cv::Mat rgba_full_;    // persistent BGR->RGBA scratch buffer, sized to match bgr_img
  cv::Mat rgba_canvas_;  // persistent 640x640x4 RGBA canvas backing vpi_canvas_
  cv::Mat rgb_canvas_;   // persistent 640x640x3 RGBA->RGB (alpha dropped) buffer, feeds blobFromImage
  int vpi_input_w_ = -1, vpi_input_h_ = -1;
  int vpi_view_w_ = -1, vpi_view_h_ = -1;

  void infer_tensorrt_vic_preprocess(const cv::Mat & bgr_img, int w, int h);
#endif

  cv::Mat infer_openvino(const cv::Mat & input);

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  // Persistent letterbox canvas, reused every frame instead of allocating +
  // zero-filling a fresh 640x640x3 buffer each call (measured as a real,
  // non-trivial per-frame cost on Jetson). Valid whenever letterbox_w_/
  // letterbox_h_ match the current frame's computed resize target; re-zeroed
  // if not (e.g. first frame, or source resolution changed).
  cv::Mat letterbox_canvas_;
  int letterbox_w_ = -1, letterbox_h_ = -1;

  Detector detector_;
  friend class MultiThreadDetector;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP