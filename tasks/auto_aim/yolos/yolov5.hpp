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
  std::string trt_input_name_, trt_output_name_;

  void trt_build_or_load_engine(const std::string & onnx_path, const std::string & engine_path);
  cv::Mat infer_tensorrt(const cv::Mat & input);
#endif

  cv::Mat infer_openvino(const cv::Mat & input);

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

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