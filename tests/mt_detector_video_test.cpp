#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// Exercises MultiThreadDetector::push()/debug_pop() against a video file --
// no camera/CBoard/aiming dependencies, unlike mt_standard.cpp/
// mt_auto_aim_debug.cpp (which need real hardware). Lets the TensorRT async
// ring-buffered pipeline (see mt_detector.hpp/.cpp) be validated for
// correctness (armor counts vs. the existing single-threaded
// detector_video_test on the same video) and measured (tegrastats GPU/CPU
// overlap, throughput) without needing a camera or gimbal attached.

const std::string keys =
  "{help h usage ? |                            | 输出命令行参数说明 }"
  "{config-path c  | configs/demo_tensorrt.yaml | yaml配置文件的路径}"
  "{@video_path    | assets/demo/demo.avi       | avi路径}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto video_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  tools::Exiter exiter;

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("failed to open {}", video_path);
    return 1;
  }
  int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

  auto_aim::multithread::MultiThreadDetector detector(config_path, true);

  std::atomic<bool> producer_done{false};
  std::atomic<int> pushed_count{0};

  // Reads frames as fast as VideoCapture::read() + push() allow -- no
  // artificial throttle, deliberately maximizes pipeline pressure to
  // actually exercise overlap/backpressure (unlike detector_video_test.cpp's
  // waitKey(33)-throttled single-threaded loop).
  auto producer = std::thread([&]() {
    cv::Mat frame;
    while (!exiter.exit()) {
      if (!video.read(frame) || frame.empty()) break;
      detector.push(frame, std::chrono::steady_clock::now());
      pushed_count++;
    }
    producer_done = true;
  });

  int popped_count = 0;
  int total_armors = 0;
  auto start = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    // Drain whatever the producer actually got through -- pushed_count is a
    // lower bound (the queue can silently drop under backpressure, see the
    // "queue is full" log lines), so this is "drain what arrived", not
    // "drain exactly total_frames".
    if (producer_done.load() && popped_count >= pushed_count.load()) break;

    auto [img, armors, t] = detector.debug_pop();
    popped_count++;
    total_armors += static_cast<int>(armors.size());

    tools::logger()->info(
      "[{}/{}] armors={} latency_since_push={:.1f}ms", popped_count, total_frames, armors.size(),
      tools::delta_time(std::chrono::steady_clock::now(), t) * 1e3);

    cv::resize(img, img, {}, 0.5, 0.5);
    for (auto & armor : armors) tools::draw_points(img, armor.points, {0, 255, 0});
    cv::imshow("mt_detector_video_test", img);
    if (cv::waitKey(1) == 'q') break;
  }

  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  tools::logger()->info(
    "done: pushed={} popped={} total_armors={} elapsed={:.1f}s ({:.1f} fps)", pushed_count.load(),
    popped_count, total_armors, elapsed, popped_count / elapsed);

  producer.join();
  return 0;
}
