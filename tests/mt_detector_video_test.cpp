#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
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
//
// All frames are decoded up front into memory before any timing starts, so
// per-frame numbers reflect ONLY the detection pipeline -- video.read() is a
// CPU file decode (unlike a real camera's fast DMA buffer grab), and mixing
// it into per-frame push/pop timing was misleading (it dominated the
// harness's aggregate fps in an earlier version of this tool, making the
// pipeline look worse than it is). --sync runs the same preloaded frames
// through the existing single-threaded YOLO::detect() path instead, for a
// direct, decode-excluded, apples-to-apples comparison against the async
// path -- both use the exact same frames.

const std::string keys =
  "{help h usage ? |                            | 输出命令行参数说明 }"
  "{config-path c  | configs/demo_tensorrt.yaml | yaml配置文件的路径}"
  "{sync           |                            | 使用单线程YOLO::detect()而不是异步流水线，用于对比}"
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
  bool sync_mode = cli.has("sync");

  tools::Exiter exiter;

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("failed to open {}", video_path);
    return 1;
  }

  std::vector<cv::Mat> frames;
  auto decode_start = std::chrono::steady_clock::now();
  for (cv::Mat frame; video.read(frame) && !frame.empty();) frames.push_back(frame.clone());
  auto decode_elapsed =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
  tools::logger()->info(
    "preloaded {} frames in {:.1f}s ({:.1f}ms/frame avg -- this is the video FILE decode cost a real "
    "camera wouldn't pay, deliberately excluded from everything timed below)",
    frames.size(), decode_elapsed, decode_elapsed * 1e3 / frames.size());

  if (sync_mode) {
    // Single-threaded comparison path: same frames, same TensorRT engine
    // (via the ordinary YOLO wrapper, exactly what auto_aim_test.cpp uses),
    // no threading/queueing/ring-buffer involved at all -- isolates the
    // synchronous per-frame cost this async pipeline is being compared
    // against.
    auto_aim::YOLO yolo(config_path, true);
    int total_armors = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < frames.size() && !exiter.exit(); i++) {
      auto t0 = std::chrono::steady_clock::now();
      auto armors = yolo.detect(frames[i], static_cast<int>(i));
      auto t1 = std::chrono::steady_clock::now();
      total_armors += static_cast<int>(armors.size());
      tools::logger()->info(
        "[sync {}] armors={} detect={:.1f}ms", i + 1, armors.size(), tools::delta_time(t1, t0) * 1e3);
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    tools::logger()->info(
      "done (sync): frames={} total_armors={} elapsed={:.1f}s ({:.1f} fps, decode excluded)",
      frames.size(), total_armors, elapsed, frames.size() / elapsed);
    return 0;
  }

  auto_aim::multithread::MultiThreadDetector detector(config_path, true);

  std::atomic<bool> producer_done{false};
  std::atomic<int> pushed_count{0};

  // Pushes preloaded frames as fast as push() allows -- no artificial
  // throttle, deliberately maximizes pipeline pressure to actually exercise
  // overlap/backpressure (unlike detector_video_test.cpp's
  // waitKey(33)-throttled single-threaded loop). push= is logged per frame
  // (mirroring auto_aim_test.cpp's per-stage "yolo: Xms, tracker: Yms"
  // style) -- this is the producer thread's own CPU cost (letterbox +
  // preprocess + H2D/enqueue/D2H *dispatch*, not waiting for completion).
  auto producer = std::thread([&]() {
    for (size_t i = 0; i < frames.size() && !exiter.exit(); i++) {
      auto t0 = std::chrono::steady_clock::now();
      detector.push(frames[i], t0);
      auto t1 = std::chrono::steady_clock::now();
      pushed_count++;
      tools::logger()->info("[push {}] push={:.1f}ms", i + 1, tools::delta_time(t1, t0) * 1e3);
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

    // Time from push() to armors being ready -- the pipeline-latency analog
    // of auto_aim_test.cpp's per-frame "yolo: Xms" line, and directly
    // comparable to --sync's detect= line (both decode-excluded, both
    // wrapping "handed a frame" -> "armors ready"). Includes any queue wait
    // under backpressure, which is real end-to-end latency, not just GPU
    // time -- imshow/waitKey below run on this same consumer thread and can
    // add to that wait for whichever frame is popped next.
    tools::logger()->info(
      "[pop {}] armors={} pipeline_latency={:.1f}ms", popped_count, armors.size(),
      tools::delta_time(std::chrono::steady_clock::now(), t) * 1e3);

    cv::resize(img, img, {}, 0.5, 0.5);
    for (auto & armor : armors) tools::draw_points(img, armor.points, {0, 255, 0});
    cv::imshow("mt_detector_video_test", img);
    if (cv::waitKey(1) == 'q') break;
  }

  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  tools::logger()->info(
    "done: pushed={} popped={} total_armors={} elapsed={:.1f}s ({:.1f} fps, decode excluded -- "
    "still includes imshow/waitKey on the consumer thread; see per-frame push=/pipeline_latency= "
    "lines above for the pure detection-pipeline numbers, and re-run with --sync for the "
    "single-threaded comparison)",
    pushed_count.load(), popped_count, total_armors, elapsed, popped_count / elapsed);

  producer.join();
  return 0;
}
