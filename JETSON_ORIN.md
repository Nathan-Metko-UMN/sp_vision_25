# sp_vision_25 — Architecture Overview & Jetson Orin Guide

This document explains what this repository does, how it's put together, and what it
takes to run it on an NVIDIA Jetson Orin instead of the Intel NUC it was designed for.
It also documents the accompanying `Dockerfile`, which builds the project without
requiring you to hand-install every dependency.

## 1. What this project is

`sp_vision_25` is Tongji University SuperPower RoboMaster team's 2025-season vision
stack for **autonomous aiming ("auto-aim")** and **rune/buff hitting ("auto-buff")**
on a combat robot. Given a camera feed and IMU/gimbal feedback from an embedded
controller, it:

1. Detects enemy armor plates (or the rotating "buff" rune panel) in each frame.
2. Estimates the target's 3D pose and motion (position, velocity, yaw spin) with an
   Extended Kalman Filter.
3. Plans a firing trajectory ("轨迹规划器") that predicts where the target will be
   after bullet flight time + system latency, accounting for the gimbal's achievable
   acceleration so the aim point is actually reachable.
4. Decides when to fire based on how well the planned gimbal trajectory matches the
   predicted target trajectory.
5. Sends aim/fire commands to a downstream STM32-based controller ("C板") over
   CAN/serial, and reads IMU quaternions back for gimbal-to-world alignment.

It is pure C++ (no ROS required for the core pipeline); ROS 2 is only used
optionally by the "sentry" (哨兵) build variant to talk to a navigation stack.

## 2. Repository layout

```
sp_vision_25
├── assets/          Demo video + neural net weights (ONNX and OpenVINO IR .xml/.bin)
├── calibration/      Camera intrinsic + hand-eye calibration utilities
├── configs/          Per-robot YAML configs (standard3/4, sentry, uav, ascento, ...)
├── io/                Hardware abstraction layer
│   ├── camera.cpp/hpp        Camera facade
│   ├── hikrobot/              HikRobot industrial camera SDK wrapper (amd64+arm64 .so bundled)
│   ├── mindvision/             MindVision industrial camera SDK wrapper (amd64+arm64 .so bundled)
│   ├── usbcamera/               Generic USB (V4L2/UVC) camera support
│   ├── cboard.cpp/hpp           STM32 controller link: CAN + IMU quaternion queue
│   ├── socketcan.hpp            Linux SocketCAN wrapper
│   ├── gimbal/                   Alternate serial-based gimbal driver
│   ├── dm_imu/                    Damiao IMU driver
│   ├── serial/                    Vendored cross-platform serial library
│   └── ros2/                       Optional nav-stack publish/subscribe bridge
├── tasks/
│   ├── auto_aim/       Detector (traditional CV + YOLOv5/v8/v11 via OpenVINO),
│   │                   classifier, EKF tracker, solver (PnP), trajectory planner
│   │                   (incl. vendored TinyMPC QP solver), aimer, shooter, voter
│   ├── auto_buff/       Rune/buff-specific detector, solver, predictor (RANSAC sine
│   │                   fit), aimer — uses Ceres for nonlinear fitting
│   └── omniperception/    Sentry-specific perception/decision helper
├── tools/             Shared utilities: logging (spdlog), YAML config loader,
│                       EKF, PID, CRC, image tools, ballistic trajectory math,
│                       thread-safe queue/pool, video recorder, PlotJuggler plotter
├── src/               Executable entry points (standard.cpp, sentry.cpp, uav.cpp, ...)
├── tests/             Standalone test/debug programs per module
└── CMakeLists.txt     Top-level build; conditionally builds ROS2/sentry targets
```

### Software layering (io → tasks → src)
`io` wraps all hardware (cameras, CAN, serial). `tasks` implements the
perception/decision algorithms as reusable libraries (`auto_aim`, `auto_buff`,
`omniperception`), independent of any specific robot. `src/*.cpp` are thin
per-robot `main()` programs that wire `io` + `tasks` together differently
depending on the robot type (e.g. `standard.cpp` for an infantry robot,
`sentry.cpp` for the ROS2-connected sentry, `uav.cpp` for a drone). `tools` is a
grab-bag of dependency-free helpers used everywhere.

## 3. Build dependencies

| Dependency | Purpose | Version pinned in repo |
|---|---|---|
| OpenCV | Image I/O, traditional CV armor detection | apt `libopencv-dev` (~4.5.4 on Ubuntu 22.04) |
| **OpenVINO** | Neural-net inference (YOLOv5/v8/v11 armor + buff detectors) | **hard-coded to `/opt/intel/openvino_2024.6.0/`** in 3 CMakeLists files |
| Eigen3 | Linear algebra (EKF, geometry) | apt |
| Ceres Solver | Nonlinear least squares (buff rune sine-motion fit) | apt |
| fmt, spdlog | Formatting/logging | apt |
| yaml-cpp | Per-robot YAML config parsing | apt |
| nlohmann-json | JSON (config/telemetry) | apt |
| libusb-1.0 | USB camera SDK transport | apt |
| TinyMPC | Vendored QP solver for trajectory planning | in-tree source (`tasks/auto_aim/planner/tinympc`) |
| HikRobot MVS SDK | Industrial camera driver | **vendored `.so` for amd64 *and* arm64** already in `io/hikrobot/lib/` |
| MindVision SDK | Industrial camera driver (alternate) | **vendored `.so` for amd64 *and* arm64** already in `io/mindvision/lib/` |
| vendored `serial` lib | Cross-platform serial port I/O | in-tree source (`io/serial`) |
| ROS 2 (rclcpp, std_msgs, `sp_msgs`) | Only for the sentry nav-bridge build | optional, `find_package(... QUIET)` — skipped entirely if absent |

The top-level `CMakeLists.txt` already checks `CMAKE_SYSTEM_PROCESSOR` and picks
`hikrobot/lib/amd64` vs `hikrobot/lib/arm64` (and the same for MindVision)
automatically — **this repo was already set up with ARM64 in mind**, it's just
never been fully wired up for a Jetson.

## 4. Can this run on a Jetson Orin? — Short answer: mostly yes, with one real caveat

Jetson Orin boards (Nano/NX/AGX) run **Ubuntu 22.04 aarch64** under JetPack 6, so
architecture-wise this is a native fit, not an emulation hack. Here's the
component-by-component breakdown:

| Component | Jetson Orin status | Notes |
|---|---|---|
| CPU architecture | ✅ Works | CMake already branches on `aarch64` for HikRobot/MindVision libs |
| HikRobot camera SDK | ✅ Works | arm64 `.so` already vendored in-repo |
| MindVision camera SDK | ✅ Works | arm64 `.so` already vendored in-repo |
| Generic USB/V4L2 camera | ✅ Works | no architecture dependency |
| CAN bus (SocketCAN) | ✅ Works | needs a CAN interface — via a Jetson carrier board's onboard CAN controller, or any USB-CAN adapter (`can0`) |
| Serial (STM32 C-board, IMU) | ✅ Works | vendored `serial` lib is a plain POSIX termios wrapper |
| OpenCV / Eigen / Ceres / fmt / spdlog / yaml-cpp / nlohmann-json | ✅ Works | all available as aarch64 apt packages on Ubuntu 22.04 |
| **OpenVINO inference — `device: CPU`** | ✅ Works | Intel ships an official **arm64 Linux archive** for OpenVINO 2024.6.0 — but only built against **Ubuntu 20.04**, not 22.04 (`l_openvino_toolkit_ubuntu20_2024.6.0.17404.4c0f47d2335_arm64.tgz`; verified this downloads correctly). There is no `ubuntu22`+arm64 combination for this OpenVINO version. This is fine in practice: the archive bundles its own runtime libraries, and Ubuntu 22.04's glibc/libstdc++ are backwards-compatible with binaries built against 20.04's older ones, so it runs unmodified on the Jetson's Ubuntu 22.04 (JetPack 6) userspace. The Dockerfile handles this automatically. CPU plugin runs on ARM. |
| **OpenVINO inference — `device: GPU`** | ❌ Does not work | OpenVINO's `GPU` plugin only targets **Intel** GPUs (iGPU/Arc via oneAPI/Level Zero). It has no backend for Jetson's NVIDIA/CUDA GPU. Every shipped config (`standard3.yaml`, `standard4.yaml`, `sentry.yaml`, `mvs.yaml`, `ascento.yaml`) sets `device: GPU` — **you must change this to `device: CPU`** in whatever config you deploy on a Jetson, or inference will fail to find a device. |
| ROS 2 (sentry mode) | ✅ Works, if wanted | JetPack 6 (Ubuntu 22.04) supports ROS 2 Humble same as any x86 Ubuntu 22.04 box. The `sentry*` executables also depend on a custom `sp_msgs` package that isn't in this repo, so full sentry functionality requires that package from elsewhere regardless of platform. |

**Bottom line:** the codebase itself is portable — nothing in the vision pipeline
is x86-specific. The one thing that will *not* magically work is GPU-accelerated
inference, because OpenVINO's GPU backend is Intel-only. On a Jetson you get
**CPU inference through OpenVINO**, which will be noticeably slower than the
Intel iGPU path this project was tuned for on the NUC. If you need the Orin's
GPU to actually accelerate inference, the models would need to be exported to
TensorRT and the `auto_aim::YOLO`/`auto_buff` inference backends
(`tasks/auto_aim/yolos/*`, `tasks/auto_buff/yolo11_buff.*`) would need a new
TensorRT-based implementation alongside the existing OpenVINO one — that code
does not exist in this repo today.

Also note the `assets/*.xml`/`*.bin` files are OpenVINO IR models exported for
FP32/INT8 — these load fine on the CPU plugin, no re-export needed to just get
it running (only needed if you later want max CPU-inference speed via
re-quantization, or if you build the TensorRT path).

## 5. Using the Dockerfile

The included `Dockerfile` builds an aarch64 image with every apt-installable
dependency plus OpenVINO 2024.6.0 (matching the version hard-coded in the repo's
CMakeLists files), and compiles the project. It's architecture-aware: build it
directly **on** a Jetson Orin (or any aarch64 host) and it pulls the arm64
(ubuntu20-packaged, see §4) OpenVINO archive automatically; build it on an
x86_64 dev machine and it pulls the x86_64 (ubuntu22-packaged) archive instead.
**This has been built and verified end-to-end for the x86_64 path** (full
compile of every non-ROS2 target succeeds); the arm64 download path was
verified to fetch a valid archive but has not been build-tested on real Jetson
hardware — see §6.

It does **not** install ROS 2 — the ROS2-gated targets (`sentry`, `sentry_bp`,
`sentry_debug`, `sentry_multithread`, `publish_test`, `subscribe_test`,
`topic_loop_test`) are simply skipped by CMake (this is the existing, intended
behavior — see the `find_package(... QUIET)` guard in `CMakeLists.txt`). Every
other executable (`standard`, `mt_standard`, `auto_aim_test`, `auto_buff_test`,
calibration tools, etc.) builds normally. If you need the sentry/ROS2 build,
see §5.3 below.

### 5.1 Build the image

On the Jetson Orin itself (recommended — no cross-compilation needed):

```bash
docker build -t sp_vision_25 .
```

From an x86_64 machine, cross-building for the Jetson via buildx:

```bash
docker buildx build --platform linux/arm64 -t sp_vision_25:arm64 --load .
```

### 5.2 Run it

The container needs access to the camera (USB), the CAN interface, and/or the
serial port for the C-board, depending on what your robot config uses. The
simplest way to get all of that is `--privileged` with the host network and
device tree bind-mounted; for tighter scoping, pass individual `--device` flags
instead:

```bash
docker run --rm -it \
  --privileged \
  --network host \
  -v /dev:/dev \
  -v "$(pwd)/configs:/root/sp_vision_25/configs" \
  -v "$(pwd)/logs:/root/sp_vision_25/logs" \
  sp_vision_25 \
  ./build/auto_aim_test configs/standard4.yaml
```

Remember to edit whichever config you mount in to set `device: CPU` (see §4).

### 5.3 Adding ROS 2 / sentry support (optional)

This repo's sentry executables require a `sp_msgs` ROS2 package that isn't
vendored here. If you have access to it and want the sentry build:

1. Start from `FROM ros:humble-ros-base` (arm64 manifest available) instead of
   plain `ubuntu:22.04`, or `apt install ros-humble-ros-base` on top of this
   Dockerfile.
2. Bring in `sp_msgs` (and any other custom message packages your nav stack
   needs) as an additional `COPY` + `colcon build`, sourced before the CMake
   configure step so `find_package(sp_msgs)` succeeds.
3. Re-run `cmake -B build && cmake --build build` — the `sentry*` targets will
   then be included automatically since the CMake ROS2 guard will pass.

### 5.4 Intel GPU acceleration on Windows (Docker Desktop / WSL2)

Not a Jetson topic, but the same `device: GPU` OpenVINO setting from §4, so
it's documented here: if you're developing on a Windows machine with an Intel
iGPU (not the Jetson itself), the Dockerfile now installs Intel's actual GPU
compute runtime (`intel-opencl-icd`, `intel-level-zero-gpu`) — the OpenVINO
dependency script alone only installs the generic OpenCL ICD loader, which is
not enough for the GPU plugin to find a device.

Even with the runtime installed, the **container still needs the host GPU
device passed through at run time**:

- **Native Linux host:** `docker run --device=/dev/dri ...`
- **Windows host via Docker Desktop/WSL2:** there is no `/dev/dri` in Docker
  Desktop's WSL VM. GPU access instead goes through the WSL GPU
  paravirtualization device `/dev/dxg` plus the vendor driver shim libraries
  Docker Desktop mirrors from the host under `/usr/lib/wsl/lib`:
  ```
  docker run --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl ...
  ```
  `.devcontainer/devcontainer.json` in this repo already sets this up via
  `runArgs`/`mounts`, so opening this repo in VS Code's Dev Containers on this
  machine gets GPU access automatically — rebuild the container
  (**Dev Containers: Rebuild Container**) to pick it up.

This was verified end-to-end: `auto_aim_test` run against `configs/demo.yaml`
(`device: GPU`) inside the container, with the host's Intel UHD Graphics iGPU
passed through this way, ran YOLO inference at ~9-10ms/frame with no
OpenVINO device-not-found error. Note this only works for an **Intel** GPU —
an NVIDIA GPU on the same Windows machine is irrelevant here, since OpenVINO's
GPU plugin never talks to NVIDIA hardware (same restriction as Jetson in §4).

## 6. Known gaps / things to verify on real hardware

- **GPU inference is not available on Jetson** (see §4) — plan for CPU-plugin
  latency, or budget time to add a TensorRT backend if frame rate is
  insufficient.
- Camera SDKs (HikRobot/MindVision) are vendored as prebuilt `.so` files with no
  visible build/version metadata in this repo — if the physical camera's
  firmware requires a newer SDK than what's bundled, you'll need to source an
  updated arm64 `.so` from the vendor.
- CAN bus setup (`can0`/`can1` bring-up via udev rule) assumes a
  SocketCAN-compatible interface is present — most Jetson carrier boards don't
  expose CAN natively (unlike the AGX Orin devkit's dev header, which requires
  extra hardware/DTS overlay setup), so you likely need a USB-CAN adapter.
- **What was actually verified while writing this doc:** the `Dockerfile`'s
  x86_64 path was built end-to-end on a real Docker daemon and every non-ROS2
  target compiled and linked successfully. The arm64 OpenVINO archive URL was
  confirmed to serve a genuine, correctly-sized gzip archive (not an error
  page — an earlier draft of this Dockerfile silently downloaded an HTML 404
  page because the URL path/filename versioning is inconsistent on Intel's
  download server; the current Dockerfile avoids that trap).
- **Also verified:** `device: GPU` inference (§5.4) — YOLO detection running
  through the OpenVINO GPU plugin on an Intel iGPU inside the container, not
  just CPU-plugin builds.
- **Not yet verified:** an actual build *on* aarch64 hardware, and running any
  executable against a real camera/CAN/serial/C-board. Treat the Jetson
  compatibility table in §4 as a well-checked static analysis (CMake logic,
  package availability, architecture guards) rather than a confirmed
  on-device test.
