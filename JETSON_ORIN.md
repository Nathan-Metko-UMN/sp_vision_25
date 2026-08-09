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
| **OpenVINO inference — `device: GPU`** | ❌ Does not work | OpenVINO's `GPU` plugin only targets **Intel** GPUs (iGPU/Arc via oneAPI/Level Zero). It has no backend for Jetson's NVIDIA/CUDA GPU. Every shipped config (`standard3.yaml`, `standard4.yaml`, `sentry.yaml`, `mvs.yaml`, `ascento.yaml`) sets `device: GPU` — **you must change this to `device: CPU` or `device: CUDA`** in whatever config you deploy on a Jetson, or inference will fail to find a device. |
| **`device: CUDA` (Jetson's own GPU)** | ✅ Works | Separate ONNX Runtime CUDA backend, not OpenVINO — see §5.6. Only `YOLOV5` supports it (the model every shipped config actually uses). |
| ROS 2 (sentry mode) | ✅ Works, if wanted | JetPack 6 (Ubuntu 22.04) supports ROS 2 Humble same as any x86 Ubuntu 22.04 box. The `sentry*` executables also depend on a custom `sp_msgs` package that isn't in this repo, so full sentry functionality requires that package from elsewhere regardless of platform. |

**Bottom line:** the codebase itself is portable — nothing in the vision pipeline
is x86-specific. GPU-accelerated inference on Jetson's own NVIDIA GPU is
possible via `device: CUDA` (§5.6), a separate backend from OpenVINO's GPU
plugin (which is Intel-only and simply won't work here). `device: CPU`
(OpenVINO's CPU plugin) remains available as a simpler fallback with no
JetPack/NVIDIA Container Runtime setup required.

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
  ./build/standard --config-path=configs/standard4.yaml
```

Note the `--config-path=` (with `=`) — `auto_aim_test`/`standard`/etc. use
OpenCV's `CommandLineParser`, which requires `--flag=value` or `-c=value`;
a bare `configs/standard4.yaml` positional argument is silently interpreted
as something else entirely (for `auto_aim_test`, the input *video* path) and
the config path quietly falls back to its default instead of erroring. Also
note `auto_aim_test` specifically replays a recorded video
(`assets/demo/demo.avi`) — for a live camera/CAN/serial robot like this
example implies, use `./build/standard` instead.

Remember to edit whichever config you mount in to set `device: CPU` or
`device: CUDA` (see §4) — every shipped config defaults to `device: GPU`,
which does not work on Jetson. For `device: CUDA`, add
`--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all` to the `docker run` command
above (see §5.6).

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

### 5.5 NVIDIA GPU acceleration (`device: CUDA`)

Since OpenVINO's `GPU` device can never use an NVIDIA card, NVIDIA GPU
acceleration is a **second, independent inference backend**: setting
`device: CUDA` in a yaml config makes `YOLOV5` (the only model any shipped
config actually uses — see below) run through **ONNX Runtime's CUDA
execution provider** instead of OpenVINO, while every other class
(`YOLO11`/`YOLOV8`) and every other part of the pipeline (tracker, aimer,
solver, ...) is completely unchanged.

**Why not just point OpenVINO at the NVIDIA GPU?** There's a community
`nvidia_plugin` for OpenVINO (in `openvino_contrib`), but it's pinned to
OpenVINO **2024.1.0** (this repo hard-codes 2024.6.0 — plugins aren't
ABI-compatible across versions), needs OpenVINO built from source plus exact
CUDA 11.8 / cuDNN 8.6.0 / cuTENSOR 1.6.1 versions, and isn't part of Intel's
distribution. Too fragile to depend on. ONNX Runtime's CUDA EP is the
actively maintained, officially supported way to run a model on an NVIDIA GPU
outside OpenVINO.

**Where `assets/yolov5.onnx` came from:** OpenVINO can convert *into* IR from
ONNX, but not back out — there's no supported IR→ONNX path. `assets/yolov5.onnx`
was produced with the third-party
[`openvino2onnx`](https://pypi.org/project/openvino2onnx/) tool
(`python -m openvino2onnx assets/yolov5.xml assets/yolov5.onnx`) from the
exact same `assets/yolov5.xml`/`.bin` weights the OpenVINO path uses — not a
retrained or re-exported model. This was numerically validated (not just
"it loads"): both the IR and the resulting ONNX were run on identical real
video frames through OpenVINO and ONNX Runtime respectively, and the outputs
matched to ~1e-3 (bounding boxes, class scores, and the top detection all
identical). The same approach was tried for `yolo11.xml`, but the converted
model failed ONNX Runtime's own load-time shape inference (a YOLO11-specific
op in its detection head didn't survive conversion) — since no shipped
config actually uses `yolo_name: yolo11` or `yolov8` (every config uses
`yolov5`), this wasn't pursued further. Extending `device: CUDA` to those
classes would mean either fixing that conversion or sourcing the original
pre-IR `.onnx`/`.pt` weights, and duplicating the small amount of
CUDA-backend plumbing added to `yolov5.hpp`/`.cpp` into `yolo11.hpp`/`.cpp`
and `yolov8.hpp`/`.cpp`.

**Build-time requirements** (all x86_64-only, added to the Dockerfile):
CUDA 12.6 runtime + cuDNN 9 (`cuda-cudart-12-6`, `libcublas-12-6`, etc., via
NVIDIA's apt repo — not the full CUDA toolkit, no `nvcc` needed) and the
prebuilt `onnxruntime-linux-x64-gpu` release, unpacked to `/opt/onnxruntime`.
`tasks/auto_aim/CMakeLists.txt` looks for it there; if it's missing, the
project still builds fine, just without `device: CUDA` support (attempting
to use it at runtime throws a clear error instead of silently falling back).

**Run-time GPU passthrough** uses the same mechanism as §5.4: on this
Windows/WSL2 machine, `--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl` covers
*both* the Intel and NVIDIA driver shims Docker Desktop mirrors there (no
separate `--gpus all` needed — the existing `devcontainer.json` config
already provides NVIDIA access as-is). On native Linux, use `--gpus all`
(requires the NVIDIA Container Toolkit on the host).

This was verified end-to-end on the actual dev machine's NVIDIA GPU: with
`configs/demo_cuda.yaml` (a copy of `demo.yaml` with `device: CUDA`),
`auto_aim_test` logged `YOLOV5: using ONNX Runtime CUDA backend`, then ran
250+ frames at ~5-19ms per yolo+tracker+aimer cycle (after an expected
~1s first-frame CUDA/cuDNN JIT warmup) with no errors.

**Not applicable to Jetson.** The CUDA runtime this installs is the generic
x86_64 build; Jetson needs JetPack/L4T's own CUDA build tied to its embedded
driver — see §5.6 for the Jetson-specific `device: CUDA` install instead.

### 5.6 NVIDIA GPU acceleration on Jetson (`device: CUDA`)

Jetson's GPU **is** NVIDIA, so unlike the x86_64 Intel-vs-NVIDIA split in
§5.4/§5.5, this is the natural way to get real GPU acceleration on a Jetson
— but it needs a different install than §5.5's x86_64 path, because
Jetson's CUDA/cuDNN/TensorRT are JetPack/L4T-specific builds tied to
whatever's flashed on the device, not a generic desktop CUDA install.

**How it works, reflecting what a real device (JetPack 6, L4T R36.4.7, CUDA
12.6.68, cuDNN 9.3.0, TensorRT 10.3) showed, including one wrong assumption
this section originally made and had to correct after testing on hardware:**

- **First attempt (wrong):** assumed CUDA/cuDNN/TensorRT should stay off the
  image and get bind-mounted in from the host by the NVIDIA Container
  Runtime at `docker run --runtime nvidia` time — this is genuinely how it
  worked on older JetPack 4.x, and is exactly how the *Intel* GPU path in
  §5.4 works via `/dev/dxg` + `/usr/lib/wsl/lib`. On this JetPack 6 device it
  doesn't apply: `cat /etc/nvidia-container-runtime/config.toml` showed
  `mode = "auto"`, and the only CSV manifests present under
  `/etc/nvidia-container-runtime/host-files-for-container.d/` were
  `devices.csv` (GPU/display device nodes) and `drivers.csv` (display/weston
  libs) — no `cuda.csv`/`cudnn.csv`/`tensorrt.csv`. Confirmed by running with
  `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`
  and finding `/usr/local/cuda*/lib64/libcublas*` simply didn't exist inside
  the container. As of JetPack 5+, NVIDIA moved to baking CUDA/cuDNN/TensorRT
  into the container image itself instead (the CSV auto-mount mechanism now
  only covers device nodes and display libs).
- **What actually works:** CUDA/cuDNN are apt-installed straight into the
  image, same as the x86_64 path, just from NVIDIA's **Jetson/L4T** apt repo
  (`repo.download.nvidia.com/jetson/{common,t234}`, suite `r36.4`) instead of
  the desktop one, using the `jetson-ota-public.asc` signing key, and pinned
  to `cuda-cudart-12-6`/`libcudnn9-cuda-12`/etc. — the exact package names
  and versions confirmed already present on the reference device via
  `apt-cache policy`. `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all` is
  still required at `docker run` time, just for a narrower purpose now: the
  actual GPU device nodes (`/dev/nvhost-gpu`, `/dev/nvmap`, etc. from
  `devices.csv`), which genuinely do still come from the host, not the image.
- ONNX Runtime's own libraries (`libonnxruntime.so`, plus its CUDA and
  TensorRT execution provider `.so`s) are separate from the CUDA/cuDNN
  install above — extracted from a community-built Jetson wheel
  ([`ultralytics/assets`](https://github.com/ultralytics/assets/releases),
  onnxruntime-gpu 1.23.0 for JetPack 6/CUDA 12.6 — Microsoft's official PyPI
  package has no aarch64+Tegra build). Prebuilt wheels don't ship C/C++
  headers, so the 5 headers `onnxruntime_cxx_api.h` actually needs are
  pulled straight from the `microsoft/onnxruntime` GitHub repo at the
  matching `v1.23.0` tag.

**If you're on a different JetPack/L4T version than R36.4.7**, the apt
suite pinned in the Dockerfile (`r36.4`) needs to match — check
`cat /etc/nv_tegra_release` and `apt-cache policy cuda-cudart-12-6` on the
device and adjust the `r36.4` in the Dockerfile's Jetson CUDA block
accordingly.

**Status: reached real GPU inference on hardware, still debugging a
missing-library error.** This was iterated against an actual Jetson Orin
Nano (not simulated) through several rounds: first hit an OpenVINO
`device: GPU`-not-registered error (expected — that's the Intel-only plugin,
§4), then a `libcublasLt.so.12` load failure (from the wrong assumption
above — being fixed by baking CUDA in directly instead of relying on
`--runtime nvidia` mounts). Update this section once the corrected
Dockerfile has been confirmed working end-to-end.

## 6. Known gaps / things to verify on real hardware

- `device: CUDA` on Jetson (§5.6) is still being iterated against real
  hardware — status/next-fix noted at the end of §5.6.
- `device: CUDA` (both §5.5 and §5.6) only has a working ONNX export for
  `yolov5` — the model every shipped config actually uses. `yolo11`/`yolov8`
  don't have a CUDA path.
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
