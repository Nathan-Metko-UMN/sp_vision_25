# sp_vision_25 build environment.
#
# Builds natively for whatever architecture it's run on: pulls the matching
# OpenVINO 2024.6.0 archive (the version hard-coded in this repo's
# CMakeLists files) and compiles the project against apt-installed deps.
#
# NOTE: Intel does not publish an "ubuntu22" *arm64* archive for OpenVINO
# 2024.6.0 (only ubuntu18/ubuntu20 arm64 builds exist for this version) even
# though this image's base OS is Ubuntu 22.04. On aarch64 we therefore pull
# the ubuntu20 arm64 archive instead -- it works fine on Ubuntu 22.04 because
# OpenVINO bundles its own runtime libs and Ubuntu 22.04's glibc/libstdc++
# are backwards-compatible with binaries built against Ubuntu 20.04's older
# ones. See JETSON_ORIN.md for details.
#
# On a Jetson Orin (Ubuntu 22.04 aarch64, JetPack 6): build directly on the
# device with `docker build .` — no cross-compilation needed. See
# JETSON_ORIN.md for why OpenVINO's GPU plugin does NOT work on Jetson's GPU
# (it's Intel-only) — set `device: CPU` in whatever config you deploy there.
#
# ROS2-gated targets (sentry*, publish_test, subscribe_test, topic_loop_test)
# are skipped since ROS2 isn't installed here — see JETSON_ORIN.md §5.3 to add it.
#
# `device: GPU` in the yaml configs means Intel's GPU plugin, which needs
# Intel's actual compute-runtime (intel-opencl-icd / intel-level-zero-gpu),
# not just the generic ICD loader that OpenVINO's own dependency script
# installs. That runtime is only meaningful on x86_64 (Jetson's GPU is
# NVIDIA, unsupported by this plugin regardless), so it's only installed
# there. At *container run time* you still need to give the container access
# to the host GPU:
#   - native Linux host: `docker run --device=/dev/dri ...`
#   - Windows host via Docker Desktop/WSL2: `docker run --device=/dev/dxg
#     -v /usr/lib/wsl:/usr/lib/wsl ...` (see JETSON_ORIN.md appendix)
#
# `device: CUDA` is a second, independent GPU backend for NVIDIA GPUs, since
# OpenVINO's GPU plugin above can never use one. It runs YOLOV5 through ONNX
# Runtime's CUDA execution provider instead of OpenVINO, using a .onnx
# sibling of assets/yolov5.xml (same weights, converted offline -- see
# JETSON_ORIN.md). x86_64 only here too: this installs a generic x86_64 CUDA
# runtime, which is NOT what Jetson needs (Jetson requires JetPack/L4T's own
# CUDA build tied to its embedded driver) -- so this path is for the NVIDIA
# GPU on an x86_64 dev machine, not Jetson. At container run time NVIDIA
# access on Windows/WSL2 comes through the same `--device=/dev/dxg
# -v /usr/lib/wsl:/usr/lib/wsl` as the Intel path above (Docker Desktop
# mirrors both vendors' driver shims there); on native Linux use `--gpus all`
# (needs the NVIDIA Container Toolkit installed on the host).

FROM ubuntu:22.04

ARG OPENVINO_VERSION=2024.6.0
ARG OPENVINO_SERIES=2024.6
ARG OPENVINO_BUILD=17404.4c0f47d2335
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        g++ \
        cmake \
        make \
        wget \
        ca-certificates \
        pkg-config \
        can-utils \
        libopencv-dev \
        libfmt-dev \
        libeigen3-dev \
        libspdlog-dev \
        libyaml-cpp-dev \
        libusb-1.0-0-dev \
        nlohmann-json3-dev \
        libceres-dev \
        libgl1 \
        libgomp1 \
        openssh-server \
        screen \
    && rm -rf /var/lib/apt/lists/*

# --- OpenVINO 2024.6.0 runtime (matches the version hard-coded in CMakeLists.txt) ---
# Picks the arm64 (ubuntu20 build, see note above) archive on Jetson/aarch64
# hosts, x86_64 (ubuntu22 build) archive otherwise.
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
        aarch64) ov_arch=arm64; ov_os=ubuntu20 ;; \
        x86_64)  ov_arch=x86_64; ov_os=ubuntu22 ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_SERIES}/linux/l_openvino_toolkit_${ov_os}_${OPENVINO_VERSION}.${OPENVINO_BUILD}_${ov_arch}.tgz"; \
    wget -q "$url" -O /tmp/openvino.tgz; \
    mkdir -p "/opt/intel/openvino_${OPENVINO_VERSION}"; \
    tar -xf /tmp/openvino.tgz -C "/opt/intel/openvino_${OPENVINO_VERSION}" --strip-components=1; \
    rm /tmp/openvino.tgz; \
    ln -s "/opt/intel/openvino_${OPENVINO_VERSION}" /opt/intel/openvino; \
    if [ -x "/opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh" ]; then \
        /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh -y || true; \
    fi

# --- Intel GPU compute runtime (needed for `device: GPU` in the yaml configs) ---
# x86_64 only: the OpenVINO GPU plugin only ever talks to an Intel GPU, and
# Jetson's GPU is NVIDIA, so there is nothing to install on aarch64.
RUN set -eux; \
    if [ "$(uname -m)" = "x86_64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gnupg gpg-agent wget ca-certificates; \
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
            | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
        echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client' \
            > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero clinfo; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# --- NVIDIA CUDA runtime + ONNX Runtime GPU (needed for `device: CUDA`) ---
# x86_64 only -- see note above. cuda-cudart/cublas/cufft/curand/cusparse/
# cusolver + cudnn9 + nvrtc are the minimal runtime pieces ONNX Runtime's
# CUDA execution provider needs (no nvcc/full toolkit required). Pinned to
# CUDA 12.6 + cuDNN 9, the combination ONNX Runtime 1.20.x's GPU build
# expects; the container only needs a driver new enough to run *some* CUDA
# 12.x, which is what NVIDIA driver backward compatibility guarantees.
ARG ONNXRUNTIME_VERSION=1.20.1
RUN set -eux; \
    if [ "$(uname -m)" = "x86_64" ]; then \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm /tmp/cuda-keyring.deb; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-cudart-12-6 cuda-nvrtc-12-6 libcublas-12-6 libcufft-12-6 \
            libcurand-12-6 libcusparse-12-6 libcusolver-12-6 libcudnn9-cuda-12; \
        rm -rf /var/lib/apt/lists/*; \
        url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"; \
        wget -q "$url" -O /tmp/onnxruntime.tgz; \
        mkdir -p /opt/onnxruntime; \
        tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1; \
        rm /tmp/onnxruntime.tgz; \
        echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf; \
        ldconfig; \
    fi

# libcuda.so itself (the actual GPU driver, as opposed to the CUDA *runtime*
# libs installed above) is never baked into the image -- on native Linux it
# comes from the host's NVIDIA driver install (standard ldconfig path via
# the NVIDIA Container Toolkit), but on Windows/WSL2 it only exists inside
# the /usr/lib/wsl/lib bind mount (see note above), which isn't in the
# dynamic linker's cache since it's not present at build time. Harmless
# no-op on hosts where that path doesn't exist.
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

WORKDIR /root/sp_vision_25
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

CMD ["/bin/bash"]
