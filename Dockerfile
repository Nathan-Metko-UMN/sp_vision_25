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

WORKDIR /root/sp_vision_25
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

CMD ["/bin/bash"]
