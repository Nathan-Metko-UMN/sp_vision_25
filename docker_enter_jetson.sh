#!/bin/bash
# Enter (creating if needed) the sp_vision_25 dev container on a Jetson Orin,
# with GPU access and a display connected to the Jetson's own X server.
#
# Usage: ./docker_enter_jetson.sh
# Run from the repo root (or anywhere -- it resolves paths relative to this
# script's own location, not the current directory).
#
# X11 auth uses the host's .Xauthority cookie (mounted into the container),
# not `xhost` -- xhost grants blanket local-root access that has to be
# re-run after every reboot; mounting .Xauthority just extends your existing
# host session's real credentials into the container, so nothing extra needs
# granting. See JETSON_ORIN.md for the `xhost`-based alternative if this
# doesn't work in your setup (e.g. no .Xauthority file, or DISPLAY not set).
#
# The container is created once and reused (not --rm) so that the compiled
# build/ directory and cached TensorRT engine (assets/yolov5.engine) persist
# across sessions instead of rebuilding from scratch every time.
#
# The ENTIRE repo is bind-mounted (not just configs/logs/assets) so the
# container always sees whatever's actually on the host's disk right now --
# git pull, branch switches, local edits, all show up immediately with no
# rebuild or manual file-copying needed. Same pattern the Windows/x86_64
# devcontainer.json already uses. Without this, only the source tree that
# existed at whatever moment `docker build` last ran stays baked into the
# image, silently diverging from the host repo -- easy to not notice.
#
# QT_X11_NO_MITSHM=1 disables the X MIT-SHM (shared memory) extension for
# Qt's own windows, but NOT GTK's (OpenCV highgui here uses the GTK backend,
# per the "Failed to load module canberra-gtk-module" log line) -- GTK can
# still crash with "BadShmSeg (invalid shared segment parameter)" without
# --ipc=host. Root cause: containers get their own IPC namespace by default,
# so SysV/POSIX shared-memory segments a process creates *inside* the
# container for MIT-SHM XImages aren't visible to the X server running on
# the host, outside that namespace -- the segment ID the client references
# simply doesn't exist from the server's point of view. --ipc=host shares
# the host's IPC namespace instead, fixing this at the root (matches how
# NVIDIA's own reference ROS container commands set up X11 too).

set -e

CONTAINER_NAME="sp_vision_jetson"
IMAGE_NAME="sp_vision_25"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$DISPLAY" ]; then
  echo "DISPLAY is not set in this shell." >&2
  echo "If you're SSH'd in and want the window on the Jetson's own monitor" >&2
  echo "(not this SSH session), set it explicitly first, e.g.:" >&2
  echo "  export DISPLAY=:1" >&2
  exit 1
fi

if [ ! -f "$HOME/.Xauthority" ]; then
  echo "Warning: $HOME/.Xauthority not found -- X11 auth may fail." >&2
  echo "Falling back to 'xhost +local:docker' may be needed instead; see JETSON_ORIN.md." >&2
fi

if sudo docker ps -q -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Container '${CONTAINER_NAME}' already running, attaching..."
elif sudo docker ps -aq -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Starting existing (stopped) container '${CONTAINER_NAME}'..."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "Creating container '${CONTAINER_NAME}'..."
  sudo docker run -d --name "${CONTAINER_NAME}" \
    --runtime nvidia \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY}" \
    -e XAUTHORITY=/root/.Xauthority \
    -e QT_X11_NO_MITSHM=1 \
    -v "${HOME}/.Xauthority:/root/.Xauthority:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${REPO_DIR}:/root/sp_vision_25" \
    -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro \
    "${IMAGE_NAME}" \
    sleep infinity
fi

echo "Entering container. First time in a fresh container, build with:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
sudo docker exec -it "${CONTAINER_NAME}" bash
