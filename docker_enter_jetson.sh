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
# GTK/highgui windows -- without it, cv::imshow crashed a few seconds in
# with a BadAccess/MIT-SHM X error on this setup (container + this X server
# combination doesn't get along with X11 shared memory for some reason).

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
 
# ---------------------------------------------------------------------------
# X11 plumbing: local display (:0, :1) vs forwarded over SSH (localhost:10.0)
# ---------------------------------------------------------------------------
X11_ARGS=()
if [[ "$DISPLAY" == :* ]]; then
  # Local X server -- connection is a unix socket, so share it.
  X11_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
else
  # Forwarded X11 -- connection is TCP to the *host's* loopback, so the
  # container needs the host network namespace to reach it.
  X11_ARGS+=(--network host)
fi
 
# Rewrite the auth cookie to FamilyWild so it is accepted under the
# container's different hostname. Falls back quietly if xauth has nothing.
XAUTH_FILE="/tmp/.docker_${CONTAINER_NAME}.xauth"
if command -v xauth >/dev/null 2>&1; then
  rm -f "$XAUTH_FILE"
  touch "$XAUTH_FILE"
  if xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' \
       | xauth -f "$XAUTH_FILE" nmerge - 2>/dev/null; then
    chmod 644 "$XAUTH_FILE"
  fi
  if [ ! -s "$XAUTH_FILE" ]; then
    echo "Warning: no X11 cookie found for DISPLAY=$DISPLAY." >&2
    echo "GUI windows may fail to open; see JETSON_ORIN.md." >&2
  fi
else
  echo "Warning: xauth not installed -- X11 auth may fail." >&2
fi
X11_ARGS+=(-e XAUTHORITY="$XAUTH_FILE" -v "${XAUTH_FILE}:${XAUTH_FILE}:ro")
 
# ---------------------------------------------------------------------------
# Build the image if it isn't present
# ---------------------------------------------------------------------------
if ! sudo docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  if [ ! -f "${REPO_DIR}/Dockerfile" ]; then
    echo "Image '${IMAGE_NAME}' not found and no Dockerfile in ${REPO_DIR}." >&2
    echo "Cannot build. Check the repo is complete." >&2
    exit 1
  fi
  echo "Image '${IMAGE_NAME}' not found locally. Building it now."
  echo "This can take a while on the Orin -- go get a coffee."
  sudo docker build -t "${IMAGE_NAME}" "${REPO_DIR}"
  echo "Build complete."
fi
 
# ---------------------------------------------------------------------------
# Create / start / attach
# ---------------------------------------------------------------------------
if sudo docker ps -q -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Container '${CONTAINER_NAME}' already running, attaching..."
elif sudo docker ps -aq -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Starting existing (stopped) container '${CONTAINER_NAME}'..."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "Creating container '${CONTAINER_NAME}'..."
  sudo docker run -d --name "${CONTAINER_NAME}" \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    "${X11_ARGS[@]}" \
    -v "${REPO_DIR}:/root/sp_vision_25" \
    -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro \
    "${IMAGE_NAME}" \
    sleep infinity
fi
 
echo "Entering container. First time in a fresh container, build with:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
sudo docker exec -it "${CONTAINER_NAME}" bash

