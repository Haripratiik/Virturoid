#!/bin/bash
# Build Madrona-MJX (the GPU batch renderer for at-scale vision RL) on a NATIVE-LINUX GPU box, fully non-sudo.
#
# This recipe is PROVEN: it built cleanly end-to-end (_madrona_mjx_batch_renderer.so, `import madrona_mjx` works)
# on the project's WSL2 box. The ONLY thing WSL2 couldn't do is RENDER -- Madrona's MWGPU GPU-ECS engine needs
# CUDA concurrent managed (unified) memory, and WSL2 reports CONCURRENT_MANAGED_ACCESS=0 (checked via
# cuDeviceGetAttribute). On NATIVE Linux that attribute is 1, so the renderer runs. Use a cloud GPU
# (Lambda/RunPod/AWS) or a dual-boot Linux. See memory: vision-cv-decision.
#
# Why each step (the blockers cleared, all non-sudo):
#   - LOAD_VULKAN=OFF        : use the CUDA software ray-tracer, so the X11/mesa libs (sudo) are not needed
#   - uv standalone python   : brings its own Python.h, so the find_package(Python Development.Module) gate
#                              passes without `sudo apt install python3.X-dev`
#   - CUDA 12.5 toolkit       : Madrona needs nvcc <=12.5.1; the pip nvidia-cuda-nvcc wheel is RUNTIME-only
#                              (ships ptxas, not the nvcc driver), so install the real toolkit to a user dir
#   - device_get patch        : renderer.py passes mat_tex_ids/cam_fovy as jax arrays; the C++ wants numpy
set -euo pipefail

CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run"
CU="$HOME/cuda125"
MJX_DIR="$HOME/madrona_mjx"
ENV_DIR="$HOME/madrona_env"

echo "[1/6] build tools (pip, non-sudo)"
pip install -q -U uv cmake ninja nanobind
UVPY="$(uv python install 3.12 >/dev/null 2>&1; uv python find 3.12)"   # standalone python3.12 WITH headers
echo "      python(+headers): $UVPY"

echo "[2/6] CUDA 12.5 toolkit -> $CU (non-sudo, ~4.4GB)"
if [ ! -x "$CU/bin/nvcc" ]; then
  wget --tries=5 --timeout=120 "$CUDA_URL" -O "$HOME/cuda125.run"
  sh "$HOME/cuda125.run" --toolkit --silent --installpath="$CU" --no-opengl-libs --no-man-page --override
fi
export PATH="$CU/bin:$PATH" CUDA_HOME="$CU" LD_LIBRARY_PATH="$CU/lib64:${LD_LIBRARY_PATH:-}"
"$CU/bin/nvcc" --version | grep release

echo "[3/6] clone Madrona-MJX (+ submodules, full clone -- NOT --depth 1)"
[ -d "$MJX_DIR" ] || git clone https://github.com/shacklettbp/madrona_mjx.git "$MJX_DIR"
cd "$MJX_DIR" && git submodule update --init --recursive

echo "[4/6] patch renderer.py (device_get mat_tex_ids/cam_fovy -> numpy for the C++ binding)"
sed -i 's/mat_tex_ids = m.mat_texid/mat_tex_ids = jax.device_get(m.mat_texid)/' src/madrona_mjx/renderer.py
sed -i 's/cam_fovy=m.cam_fovy/cam_fovy=jax.device_get(m.cam_fovy)/' src/madrona_mjx/renderer.py

echo "[5/6] configure + build (CUDA software ray-tracer)"
mkdir -p build && cd build && rm -rf CMakeCache.txt CMakeFiles
cmake -DLOAD_VULKAN=OFF -DPython_EXECUTABLE="$UVPY" \
      -DCUDAToolkit_ROOT="$CU" -DCMAKE_CUDA_COMPILER="$CU/bin/nvcc" ..
make -j"$(nproc)"
ls -la ./_madrona_mjx_batch_renderer*.so

echo "[6/6] isolated jax<0.6 venv + install + import smoke"
"$UVPY" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install -q -U pip "jax[cuda12]<0.6.0" "mujoco>=3.2.7" mujoco-mjx nanobind
cd "$MJX_DIR" && "$ENV_DIR/bin/pip" install -q -e .
"$ENV_DIR/bin/python" -c "import madrona_mjx; print('MADRONA OK', madrona_mjx.__file__)"

echo "DONE. On NATIVE Linux, render with the BatchRenderer (see scripts/debug.py); on WSL2 it aborts with"
echo "CUDA_ERROR_INVALID_DEVICE (no concurrent managed access)."
