# CUDA N-Body Simulation

Gravitational N-body simulation accelerated on the GPU with CUDA, taking a
50,000-body simulation from ~26 minutes on CPU to ~1.5 seconds on a T4.

![galaxy](galaxy.gif)

## Benchmark (50,000 bodies, 100 steps, T4 GPU)

| Version              | Time         | Speedup vs CPU |
|----------------------|--------------|----------------|
| CPU (single-thread)  | 1,545,927 ms | 1×             |
| Naive CUDA           | 2,003 ms     | ~772×          |
| + Shared-mem tiling  | 1,500 ms     | ~1,030×        |
| + SoA layout         | 1,432 ms     | ~1,080×        |

## What I learned
- Naive parallelism (one thread per body) delivered the biggest win — ~772× — just from doing 50k independent force calculations at once instead of in sequence.
- Shared-memory tiling and an AoS→SoA layout gave only single-digit-to-25% further gains. Profiling that told me the kernel is **compute-bound** on `rsqrtf`, not memory-bound — so memory optimizations had limited headroom.
- Simulation stability required matching each body's orbital velocity to the softened gravity and using a small timestep; otherwise the disk gains energy and flies apart.

## Files
- `nbody_cpu.cpp` — CPU baseline
- `nbody.cu` — naive CUDA kernel
- `nbody_tiled.cu` — shared-memory tiled version
- `nbody_soa.cu` — structure-of-arrays layout
- `nbody_render.cu` — disk-galaxy renderer (writes frame data)
- `render.py` — reads `frames.bin` and renders `galaxy.gif`

## Run it on Colab

No local CUDA setup needed — [Google Colab](https://colab.research.google.com)
gives you a free GPU (usually a T4, matching the benchmarks above) with `nvcc` and
numpy/matplotlib/Pillow already installed. Run each block below in its own cell.

**1. Enable the GPU:** Runtime → Change runtime type → Hardware accelerator → GPU → Save. Then verify:
```python
!nvidia-smi
```

**2. Clone the repo:**
```python
!git clone https://github.com/henry-tran07/cuda-nbody.git
%cd cuda-nbody
```
(Use the `%cd` magic, not `!cd` — it persists the working directory across cells.)

**3. Compile and run the simulation** (writes `frames.bin`):
```python
!nvcc -O3 nbody_render.cu -o nbody_render
!./nbody_render
```

**4. Render the GIF** (reads `frames.bin`, writes `galaxy.gif`):
```python
!python3 render.py
```

**5. View / download the result:**
```python
from IPython.display import Image
Image("galaxy.gif")
```
```python
from google.colab import files
files.download("galaxy.gif")
```

No `pip install` needed — the Python dependencies ship with Colab. To change the
galaxy, tweak the parameters in `nbody_render.cu` (`n`, `steps`, `every`, `dt`,
`soft`, `Mcenter`) and re-run steps 3–4; `render.py` reads `n` and the frame count
from the file, so it adapts automatically.
