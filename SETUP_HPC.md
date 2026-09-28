# PanSim — HPC Setup Guide

This is a step-by-step guide for building and running PanSim on a shared HPC cluster,
based on a real setup session. It complements [README.md](README.md), which documents
the simulation itself and its CLI options — this file is only about getting a working
build and your first successful run.

## 1. Prerequisites

You need:
- CMake ≥ 3.15
- A C++17 compiler with OpenMP support (e.g. GCC ≥ 8)
- CUDA Toolkit, **only if building for GPU**, with a version compatible with the
  cluster's installed NVIDIA driver (see [GPU/driver compatibility](#5-gpudriver-compatibility) below)

Most clusters provide these via environment modules:

```bash
module avail cmake gcc cuda
module load cmake/<version> gcc/<version> cuda/<version>
```

Module names/versions vary a lot between clusters — always check `module avail` rather
than assuming a name from another system.

## 2. Get the submodules

PanSim depends on bundled third-party libraries (`thrust`, `cub`, `cxxopts`, `rapidjson`,
`libcudacxx`) checked out as git submodules. If you cloned with plain `git clone` (no
`--recursive`), or if you copied the repo instead of cloning it, these directories will
exist but be **empty**, which breaks the build (see [Troubleshooting](#troubleshooting)).

From inside the `pansim/` directory:

```bash
git submodule update --init --recursive
```

Sanity check that they're populated:

```bash
ls thrust/thrust/device_vector.h   # should exist, not error
du -sh thrust cub libcudacxx cxxopts rapidjson
```

## 3. Extract the example input data

The large example agent/location files are stored as `.tar` archives in git (the
plain `.json` files are gitignored to keep the repo small) and need to be extracted
once per checkout:

```bash
cd inputRealExample
tar -xf agents.tar
tar -xf locations.tar
cd ..
ls inputRealExample   # agents.json and locations.json should now exist
```

The default paths in `defaultargs.json` point here, so this step is required even for
the simplest test run.

## 4. Build

The `Makefile` wraps the CMake configure + build steps:

```bash
cd pansim
rm -rf build          # always start clean when changing GPU/CPU or toolchain modules
make buildCPU          # OpenMP-only build
# or
make buildGPU          # CUDA build
```

This is equivalent to:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON   # or OFF
make -j
```

The resulting binary is `build/panSim`.

**Important:** `make buildGPU` does *not* guarantee a GPU build. If CMake can't find
`nvcc`, it silently falls back to a CPU-only build and just prints a warning
(`Could not find nvcc, please set CUDAToolkit_ROOT.` / `Using CPU only`). Always check
the CMake configure output to confirm which mode it actually selected.

## 5. GPU/driver compatibility

Two independent things have to line up for a GPU build to actually run:

1. **Driver vs. toolkit version.** The CUDA driver on the node must support the CUDA
   runtime version the binary was compiled with, or you get a runtime error like:
   ```
   terminate called after throwing an instance of 'thrust::system::detail::bad_alloc'
     what():  std::bad_alloc: cudaErrorInsufficientDriver: CUDA driver version is
     insufficient for CUDA runtime version
   ```
   Check both sides:
   ```bash
   nvidia-smi            # top-right "CUDA Version: X.Y" = max supported by the driver
   nvcc --version         # toolkit version the binary is built against
   ```
   If `nvcc`'s version is newer than what `nvidia-smi` reports, `module load` an older
   `cuda/x.y` module (≤ the driver's max) and rebuild.

2. **GPU architecture / compute capability.** Older GPUs (e.g. Tesla K80, compute
   capability 3.7 / Kepler) are dropped entirely by CUDA 12.x toolkits — you need a
   CUDA 11.x toolkit to target them at all, no matter how new the driver is. Identify
   your GPU's compute capability from `nvidia-smi` (GPU name) and pass it explicitly
   rather than relying on `AUTO` detection, which can behave unreliably on old cards:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DARCHITECTURE=37   # e.g. K80 = 3.7
   ```

Also confirm you're actually building/running with a GPU allocated — many clusters'
login nodes have no GPU at all, so `nvidia-smi` there tells you nothing about compute
nodes. Get an interactive GPU allocation to test:

```bash
# SLURM example
srun --partition=gpu --gres=gpu:1 --pty nvidia-smi
salloc --partition=gpu --gres=gpu:1 --time=01:00:00
```

If your cluster has no matching CUDA toolkit available for your GPU's compute
capability, build `make buildCPU` instead — OpenMP with enough cores can be
competitive with an old GPU.

## 6. Run

Run from inside `pansim/` (all default paths in `defaultargs.json` are relative to it):

```bash
./build/panSim -w 12
```

To use a different/larger dataset (e.g. files staged one level above `pansim/`):

```bash
./build/panSim -w 12 \
  -a ../inputRealExample/agents1.json \
  -l ../inputRealExample/locations0.json \
  -A ../inputConfigFiles/agentTypes_3.json \
  -P ../inputConfigFiles/progressions_Jun17_tune/transition_config.json
```

For CPU/OpenMP runs, control thread count and pinning:

```bash
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=TRUE
numactl --cpunodebind=0 ./build/panSim ...
```

### Example SLURM batch script

```bash
#!/bin/bash
#SBATCH --job-name=pansim
#SBATCH --partition=gpu          # or a CPU partition for a buildCPU binary
#SBATCH --gres=gpu:1             # omit for CPU-only
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

module load cmake gcc cuda/11.4   # match what you built with
cd pansim
./build/panSim -w 12 -a ../inputRealExample/agents1.json -l ../inputRealExample/locations0.json
```

Build interactively first (via `salloc`/`srun`), rather than inside the batch script,
so you can see compiler/CMake errors live.

## 7. Extracting the output

PanSim's main output is a **tab-separated time series printed to stdout**: one header
line, then one row per timestep, with columns for disease states, hospitalizations
(`H`), tests (`T`), quarantine (`Q`/`QT`/`NQ`), mutation shares (`MUT`), stayed-home
count (`HOM`), vaccinations (`VAC`), new infections (`NI`), etc. (defined in
`include/simulation.h`).

Redirect it to a file:

```bash
./build/panSim -w 12 > results.tsv
```

To watch progress live while also saving it:

```bash
./build/panSim -w 12 | tee results.tsv
```

To keep diagnostics/warnings separate from the data:

```bash
./build/panSim -w 12 > results.tsv 2> run.log
```

Load it for analysis:

```python
import pandas as pd
df = pd.read_csv("results.tsv", sep="\t")
```

### Optional: per-agent detail

Pass `--outAgentStat <file>` to also dump a full per-agent JSON record (infection
time, diagnosis time, quarantine, hospitalization, worst disease state reached, etc.
— see `src/IO/outputs/agentStatOutput.cpp`). This is off by default (empty path):

```bash
./build/panSim -w 12 --outAgentStat agentstats.json > results.tsv
```

There are also `--dumpLocationInfections N`, `--dumpLocationInfectiousList <file>`,
and `--dumpLocationAgentList <file>` for per-location dumps at a given timestep
interval, if you need spatial detail rather than just aggregate counts.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find nvcc, please set CUDAToolkit_ROOT.` / `Using CPU only` during `make buildGPU` | No CUDA toolkit module loaded | `module load cuda/<version>`, `rm -rf build`, rebuild |
| `fatal error: thrust/device_vector.h: No such file or directory` | Git submodules not initialized (empty `thrust`/`cub`/etc. directories) | `git submodule update --init --recursive` |
| `inputRealExample/locations.json does not exists or cannot be opened` | Example data still in `.tar` form (gitignored `.json`) | `tar -xf` the archives in `inputRealExample/` (see [step 3](#3-extract-the-example-input-data)) |
| `cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version` | Binary built with a newer CUDA toolkit than the node's driver supports | `nvidia-smi` vs `nvcc --version`, load an older `cuda/x.y` module ≤ driver's max, rebuild |
| GPU build runs on login node but does nothing / errors immediately | No GPU allocated on that node | Get an interactive GPU allocation (`srun`/`salloc --gres=gpu:1`) before running |
| Old GPU (e.g. Tesla K80, Kepler) not supported however new the driver is | CUDA 12.x dropped Kepler (compute capability < 5.0) support entirely | Use a CUDA 11.x toolkit and set `-DARCHITECTURE=<compute capability>` explicitly (e.g. `37` for K80) |
