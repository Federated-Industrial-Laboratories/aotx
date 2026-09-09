<p align="center">
  <img src=".github/assets/mark.png" width="720" alt="AOTX-1, Ahead Of Time eXecutive">
</p>

<p align="center">A local inference operating system designed in CUDA.</p>

<p align="center">
  <a href="LICENSE"><img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Version 0.3.0" src="https://img.shields.io/badge/version-0.3.0-2ea44f">
  <img alt="CUDA 13.2" src="https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white">
  <img alt="Compute capability 8.0 and above" src="https://img.shields.io/badge/compute%20capability-8.0%2B-76B900">
</p>

<p align="center">
  <img alt="Languages C, C++, CUDA and PTX" src="https://img.shields.io/badge/languages-C%20%7C%20C%2B%2B%20%7C%20CUDA%20%7C%20PTX-555555">
  <img alt="Profiles 8g, 12g, 24g and 48g" src="https://img.shields.io/badge/profiles-8g%20%7C%2012g%20%7C%2024g%20%7C%2048g-555555">
  <img alt="Platform Linux" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
</p>

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

Authoritative system state resides in GPU memory: agents, models, a message bus, a text
interface and a command line. The disk maintains an asynchronous journal replica.

Each completed tick is copied to the host ring. The drain writes and synchronizes the journal.
Disk lag depends on the drain. Backpressure can hold a tick when the ring does not have enough room.
GPU kernels do not wait on disk input or output.

> [!IMPORTANT]
> The repository does not include model files. The model store fetches each file from its
> source and verifies its digest before use.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Overview

AOTX is a Linux program that runs on a single NVIDIA GPU. It is not a Linux distribution, a
device driver or a remote inference service.

The system comprises agents, models, a module catalog, tools, a command line and two display
surfaces: a window and a terminal. A journal restores the authoritative state after a process
stop. The terminal can start, attach to and restore a system without a window.

| GPU memory (authoritative) | Disk (replica) |
| --- | --- |
| agents and their memory tiers | the journal, written asynchronously |
| models and the module catalog | the model store and the module files |
| the message bus, the grid and the mirror | the transcripts and the bus file |

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Terms

The documentation uses a small set of project terms. Each one names a standard mechanism.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| ring | a single-producer, single-consumer ring buffer in pinned host memory |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| drain | the disk-side process that writes the outbound ring to the journal (a log writer) |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| catalog | the GPU-resident registry of imported modules: skills, roles and tools |
| profile | a build-time table-size configuration for one class of card |
| bus | an append-only message log between agents (a message bus) |
| arena | a contiguous memory region for offset-addressed allocations |
| pump | the host glue that launches the device scheduling graph once per tick |

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Requirements

<details>
<summary>The list</summary>

- CUDA Toolkit 13.2 or later, and a driver that supports it
- A GPU of compute capability 8.0 or above; 8.6 is the reference capability
- CMake 3.28 or later, and Ninja
- Python 3, for the gates
- GLFW 3, GLEW, OpenGL, EGL and X11, for the window
- libcurl, for the model fetch (optional)
- Linux, for the disk side

`docs/06-build.md` specifies every requirement.

</details>

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Install, build and test

Install the requirements in `docs/06-build.md`. The detector then reports the profile and the
architecture of the installed card:

```
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash tools/profile-detect.sh
```

> [!NOTE]
> A profile fixes the table sizes for the card at build time. The profiles are `8g`, `12g`,
> `24g` and `48g`.

Pass the two reported values to CMake. The example below is for the reference card:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_PROFILE=12g -DAOTX_ARCH=86
cmake --build build
ctest --test-dir build
```

Run the gates:

```
tools/gate.sh
```

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Run

A settings file contains one `key = value` pair per line. The example below starts the
terminal and uses a model store below the repository:

```
# aotx.settings
journal.dir = build/run
models.dir = models
models.roles = language
tui.on = 1
```

List the model catalog, fetch one file and activate it for its catalog role:

```
build/aotx_models --dir models list
build/aotx_models --dir models fetch language
build/aotx_models --dir models activate language language
```

Start the system. With `tui.on` set, the terminal starts with it:

```
build/aotx_boot --settings aotx.settings
```

Enter `say what is a tick` at the console. The reply of the conductor agent appears on the
console. `import <path>` installs a skill directory. `docs/07-operation.md` documents the
complete start, model, module, conversation and restore procedures.

A second terminal attaches to the running system with:

```
build/aotx_tui --attach build/run --settings aotx.settings
```

The graphical control program `build/aotx_ctrl` starts, attaches to and stops systems in
windows, and `docs/13-control.md` documents it.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Layout

<details>
<summary>The directories</summary>

```
cuda/    device modules, one directory per module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
ctrl/    the graphical control program; C++ with the vendored ImGui sources
modules/ the role modules imported at the start of a run
sdk/     the tool module contract and examples
share/   the model catalog and the terminal art
tests/   the checks; device batches use the slot count of the selected profile
docs/    the documentation; start at docs/00-writing.md
tools/   the gates
```

</details>

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Documentation

`docs/00-writing.md` indexes the documentation set and states its writing rules.
`docs/08-measured.md` reports measurements from the named earlier versions.
See [Model files](docs/16-model-files.md) to inspect a file and prepare its model store.
See [Tool selection](docs/09-modules.md#tool-selection) for instance defaults and conversation choices.

Optional [live memory](docs/20-live-memory.md) binds fresh conversations to a typed GPU store.
Prompts use selected memory and current input. Unbound conversations keep their existing transcript policy.

The GPU prepares [text query vectors](docs/21-text-memory.md).
Each input has a 192-byte limit.

The [CCIR profile](docs/17-ccir.md) stores bounded typed state.
It is not a complete runtime package.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

<p align="center">Apache License, Version 2.0. See <a href="LICENSE">LICENSE</a>.</p>
