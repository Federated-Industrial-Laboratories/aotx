<p align="center">
  <img src=".github/assets/mark.png" width="360" alt="The AOTX mark">
</p>

<h1 align="center">AOTX</h1>

<p align="center">A local inference operating system designed in CUDA.</p>

<p align="center">
  <a href="LICENSE"><img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Version 0.2.0" src="https://img.shields.io/badge/version-0.2.0-2ea44f">
  <img alt="CUDA 13.2" src="https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white">
  <img alt="Compute capability 8.0 and above" src="https://img.shields.io/badge/compute%20capability-8.0%2B-76B900">
</p>

<p align="center">
  <img alt="Languages C, CUDA and PTX" src="https://img.shields.io/badge/languages-C%20%7C%20CUDA%20%7C%20PTX-555555">
  <img alt="Profiles 8g, 12g, 24g and 48g" src="https://img.shields.io/badge/profiles-8g%20%7C%2012g%20%7C%2024g%20%7C%2048g-555555">
  <img alt="Platform Linux" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
</p>

---

Authoritative system state resides in GPU memory: agents, models, a message bus, a text
interface and a command line. The disk maintains a replica that lags by one tick. No GPU
operation blocks on disk input or output.

> [!IMPORTANT]
> The repository does not include model files. The model store fetches each file from its
> source and verifies its digest before use.

## Overview

AOTX is a Linux program that runs on a single NVIDIA GPU. It is not a Linux distribution, a
device driver or a remote inference service.

The system comprises agents, models, a module catalog, tools, a command line and two display
surfaces: a window and a terminal. A journal restores the authoritative state after a process
stop. The terminal can start, attach to and restore a system without a window.

| GPU memory (authoritative) | Disk (replica) |
| --- | --- |
| agents and their memory tiers | the journal, one tick behind |
| models and the module catalog | the model store and the module files |
| the message bus, the grid and the mirror | the transcripts and the bus file |

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

## Layout

<details>
<summary>The directories</summary>

```
cuda/    device modules, one directory per module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
modules/ the role modules imported at the start of a run
sdk/     the tool module contract and examples
share/   the model catalog and the terminal art
tests/   the checks; device batches use the slot count of the selected profile
docs/    the documentation; start at docs/00-writing.md
tools/   the gates
```

</details>

## Documentation

`docs/00-writing.md` indexes the documentation set and states its writing rules.
`docs/08-measured.md` reports the measured figures of this version.

---

<p align="center">Apache License, Version 2.0. See <a href="LICENSE">LICENSE</a>.</p>
