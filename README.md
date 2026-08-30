# aotx

A local inference operating system designed in CUDA. The GPU holds the authoritative state:
agents, models, a message bus, a text interface and a command line. The disk holds a copy that
is one tick behind. The GPU never waits for the disk.

## Status

Version 0.2.0. AOTX is a local inference operating system that runs as a Linux program.
It is not a Linux distribution, a device driver or a remote inference service. The repository
does not include model files.

The system holds agents, models, a module catalog, tools, a command line and two display
surfaces. A journal restores the authoritative state after a stopped process. A terminal can
start, attach to and restore a system without a window.

## Requirements

- CUDA Toolkit 13.2 or later, and a driver that supports it
- A GPU of compute capability 8.0 or above; 8.6 is the reference capability
- CMake 3.28 or later, and Ninja
- Python 3, for the gates
- GLFW 3, GLEW, OpenGL, EGL and X11, for the window
- Linux, for the disk side

`docs/06-build.md` states every requirement.

## Install, build and test

Install the requirements in `docs/06-build.md`. Then ask the detector for the profile and
architecture of the card:

```
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash tools/profile-detect.sh
```

Use the two values that it prints. This example is for the reference card:

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

A settings file gives one `key = value` on each line. This example starts the terminal and
uses the model store below the repository:

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

Start the system. The `tui.on` setting starts its terminal:

```
build/aotx_boot --settings aotx.settings
```

Type `say what is a tick` at the console. The conductor agent writes its reply there.
Use `import <path>` to install a skill directory. `docs/07-operation.md` gives the complete
start, model, module, conversation and restore procedures.

To attach another terminal to the system, run:

```
build/aotx_tui --attach build/run --settings aotx.settings
```

## Layout

```
cuda/    device modules, one directory for each module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
modules/ the role modules that a run imports at the start
sdk/     the tool module contract and examples
share/   the model catalog and terminal art
tests/   the checks; device batches use the count of the selected profile
docs/    the documentation; start at docs/00-writing.md
tools/   the gates
```

## Where to start reading

Read `docs/00-writing.md` first. It indexes the complete documentation set and gives its
writing rules.

## License

Apache License, Version 2.0. See LICENSE.
