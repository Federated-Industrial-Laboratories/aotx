# aotx

A local inference operating system designed in CUDA. The GPU holds the authoritative state:
agents, models, a message bus, a text interface and a command line. The disk holds a copy that
is one tick behind. The GPU never waits for the disk.

## Status

Version 0.1.0. This version holds the seam, the tick graph, the six panels and the command
line. It holds the model file reader, the tokenizer of the device, the matrix kernels and the
forward pass. It decodes through the tick graph, and it runs agents with their roles and
their tools. A run writes a journal, and a restore gives the state back from that journal. The
gates and 35 checks ship with the product.

## Requirements

- CUDA Toolkit 13.2 or later, and a driver that supports it
- A GPU of compute capability 8.6
- CMake 3.28 or later, and Ninja
- Python 3, for the gates
- GLFW 3, GLEW, OpenGL, EGL and X11, for the window
- Linux, for the disk side

`docs/06-build.md` states every requirement.

## Build and test

```
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash tools/profile-detect.sh
cmake -S . -B build -G Ninja -DAOTX_PROFILE=12g -DAOTX_ARCH=86
cmake --build build
ctest --test-dir build
```

Run the gates before a commit:

```
tools/gate.sh
```

## Run

A model file is large and is not in the repository. Put the files in one directory with the
manifest that `docs/06-build.md` describes. Then start a run with a language model and a
window:

```
build/aotx_boot --journal build/run --models models --roles language --window
```

Type one line at the console:

```
say what is a tick
```

The conductor agent answers, and the reply grows one console line as the tokens come. The
document `docs/07-operation.md` states every option, every command and every file a run
leaves.

## Layout

```
cuda/    device modules, one directory for each module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
tests/   one test program for each module; each test runs at N=1 and N=64
docs/    the documentation; start at docs/00-writing.md
tools/   the gates
```

## Where to start reading

Read `docs/00-writing.md` for the writing rules. Read `docs/01-architecture.md` next, for the
shape of the system.

- `docs/00-writing.md`: the register that every line of this repository follows.
- `docs/01-architecture.md`: the modules, the boundary they stand on, and the two graphs.
- `docs/02-temporal-model.md`: the device ahead and the disk behind, the record classes, and
  what a restore gives back.
- `docs/03-seam-contract.md`: the byte layouts, the publication protocol and the attach
  procedure of the rings.
- `docs/04-journal-format.md`: the segments a run writes, and the files the drain derives.
- `docs/05-bus-schema.md`: the message on the device, and the line on the disk.
- `docs/06-build.md`: the requirements, the build options, the checks and the gates.
- `docs/07-operation.md`: the options of a run, the window, the commands and the journal.
- `docs/08-measured.md`: the rates, ticks and lags of one machine at one commit.

## License

Apache License, Version 2.0. See LICENSE.
