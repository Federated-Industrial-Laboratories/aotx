# aotx

A local inference operating system designed in CUDA. The GPU holds the authoritative state:
agents, models, a message bus, a text interface and a command line. The disk holds a copy that
is one tick behind. The GPU never waits for the disk.

## Status

Version 0.0. The repository holds the seam: the device ring, the per-tick flush into a pinned
host ring, and the disk-side drain, feeder and restore programs. No release exists.

## Requirements

- CUDA Toolkit 13.2 or later, and a driver that supports it
- A GPU with compute capability 8.6
- CMake 3.28 or later, Ninja
- Python 3.10 or later, for the gates

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

## Gates

Run `tools/gate.sh` before each commit. It examines the staged files with three checks:
`tools/ste-lint.py` (text register), `tools/size-gate.py` (file size), `tools/seam-gate.py`
(host and device separation). Give paths to examine files that are not staged.

## Layout

```
cuda/    device modules, one directory for each module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
tests/   one test program for each module; each test runs at N=1 and N=64
docs/    reference documentation; start at docs/00-writing.md
tools/   the gates
```

## Where to start reading

Read `docs/00-writing.md` for the writing rules, then `cuda/seam/wire.h` for the layouts that
cross the seam.

## License

Apache License, Version 2.0. See LICENSE.
