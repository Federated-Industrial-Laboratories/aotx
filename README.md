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

## Tests

`ctest --test-dir build` runs every check that needs no desktop. The interface is part of
it. The check `raster_headless` makes its drawing context on a display device. It then runs
the raster graph for 60 frames through the pixel buffer path. It compares the pixels with
the figure the host computes. No window opens.

Two groups of checks are registered only when a build asks for them:

```
cmake -S . -B build -G Ninja -DAOTX_DISPLAY_TESTS=ON   # the window check, label display
cmake -S . -B build -G Ninja -DAOTX_FAULT_TESTS=ON     # the guard gap checks
```

The window check opens a window on the display of the operator; run it alone with
`ctest --test-dir build -L display`. The same program sends the close request of a window
manager to a window of a title. A script stops a run with a window that way:

```
build/aotx_window_test --close AOTX-1
```

No check and no tool of this repository destroys or kills the window of another program.

## Load runs

A run with a tick load writes many records for each tick. The drain makes a line of text for
every record of the types it derives. At 12,000 records a tick, those lines fill a disk in
minutes. Give `--derive` to name the types the drain makes lines from:

```
aotx_boot --journal build/run --workload 12000 --derive console,bus
```

The names are `console`, `note`, `bus`, `bulk` and `none`, with commas between them. A run
that gives no list leaves the drain with its default, which is every type. The journal keeps
every record, whatever the list holds; the list changes the derived files only.

A short list makes the drain faster, and a faster drain lets the tick load make more
records. Measured on one machine at `--workload 12000` for 30 seconds:

| derive | journal | derived lines | ticks held |
| --- | --- | --- | --- |
| default | 755 MB | 482 MB | 2851 of 2944 |
| `console,bus` | 6.4 GB | 0 | 730 of 2943 |

Give `--records` or a smaller `--workload` to bound the size of a journal. The option
`--derive` bounds the derived lines only.

A run stops at the `quit` command, at the close request of the window manager, and at the
signals SIGTERM and SIGINT. Each of them ends the run the same way: the last flush, the
closed rings, the wait for the disk side programs, and the reports. A second signal changes
nothing, because the run is already stopping. An operator who must end a run that stopped
answering sends SIGKILL, and knows what that leaves behind.

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
