# Build

This document states what the build needs and how to run it. It then states what each build
option registers, what each check covers, and what each gate refuses. The last section
states how the model files are recorded.

## Requirements

- CUDA Toolkit 13.2 or later. The build takes `nvcc` from the toolkit. Put the `bin`
  directory of the toolkit on `PATH`, such as `/usr/local/cuda-13.2/bin`.
- A driver that supports the toolkit. The build links the driver library `libcuda`.
- A GPU of compute capability 8.0 or above. The reference card is 8.6. The build compiles
  device code for the architecture that `AOTX_ARCH` names, 86 unless given. It carries the
  PTX of that architecture too, so a newer card runs it through the driver. Build for the
  card itself when you can (the section below). A card below 8.0 is not supported.
- CMake 3.28 or later, and Ninja.
- A C compiler for C11, and a C++ compiler for C++17.
- Python 3, for the gates.
- pkg-config, GLFW 3, GLEW and OpenGL, for the window.
- EGL, for the raster check that opens no window.
- X11, for the close request that the window check sends.
- A thread library. Each check program links it.
- Linux, for the disk side. The rings are memfd files, and the disk side builds with
  `_GNU_SOURCE`.

The disk side is C with no CUDA dependency. On an x86_64 machine the build adds a second
checksum path in one file, built with SSE 4.2. No other file takes that instruction set.

## Configure and build

```
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Every program of the build lands in the build directory itself. A program therefore finds
its siblings beside it, which is how the boot program starts the disk-side programs.

## The profile and the architecture

Two options fix the build for a card. `AOTX_PROFILE` selects the profile. The profile fixes every
figure that sizes a device table. Those are the slots, the ring sizes, the key value range,
the prompt bytes and the weights region. `AOTX_ARCH` selects the architecture the `.cu` files are built
for. `tools/profile-detect.sh` reads the card in the machine and prints the two options it
proposes.

| profile | slots | weights region | key value range | default language file | status |
| --- | --- | --- | --- | --- | --- |
| `8g` | 32 | 5 GB | 1 GB | `language-q4` | measured on the reference card |
| `12g` | 64 | 8 GB | 2 GB | `language` | the reference; measured |
| `24g` | 128 | 16 GB | 8 GB | `language` | built; the figures are estimates |
| `48g` | 256 | 40 GB | 24 GB | `language` | built; the figures are estimates |

The default is `12g` with `AOTX_ARCH` 86. A profile header (`cuda/profile/<name>.cuh`)
states its status in its banner. The checks take their batch size from the profile, so the
same list runs at 32 slots on `8g` and at 64 on `12g`. A boot on a card that cannot hold
the profile refuses with the figures and names the profile that fits (`docs/07-operation.md`).

```
bash tools/profile-detect.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_PROFILE=8g -DAOTX_ARCH=86
```

## Build options

Three options register checks that the default build leaves out. Two cache values name a
directory and a program.

| option | default | what it does |
| --- | --- | --- |
| `AOTX_PROFILE` | `12g` | the build profile: `8g`, `12g`, `24g` or `48g` |
| `AOTX_ARCH` | 86 | the compute architecture of the `.cu` files, as `sm_<n>` |
| `AOTX_DISPLAY_TESTS` | OFF | the check `window`, with the label `display` |
| `AOTX_FAULT_TESTS` | OFF | the checks `mem_fault` and `kvcache_fault` |
| `AOTX_SANITIZER_TESTS` | OFF | the checks `sanitizer_memcheck` and `sanitizer_racecheck`, with the label `sanitizer` |
| `AOTX_MODELS_DIR` | `models` | the directory the checks read the model files from |
| `AOTX_BUS_LINT` | empty | a program that validates a bus line file, for `disk_drain` and `disk_derive` |

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_DISPLAY_TESTS=ON
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_FAULT_TESTS=ON
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_SANITIZER_TESTS=ON
```

The window check opens a window on the display of the operator. Run it alone with
`ctest --test-dir build -L display`. The fault checks write past a guard gap on purpose,
which kills a context on the display device. They run serially, with `AOTX_FAULT_TESTS` set
in their environment. The sanitizer checks run serially with a time allowance of 7,200
seconds; run them alone with `ctest --test-dir build -L sanitizer`.

## The checks

`ctest --test-dir build` runs the 64 checks that the default Release build registers. A check that
reads a model file reports a skip when the file is not there, and the skip is a figure of
the report.

| check | what it covers |
| --- | --- |
| `clock_module` | the boot program with `--clock-only`; the clock module gives a sample |
| `parity` | each terminal action and static Bus word against the command parser |
| `parity_refuse_command` | the parity gate refuses a screen command that the parser does not name |
| `parity_refuse_key` | the parity gate refuses a key bar row that no screen takes |
| `parity_refuse_help` | the parity gate refuses a help command that the parser does not dispatch |
| `rng` | the Philox generator against known answers, and its spread |
| `mem` | the region map: the table, the bounds and the guard gap that faults |
| `seam` | the seam: the rate, the sequences, a held tick and the apply |
| `bus` | the bus: the writer stamp, the writer count, the refusals and the cursors |
| `bulk` | the bulk channel: staging, the handle, the block and the refusal |
| `kvcache` | the page cache: requests, maps, the page header, release and re-use |
| `sched` | the tick graph: its shape never changes, and a tick stays in its budget |
| `settings` | the settings table: the records, the refusals, the set and settings commands, the control page |
| `profile` | the card refusal at the four profiles and at a free memory beside the need |
| `model` | every kernel of the forward pass against a reference on the processor |
| `sample` | the sample kernel against the distribution it is asked for |
| `text` | the tokenizer against the golden lists, and the parts it is made of |
| `matrix` | the matrix kernels: dequantization, the tensor core product, the memory bound product |
| `model_gate` | the forward pass against the reference lists of the model files |
| `decode` | the decode of the tick graph: its records, its states and its rate |
| `tool` | the tool path: the parser, the request table and the two memory tools |
| `agent` | the agent record, the turn loop, the agenda engine and the manifest |
| `spill` | the spill gate over the built objects |
| `replay` | three scenarios that run the system, kill it, restore it, and compare the state hash |
| `disk_crc32c` | the checksum against the published value, and the two paths agree |
| `disk_segment` | the segment frames round trip, and damage is refused |
| `disk_hostring` | the block acceptor of the drain against a device that fills a host ring |
| `disk_inbound` | the records the feeder publishes into the inbound ring |
| `disk_drain` | the drain against a host ring: the segments and the derived files |
| `disk_derive` | the message lines the drain derives, and the switch for each type |
| `disk_bulk` | the drain writes every payload of the bulk ring with its checksum |
| `disk_feed` | the feeder against two pipes: the records that reach the ring |
| `disk_restore` | a journal is built, the restore runs, and the summary and the records are read |
| `disk_settings` | the settings file reader: the defaults, every refusal, the format, the write in place |
| `disk_journal` | the text the journal reader prints for the token records of a run |
| `disk_attach` | the terminal socket, its frames, the peer rule and a long journal path |
| `disk_keys` | the terminal key sequences, split input and the Escape wait |
| `disk_raster_tui` | the terminal viewport, cursor follow, seqlock read and changed cells |
| `disk_splash` | the splash forms, their sizes and the fixed dissolve order |
| `disk_screens` | the screen rows, their local actions and the command lines they send |
| `disk_sha256` | the digest against the published values and against the system tool |
| `disk_gguf` | model files with every value type, read back through the reader |
| `disk_manifest` | the models manifest: the write command, the check command and the reader |
| `disk_modelfile` | the model files that are there, and one of them streamed end to end |
| `cli` | the line editor and the command parser at one line and at 64 lines |
| `ui` | the panels at one record and at 64 records, and the raster |
| `mirror` | mirror publication, attached rates and whole snapshots |
| `raster_headless` | the raster and the pixel buffer path, with no window |
| `window` | the window on the display: frames drawn, and one frame read back |
| `sanitizer_memcheck` | compute-sanitizer memcheck over `seam`, `decode` and `agent` |
| `sanitizer_racecheck` | compute-sanitizer racecheck over `seam`, `ui` and `matrix` |
| `mem_fault` | the guard gap of the region map, which faults the device |
| `kvcache_fault` | the guard gap of the page cache, which faults the device |

The checks `raster_headless`, `window`, `sanitizer_memcheck`, `sanitizer_racecheck`,
`mem_fault` and `kvcache_fault` hold a resource of the device alone, so each runs beside no
other check. The check `agent` carries the longest time allowance, at 2,400 seconds.

## Environment values the checks read

- `AOTX_SANITIZER`: the sanitizer gate sets it to the tool name. The checks `seam`,
  `decode`, `agent`, `matrix` and `ui` read it, take fewer ticks, and leave their rate cases
  out.
- `AOTX_SANITIZER_SKIP`: names of programs the sanitizer gate leaves out, with spaces
  between them.
- `AOTX_SANITIZER_BIN`: the compute-sanitizer to run. Without it the gate takes the one on
  `PATH`. The build sets it to the one beside the compiler.
- `AOTX_FAULT_TESTS`: the checks `mem` and `kvcache` run their guard gap cases only when it
  is set.
- `DISPLAY` and `WAYLAND_DISPLAY`: the window check ends with status 1 when neither is set.
- `TMPDIR`: the disk-side checks make their working directory there. Without it they use
  `/tmp`.

## The gates

`tools/gate.sh` runs the three text and shape gates. With no arguments it examines staged
content, which suits a hook before a commit. With arguments it examines the files or the
directories given. It ends with status 0 when every gate is clean, 1 when a gate has
findings, and 2 on an environment error.

```
tools/gate.sh
tools/gate.sh docs cuda
python3 tools/ste-lint.py docs/06-build.md
python3 tools/size-gate.py cuda
python3 tools/seam-gate.py cuda disk
python3 tools/spill-gate.py --cuobjdump /usr/local/cuda-13.2/bin/cuobjdump build
tools/sanitizer-gate.sh memcheck build models tests/fixtures/tokenizer
```

- `ste-lint.py` refuses a sentence over 25 words, a paragraph over 6 sentences, a word or
  pattern that `tools/ste-words.txt` lists, and punctuation that is not ASCII. It reads
  prose in documents and comments in source files.
- `size-gate.py` refuses a file over 1,000 lines. It refuses a host glue file, whose name
  ends in `_host.cu`, over 300 lines. It warns at 800 lines.
- `seam-gate.py` refuses host work in a device file: runtime calls, driver calls, managed
  memory, device printf, device malloc, the C++ standard library and exceptions. In a `.c`
  or `.h` file under `disk/` it refuses every CUDA symbol.
- `spill-gate.py` reads the resource usage the device linker recorded. It refuses a listed
  hot kernel that keeps local memory, or a stack frame over its allowance. `--list` prints
  the kernel list and the allowances.
- `sanitizer-gate.sh` refuses a run in which the sanitizer reports a memory error or a data
  hazard. The first argument is `memcheck` or `racecheck`.

Each gate ends with status 0 when it is clean, 1 on a finding, and 2 on a usage or
environment error. The checks `spill`, `sanitizer_memcheck` and `sanitizer_racecheck`
register the last two gates with ctest.

## The model files

A model file is large and is not in the repository. The files go in one directory, which
`--models` names for a run and `AOTX_MODELS_DIR` names for the checks. One manifest file,
`manifest.jsonl`, sits in that directory and names each file.

`aotx_manifest` writes and checks that manifest:

```
aotx_manifest write <dir> <name> <file> <source> <revision> <license>
aotx_manifest check <dir>
```

The write command hashes the file and adds one line. The check command hashes each file and
compares the digest with its line. The write command refuses a name or a file that the
manifest already holds. Exit status 0 states that every file is right, 1 that a file is
different, missing or already in the manifest, and 2 an error.

One line holds seven fields: the name, the path, the source, the revision, the license, the
byte count and the SHA-256 digest. The name is the role of the file. The four roles are
`embedding`, `reranker`, `language` and `language-q4`. A run that names no role loads
`embedding,reranker,language`.

The start of a run reads the digest of each file of the roles it asks for, and compares it
with the manifest. A file that does not match its line ends the start with status 2. A role
the run does not ask for costs nothing: no digest, no bytes of the device, and no place in
the vocabulary.
