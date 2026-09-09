# Build

This document uses these project terms.

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

This document states what the build needs and how to run it. It then states what each build
option registers, what each check covers, and what each gate refuses. The last section
states how the model files are recorded.

## Requirements

- CUDA Toolkit 13.2 or later. The build uses `nvcc` from the toolkit. Put the `bin`
  directory of the toolkit on `PATH`, such as `/usr/local/cuda-13.2/bin`.
- A driver that supports the toolkit. The build links the driver library `libcuda`.
- A GPU of compute capability 8.0 or above. The reference card is 8.6. The build compiles
  device code for the architecture that `AOTX_ARCH` names, 86 unless given. It carries the
  PTX of that architecture too, so a newer card runs it through the driver. Build for the
  card itself when you can (the section below). A card below 8.0 is not supported.
- CMake 3.28 or later, and Ninja.
- A C compiler for C11, and a C++ compiler for C++17.
- Python 3, for the gates. The full-row accuracy checks also require NumPy.
- pkg-config, GLFW 3, GLEW and OpenGL, for the window.
- EGL, for the raster check that opens no window.
- X11, for the close request that the window check sends.
- A thread library. Each check program links it.
- Linux, for the disk side. The rings are memfd files, and the disk side builds with
  `_GNU_SOURCE`.

The disk side is C with no CUDA dependency. On an x86_64 machine the build adds a second
checksum path in one file, built with SSE 4.2. No other file uses that instruction set.

On Ubuntu, this command installs the host build packages:

```
sudo apt install build-essential cmake ninja-build python3 python3-numpy pkg-config \
  libglfw3-dev libglew-dev libegl1-mesa-dev libx11-dev libxtst-dev libcurl4-openssl-dev
```

Install the CUDA Toolkit separately. Put the `bin` directory of that installation on
`PATH` before configuration.

## Configure and build

```
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
build/aotx_boot --version
```

Every program of the build lands in the build directory itself. A program therefore finds
its siblings beside it, which is how the boot program starts the disk-side programs.

## The profile and the architecture

Two options select the card configuration. `AOTX_PROFILE` selects the profile. The profile fixes
the base device table sizes. Those are the slots, the ring sizes, the key value range,
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
states its status in its banner. The checks get their batch size from the profile, so the
same list runs at 32 slots on `8g` and at 64 on `12g`. A boot on a card that cannot support
the profile refuses with the figures and names the profile that fits (`docs/07-operation.md`).

```
bash tools/profile-detect.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_PROFILE=8g -DAOTX_ARCH=86
```

## Build options

Three options register checks that the default build leaves out. Other values select the
card, model directory, fetch support, the affect substrate and a bus validator.

| option | default | what it does |
| --- | --- | --- |
| `AOTX_PROFILE` | `12g` | the build profile: `8g`, `12g`, `24g` or `48g` |
| `AOTX_ARCH` | 86 | the compute architecture of the `.cu` files, as `sm_<n>` |
| `AOTX_MEMORY_OBJECTS` | 8192 | stored object-version slots per typed memory store |
| `AOTX_MEMORY_BYTES` | 16777216 | payload bytes per typed memory store |
| `AOTX_FETCH` | ON when CMake finds libcurl | build model fetch support; ON without libcurl is an error |
| `AOTX_AFFECT` | ON | build the affect substrate and the conversation quality instrument |
| `AOTX_DISPLAY_TESTS` | OFF | the check `window`, with the label `display` |
| `AOTX_FAULT_TESTS` | OFF | the checks `mem_fault` and `kvcache_fault` |
| `AOTX_SANITIZER_TESTS` | OFF | the checks `sanitizer_memcheck` and `sanitizer_racecheck`, with the label `sanitizer` |
| `AOTX_MODELS_DIR` | `models` | the directory the checks read the model files from |
| `AOTX_BUS_LINT` | empty | a program that validates a bus line file, for `disk_drain` and `disk_derive` |

A build without `AOTX_AFFECT` leaves out `cuda/affect/` and `cuda/quality/`, the thirteen
affect settings and the two derived streams. It leaves out the Trace window, the Dials window
and the `Couple` control of the Voice window as well. `docs/14-affect.md` states the feature,
and `docs/13-control.md` states the three parts of the control program.

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

## Memory capacity

Memory capacity options apply to CUDA, disk transport, state and recall commands, and recovery.
They are separate from the card profile. Configure the object and payload bounds together,
then rebuild the complete runtime and its disk programs. A running instance cannot resize.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_MEMORY_OBJECTS=16384 -DAOTX_MEMORY_BYTES=67108864
cmake --build build
build/aotx_ccir_state --limits
```

The capacity report does not need a GPU. Its `image_bytes` value includes the 128-byte
header, all 256-byte object rows and the payload allocation. CMake rejects values that
overflow the two-image transfer's 32-bit byte count.

The live runtime allocates three stores, a two-image input buffer and a retained-result image.
Recall also needs 64 separate scratch rows. The main capacity-dependent GPU cost in bytes is approximately:

```
6 * (objects * 256 + payload_bytes) + 64 * 12 * objects + fixed_buffers
```

Fixed buffers include bindings, results, headers and text preparation state. Models and
the base runtime need additional memory. Full-capacity state copies and linear lookup
remain part of the cost. Capacity pressure does not select paging, eviction or reclamation.

With the default `12g` profile and memory capacity, the live allocation has these byte counts:

| Allocation | Bytes |
| --- | --- |
| One typed store; three are resident | 18,874,408 |
| Live input, result and text state | 57,757,360 |
| Bindings | 1,111,552 |
| Recall scratch | 6,291,456 |
| Total of the listed GPU buffers | 121,783,592 |

The total excludes command formatting, the base runtime, model weights and CUDA context.
Offline state and recall commands allocate their own stores and transfer buffers.

The [typed state format](18-typed-state.md) keeps schema 1 across these configurations.
A larger build can restore a smaller admitted file. A smaller build refuses excess objects
or payload without publishing a partial result.

## The checks

`ctest --test-dir build` runs the checks registered for the selected build options. The
checks `load`, `text`, `matrix`, `model_gate`, `decode`, `tool`, `agent`, `replay`,
`terminal_path`, `disk_screens`, `disk_sha256`, `disk_manifest` and `disk_modelfile` need a
model file. Each check reports CTest status `Skipped` when the models manifest is not there.
A skipped check does not count as a passed check.

The `memory_config` and `memory_capacity` checks test configured bounds and batched recall.
With a local language model store, check live retention and cold recovery at both batch sizes:

```
python3 tests/capacity_boot_test.py build . MODEL_STORE NEW_OUTPUT_1 1
python3 tests/capacity_boot_test.py build . MODEL_STORE NEW_OUTPUT_64 64
```

Use at least 1,600 object slots and 2 MiB of payload capacity.
Each output path must be new. The checks start and stop their own runtime processes.

To require a real tensor check, run `build/aotx_matrix_device_test --real-file FILE TYPE TENSOR`.
`TYPE` is Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K or Q6_K.
Without `TENSOR`, the check selects the first two-dimensional tensor of that type.

Missing input fails; it does not skip. The check compares every element with a double reference
and reports value and group reader errors. Reader equality compares half values after group conversion.
The check also runs both matrix products at batches of one and 64.

| check | what it covers |
| --- | --- |
| `clock_module` | the boot program with `--clock-only`; the clock module gives a sample |
| `parity` | each terminal action and static Bus word against the command parser |
| `parity_refuse_command` | the parity gate refuses a screen command that the parser does not name |
| `parity_refuse_key` | the parity gate refuses a key bar row that no screen takes |
| `parity_refuse_help` | the parity gate refuses a help command that the parser does not dispatch |
| `size_gate_boundary` | the size gate refuses each first value above a file limit |
| `fault_status` | child exit and signal controls for memory fault checks; no device required |
| `rng` | the Philox generator against known answers, and its spread |
| `mem` | the region map: the table, the bounds and the guard gap that faults |
| `seam` | the seam: the rate, the sequences, a held tick and the apply |
| `bus` | the bus: the writer stamp, the writer count, the refusals and the cursors |
| `bulk` | the bulk channel: staging, the handle, the block and the refusal |
| `kvcache` | the page cache: requests, maps, the page header, release and re-use |
| `sched` | the tick graph: its shape never changes, and a tick stays in its budget |
| `settings` | the settings table: the records, the refusals, the set and settings commands, the control page |
| `load` | run-time model replacement, records, digest checks and sequence preservation |
| `profile` | card bounds, layer selection, rotary family rules, head metadata, and expert and bias tensor refusals |
| `model` | forward kernels, batched tensor binding, block types, absent slots and descriptor bounds |
| `sample` | the sample kernel against the distribution it is asked for |
| `text` | the tokenizer against the golden lists, and the parts it is made of |
| `matrix` | the matrix kernels: dequantization, the tensor core product, the memory bound product |
| `matrix_blocks` | packed readers against a double reference, with distinct rows at batches of 1 and 64; no model file required |
| `expert` | per-token routing, selected matrix slices, weighted sums and full-width norms at batches of 1 and 64 |
| `bias` | query and key bias before rotation, value bias before cache writes, grouped heads and token bounds |
| `arch_logits_input` | complete device capture requests and invalid binary inputs |
| `arch_accuracy_metrics` | full-row error metrics and fixed-bound checks |
| `arch_accuracy_inputs` | corpus, row, token and capture identities |
| `arch_accuracy_reference` | decoded reference proofs, fixed input groups and model identities |
| `arch_accuracy_bundle` | original calibration membership and clear winners in each fresh input mode |
| `arch_reference_inputs` | fixed teacher inputs, reference reproduction and library identities |
| `delta` | fixed matrix and convolution state, gates, resets, slot refusal and exact chunk replay at batches of 1 and 64 |
| `gated` | joint query/gate layout, partial rotary turn, 256-value heads, causal pages and output gate |
| `hybrid_file` | literal mixed tensor sets, metadata refusals, compact page maps and fixed allocation bytes at 1 and 64 layers |
| `model_gate` | the forward pass against the reference lists of the model files |
| `decode` | the decode of the tick graph: its records, its states and its rate |
| `tool` | the tool path: the parser, the request table and the two memory tools |
| `agent` | the agent record, the turn loop, the agenda engine and the manifest |
| `conversation` | line parts, hot and warm memory, recall, compaction and selection replay |
| `catalog` | module import, replacement, remove, catalog limits and arena integrity |
| `module_setup` | build the example modules and refusal fixtures for the module checks |
| `module` | a device tool module at one row and at the profile row count |
| `example_word_count`, `example_echo_upper` | the two SDK examples through the module check program |
| `module_refuse_*` | seven malformed, unsafe or late modules that the check program refuses |
| `module_build_host_call` | the module build script refuses a device module with a host call |
| `spill` | the spill gate over the built objects |
| `replay` | system, settings, module, model and conversation scenarios across a kill and restore |
| `disk_catalog`, `disk_store`, `disk_fetch`, `disk_models` | catalog JSON, local state, fetch guards and store commands |
| `disk_feed_models` | model fetch lines and store publication through the feeder |
| `disk_import` | module directory import and its refusal rules |
| `disk_fs_tools`, `disk_run_tool` | file tools and program tools through the feeder |
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
| `terminal_path` | terminal import, skill list and conductor reply through a PTY and the attach socket |
| `disk_parts` | atomic publication of every part of one long line |
| `disk_transcript` | all transcript kinds and comparison with journal records |
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
| `window` | drawn frames, frame readback, and text/Enter order for one and 64 input lines |
| `sanitizer_memcheck` | compute-sanitizer memcheck over `seam`, `decode` and `agent` |
| `sanitizer_racecheck` | compute-sanitizer racecheck over `seam`, `ui` and `matrix` |
| `mem_fault` | the guard gap of the region map, which faults the device |
| `kvcache_fault` | the guard gap of the page cache, which faults the device |

The architecture executable also compares each fixed-state byte after carried decode, whole
prompt replay and split prompt replay. These checks run at one and 64 distinct sequence slots
when the file has recurrent layers. They supplement, but do not replace, process restore.

The checks `raster_headless`, `window`, `sanitizer_memcheck`, `sanitizer_racecheck`,
`mem_fault` and `kvcache_fault` use a device resource exclusively, so each runs beside no
other check. The check `agent` carries the longest time allowance, at 2,400 seconds.

The [architecture accuracy procedure](17-accuracy.md) covers full-row reference checks and CPU reproduction.

## Environment values the checks read

- `AOTX_SANITIZER`: the sanitizer gate sets it to the tool name. The checks `seam`,
  `decode`, `agent`, `matrix` and `ui` read it, use fewer ticks, and leave their rate cases
  out.
- `AOTX_SANITIZER_SKIP`: names of programs the sanitizer gate leaves out, with spaces
  between them.
- `AOTX_SANITIZER_BIN`: the compute-sanitizer to run. Without it the gate uses the one on
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
manifest already contains. Exit status 0 states that every file is right, 1 that a file is
different, missing or already in the manifest, and 2 an error.

One current line contains the name, role, path, source, revision, license, byte count and SHA-256
digest. An old line without `role` uses its name as its role. The four roles are `embedding`,
`reranker`, `language` and `language-q4`.

The repository catalog names each offered file with its source, revision, license, size and
SHA-256 digest. The local manifest contains the same identity for each active file. Run the store
program to use that catalog:

```
aotx_models [--dir <dir>] [--catalog <file>] list
aotx_models [--dir <dir>] [--catalog <file>] fetch <name>
aotx_models [--dir <dir>] check
aotx_models [--dir <dir>] [--catalog <file>] activate <role> <name>
aotx_models [--dir <dir>] [--catalog <file>] remove <name>
```

The default catalog is `share/models/catalog.jsonl`. A fetch writes the final file only when
its byte count and digest agree. Activation requires the catalog role and writes the separate
name and role into `manifest.jsonl`.

The start of a system reads the digest of each required role file. It compares each
digest with the manifest. The 12g default roles are `embedding,reranker,language`. The 8g
default roles are `embedding,reranker,language-q4`.

A file that does not match its line ends the start with status 2. An unused role costs no digest,
device bytes or vocabulary place.

See `docs/07-operation.md`, "Use another model file", for a local catalog, activation, and a language-only boot.
That procedure also states the limits of the store and device checks.
