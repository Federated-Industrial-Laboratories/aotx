# Architecture

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
| pump | the host glue that launches the device scheduling graph once per tick |

AOTX-1 is a local inference operating system in CUDA. The authoritative state resides in device
memory: agents, models, a catalog, a message bus, a text interface and a command line. The disk
maintains a copy that lags by one tick. This document names the modules, the boundary, the crossings and
the graphs that run them.

## The seam

The seam is the host-device memory boundary: pinned host memory mapped for the GPU. Ring buffers
are the only structures that cross it. Four file types reside on the two sides of the seam.

| kind | path | holds | must not hold |
| --- | --- | --- | --- |
| device | `cuda/**/*.cu`, `cuda/**/*.cuh` | kernels, device functions, device data layouts | runtime and driver calls, host allocation, the C++ standard library, `main` |
| host glue | `cuda/**/*_host.cu` | allocation, module load, ring registration, graph capture, graph launch, event waits, display copy | loops over data, computation over model state, parsing |
| disk side | `disk/**/*.c`, `disk/**/*.h` | file input and output, memfd, checksum, digest, model file parsing, framing | a CUDA symbol, kernel syntax |
| raw PTX | `ptx/*.ptx` | kernels written by hand and loaded as modules | anything a `.cu` kernel calls |

Host glue moves bytes and starts work. It does not decide, parse, format, hash, search or count. A
loop over data outside the disk side is a defect, whatever the loop computes.

A host glue file comprises 300 lines at the most. Every other file comprises 1,000 lines at the most.
`tools/seam-gate.py` refuses a host call in a device file and a CUDA symbol in a disk-side file.
`tools/size-gate.py` refuses a file above its ceiling.

One header crosses the seam: `cuda/seam/wire.h`. The header is plain C and contains layouts and
constants only, and both sides include it. The dependency runs from the device on the layout, and
never from the device on disk-side code.

## The device modules

Each directory under `cuda/` is one module. The table gives what each module owns and the shape of
its kernels, from the banner of its header.

| module | owns | launch shape |
| --- | --- | --- |
| `boot` | the boot record and the module table | host glue only; no kernels |
| `mem` | the region table and the handle table | one thread for each handle lookup |
| `time` | the tick counter | one thread reads the clock |
| `settings` | the device setting table and the control page | one thread applies a change |
| `rng` | the random state of each agent | one thread for each agent |
| `seam` | the device ring, the host ring layout and the inbound cursor | one thread for each record; one block for the flush |
| `bus` | the message layouts, the sequence counter of each writer, the message buffer | one thread for each message |
| `sched` | the work queues and the tick statistics | one block for each queue |
| `text` | the vocabulary tables | one thread for each sequence, then one warp for each chunk |
| `model` | descriptors, layer parameters and run-time model placement | one block for each tile; the batch is the tokens of all sequences |
| `kvcache` | the page table of each agent and the queue of page requests | one thread for each agent; one thread for each page of the stamp |
| `embed` | the note store: one vector, one record sequence and the text of each note | one block for each sequence; the threads hold the hidden width |
| `rerank` | nothing | one block for each pair |
| `catalog` | imported skills, roles and tools | one thread commits each import or remove record |
| `agent` | agents, tasks, transcripts and conversation memory | one thread for each agent in the control step |
| `tool` | built-in tools, tool modules and pending requests | one thread for each request; one block for each device module row |
| `ui` | the cell grid, the panel table, the font and the pixel buffer | one block for each panel; one thread for each pixel |
| `cli` | the line buffer, the history, the command table and the console buffer | one thread; the apply step calls it in slot order |
| `moe` | nothing; this build has no mixture of experts model | not defined |

The counts that bound the modules are figures of the build profile (`cuda/profile/`, one
header for each profile). One figure, `AOTX_SLOTS`, gives the agent, sequence, cache, request
and bus-writer slots. Agent number `i` owns slot `i`. The 12g profile provides 64 slots, and the
8g profile provides 32. A system provides 256 task slots (`cuda/agent/agent.cuh`, `AOTX_TASK_SLOTS`).
The repository supplies conductor, worker and verifier role modules.

## Device memory

The host glue reserves one virtual range and maps the regions of the system into it
(`cuda/mem/mem_host.cu`, `aotx_mem_reserve`). The order of the range is the record ring, a guard
gap, the scratch arena, a guard gap, the weights region, and a guard gap. A guard gap has no
physical memory behind it, so a write past the end of a region faults.

| region | size | source |
| --- | --- | --- |
| record ring | 16 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_RING_BYTES` |
| scratch arena | 64 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_SCRATCH_BYTES` |
| guard gap | 2 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_GUARD_BYTES` |
| weights region | 5 GB to 40 GB of virtual range, mapped in pieces of 2 MB | the profile, `AOTX_MEM_WEIGHTS_BYTES`; `cuda/mem/mem.cuh`, `AOTX_MEM_WEIGHTS_GRAIN` |

The weights region has no physical memory at start. Physical memory is mapped in pieces while each
tensor streams in. The key value cache has a separate range in 2 MB pages. Its size is 1 GB
on 8g and 2 GB on 12g. The first 8 MB of the scratch arena is the bulk staging region
(`cuda/seam/seam.cuh`, `AOTX_BULK_STAGE_BYTES`).

## Conversation memory tiers

Each agent owns an ordered transcript (`cuda/agent/transcript.cu`). The newest turns are hot.
Their prompt tokens and key value pages stay within the page limit selected for the agent.

Turns that exceed the hot page limit become warm. The embedding batch gives each warm turn a
vector. A new prompt recalls the nearest warm turns and records that choice in a SELECTION
record. A restore applies the recorded choice and does not search again.

Compaction folds the oldest half of warm memory into a summary finding. The folded turns keep
their vectors. Their text is removed first when the circular text arena needs room. The profile
sets the turn count and text bytes for each agent. `docs/07-operation.md` gives the profile
figures and the controls for these tiers.

## The tick graph

The tick graph is captured once, instantiated once, and launched once for each tick
(`cuda/sched/pump_host.cu`, `aotx_pump_build`). The capture uses the nonblocking stream of the pump.
The shape of the graph never changes. Two nodes receive a new parameter for each tick: the tick
start node and the tick load node.

The node order is the order of the tick.

| position | node | grid and block |
| --- | --- | --- |
| 1 | `aotx_sched_tick_start` | 1 block of 1 thread |
| 2 | `aotx_seam_apply_inbound` | 8 blocks of 128 threads |
| 3 | the say path of the command layer | 6 nodes at the most |
| 4 | the decode: the plan, the forward pass as one child node, the commit | 3 nodes |
| 5 | the tool path | 9 nodes, or 1 node with no embedding role |
| 6 | the agent step | 1 node |
| 7 | the reply of the console | 1 node |
| 8 | `aotx_sched_workload` | the tick load blocks of 256 threads |
| 9 | `aotx_sched_commit` | 1 block of 1 thread |
| 10 | `aotx_seam_flush` | 1 block of 1,024 threads |
| 11 | `aotx_seam_bulk_flush` | 1 block of 1,024 threads |

The node counts come from the names that begin `AOTX_TICK_NODES_` in `cuda/sched/sched.cuh`. The
graph comprises 64 nodes at the most (`AOTX_TICK_NODES_MAX`). The agent step comes after the tool
step, because an agent processes its tool result in the tick that result arrives. The reply of
the console comes after the agent step, so a reply that no agent streams shows nothing.

The pump launches the graph and records an event (`cuda/sched/pump_host.cu`, `aotx_pump_tick`).
The event reports completion. The pump services page requests when no tick-graph kernel runs.
The pump then sleeps for the remainder of the tick period.

The period is the setting `tick.period_ms` (`cuda/settings/keys.h`, `AOTX_SET_TICK_PERIOD_MS`).
It is 10 ms unless the settings file or a `set` line changes it. The pump therefore makes
100 ticks in one second at the most. The pump reads the period from the control page
(`cuda/settings/settings.cuh`).

## The raster graph

The raster graph is captured once and comprises seven nodes (`cuda/ui/raster_host.cu`,
`aotx_ui_graph_build`). Six panel kernels write the cells of their own panel: the console, the
agents, the bus, the arena, the tick and the seam. Each panel kernel uses one block of 128
threads. The raster kernel then composes the cell grid into the pixel buffer with 1,024 blocks of
256 threads.

The graph uses a stream of the highest priority that the device gives. A long tick on the pump
stream therefore does not block the display. The grid is 160 columns by 50 rows of cells, and a
cell is 8 pixels by 16 pixels. The pixel buffer is 1,280 by 800 pixels. The figures come from
`AOTX_UI_COLS`, `AOTX_UI_PANEL_THREADS` and `AOTX_UI_RASTER_BLOCKS`, in `cuda/ui/ui.cuh`.

The graph also publishes the cells and their header to a two-slot mirror memfd. `aotx_tui`
receives a read-only descriptor from the feeder and draws the same six panels in a terminal.
The mirror uses a sequence around each snapshot so a reader can refuse a torn copy. It is a
display copy, and device memory remains the authoritative state.

A run with a window uses one thread for the tick pump and draws on the first thread
(`cuda/boot/window_boot_host.cu`, `aotx_boot_window_run`). The window glue writes each key event
as a 16-byte frame into a pipe, and the feeder makes one key record of each frame.

## The raw PTX modules

A PTX module is loaded with `cuModuleLoadData` and added to a graph as a kernel node through the
driver. No `.cu` kernel calls a PTX module.

| module | entry point | launch shape | loaded by |
| --- | --- | --- | --- |
| `ptx/clock.ptx` | `aotx_clock_sample` | one thread | `cuda/boot/clock_host.cu`, `aotx_boot_clock_check` |
| `ptx/gemv_q8.ptx` | `aotx_gemv_q8` | one warp for each run of 2 rows, 256 threads a block | `cuda/model/decode_host.cu`, `aotx_decode_build` |

The clock module samples the device clock into one 64-bit word. The check of that module is the
first act of every start (`cuda/boot/boot_host.cu`, `aotx_boot_clock_check`). The product module
multiplies one row of activations by a weight tensor of the 8-bit block type. Its node enters the
capture of the forward pass with `cuGraphAddKernelNode` (`cuda/model/decode_host.cu`,
`aotx_model_module_node`).

## The disk side

The disk side is C with no CUDA dependency. Each program receives the ring descriptors it must map
and no others (`cuda/seam/seam_host.cu`, `aotx_seam_only`).

| name | source | what it does |
| --- | --- | --- |
| `aotx_drain` | `disk/drain/` | copies blocks from the host ring into journal segments and derived files; writes the payloads of the bulk ring |
| `aotx_feed` | `disk/feed/` | writes terminal lines, key frames and tool replies into the inbound ring; reads a file under the allowed root |
| `aotx_restore` | `disk/restore/` | replays the class A records of the newest complete journal into the inbound ring |
| `aotx_journal` | `disk/journal/` | prints the records of a journal as text, so a reader can compare two runs |
| `aotx_manifest` | `disk/manifest/` | writes and checks the manifest that names each model file and its digest |
| `aotx_disk` | `disk/wire/` | the library: ring maps, block reads, segment files, CRC-32C, SHA-256 |
| `aotx_modelfile` | `disk/modelfile/` | the model file reader, linked into host glue |
| `aotx_models` | `disk/models/` | lists, fetches, checks, activates and removes model-store files |
| settings library | `disk/settings/` | reads and writes the settings file |
| `aotx_tui` | `disk/tui/` | reads the mirror, draws the terminal, and sends keys and complete lines to the feeder |

The drain writes the journal segments and the derived files. The derived files are outputs only,
and no program reads them back as inputs. The requests file is the one exception. The drain writes
it (`disk/drain/derive_manifest.c`, `open_requests`) and the feeder reads it, to find the file
read requests it must answer (`cuda/boot/children_host.cu`, `aotx_boot_start_feed`).

Nothing hashes on the device. The disk side computes checksums and SHA-256 digests. The disk
copy is the chain that outlives a lost context.

## The catalog, tools and model store

The catalog is a device table of installed skills, roles and tools. The feeder reads a module
directory and publishes its bytes as class A IMPORT records. REMOVE records remove modules.
The journal therefore rebuilds the catalog without reading module text again.

Nine built-in tools enter the catalog before the first tick. Device tools run inside the tick
graph. Host tools run as feeder programs after the operator installs or permits them.
`docs/09-modules.md` gives the catalog and `docs/10-tool-sdk.md` gives both tool contracts.

The model store is a disk directory with `store.jsonl` and `manifest.jsonl`. Its catalog is
`share/models/catalog.jsonl`. The store program fetches and verifies files. The command parser
can load an active manifest entry between ticks. A MODEL record keeps the role, file, digest
and placement tick for restore.

## What the system reaches

The system opens no connection of its own and starts no program of its own. A run reaches past
its own state only where the operator lets it. One path does so: a host tool module.

A host tool is a program in any language. The feeder starts it with the rights of the operator
and provides no sandbox (`docs/10-tool-sdk.md`). Module installation therefore grants that access
to a run. Two authorization guards apply after installation: the manifest key `authorise` with
the value `always`, and the `authorise` list of the role. Each guard requires operator authorization
for every call.

A device tool module reaches nothing. Its two parameters give it the arguments of the call, a
scratch run of its own row and one output row. It sees no ring, no file and no row of another
tool.
