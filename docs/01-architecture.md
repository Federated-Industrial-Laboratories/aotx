# Architecture

AOTX-1 is a local inference operating system in CUDA. Device memory holds the authoritative state:
agents, models, a message bus, a text interface and a command line. The disk holds a copy that is
one tick behind. This document names the modules, the boundary they stand on, and the two graphs
that run them.

## The seam

The seam is the boundary between device memory and pinned host memory. Rings cross it and nothing
else crosses it. Four kinds of file stand on the two sides of the seam.

| kind | path | holds | must not hold |
| --- | --- | --- | --- |
| device | `cuda/**/*.cu`, `cuda/**/*.cuh` | kernels, device functions, device data layouts | runtime and driver calls, host allocation, the C++ standard library, `main` |
| host glue | `cuda/**/*_host.cu` | allocation, module load, ring registration, graph capture, graph launch, event waits, display copy | loops over data, computation over model state, parsing |
| disk side | `disk/**/*.c`, `disk/**/*.h` | file input and output, memfd, checksum, digest, model file parsing, framing | a CUDA symbol, kernel syntax |
| raw PTX | `ptx/*.ptx` | kernels written by hand and loaded as modules | anything a `.cu` kernel calls |

Host glue moves bytes and starts work. It does not decide, parse, format, hash, search or count. A
loop over data outside the disk side is a defect, whatever the loop computes.

A host glue file holds 300 lines at the most. Every other file holds 1,000 lines at the most.
`tools/seam-gate.py` refuses a host call in a device file and a CUDA symbol in a disk-side file.
`tools/size-gate.py` refuses a file above its ceiling.

One header crosses the seam: `cuda/seam/wire.h`. The header is plain C, it holds layouts and
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
| `rng` | the random state of each agent | one thread for each agent |
| `seam` | the device ring, the host ring layout and the inbound cursor | one thread for each record; one block for the flush |
| `bus` | the message layouts, the sequence counter of each writer, the message buffer | one thread for each message |
| `sched` | the work queues and the tick statistics | one block for each queue |
| `text` | the vocabulary tables | one thread for each sequence, then one warp for each chunk |
| `model` | the model descriptors and the layer parameters | one block for each tile; the batch is the tokens of all sequences |
| `kvcache` | the page table of each agent and the queue of page requests | one thread for each agent; one thread for each page of the stamp |
| `embed` | the note store: one vector, one record sequence and the text of each note | one block for each sequence; the threads hold the hidden width |
| `rerank` | nothing | one block for each pair |
| `agent` | the agent records, the role table and the task table | one thread for each agent in the control step |
| `tool` | the tool table and the pending request table | one thread for each request |
| `ui` | the cell grid, the panel table, the font and the pixel buffer | one block for each panel; one thread for each pixel |
| `cli` | the line buffer, the history, the command table and the console buffer | one thread; the apply step calls it in slot order |
| `moe` | nothing; the module is not part of version 0.1 | not defined |

The counts that bound the modules are figures of the build profile (`cuda/profile/`, one
header for each profile; `docs/07-operation.md` names the profiles). One figure, `AOTX_SLOTS`,
gives the agent slots, the sequence slots, the key value cache slots, the request slots and
the bus writers. Agent number `i` owns slot `i`. The reference profile holds 64 slots. A
run holds 256 task slots (`cuda/agent/agent.cuh`, `AOTX_TASK_SLOTS`). A role is one of three:
conductor, worker, verifier (`cuda/agent/agent.cuh`, `AOTX_ROLE_COUNT`).

## Device memory

The host glue reserves one virtual range and puts the regions of the system in it
(`cuda/mem/mem_host.cu`, `aotx_mem_reserve`). The order of the range is the record ring, a guard
gap, the scratch arena, a guard gap, the weights region, and a guard gap. A guard gap has no
physical memory behind it, so a write past the end of a region faults.

| region | size | source |
| --- | --- | --- |
| record ring | 16 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_RING_BYTES` |
| scratch arena | 64 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_SCRATCH_BYTES` |
| guard gap | 2 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_GUARD_BYTES` |
| weights region | 8 GB of virtual range, mapped in pieces of 2 MB | `cuda/mem/mem.cuh`, `AOTX_MEM_WEIGHTS_BYTES` and `AOTX_MEM_WEIGHTS_GRAIN` |

The weights region has no physical memory at start. Each tensor asks for the pieces it needs while
the tensor streams in. The key value pages hold a range of their own of 2,048 MB in pages of 2 MB
(`cuda/kvcache/kvcache.cuh`, `AOTX_KV_RANGE_BYTES` and `AOTX_KV_PAGE_BYTES`). The first 8 MB of
the scratch arena is the staging region of the bulk channel (`cuda/seam/seam.cuh`,
`AOTX_BULK_STAGE_BYTES`).

## The tick graph

The tick graph is captured once, instantiated once, and launched once for each tick
(`cuda/sched/pump_host.cu`, `aotx_pump_build`). The capture takes the stream of the pump, which is
made with the flag that does not block. The shape of the graph never changes. Two nodes take a new
parameter for each tick: the tick start node and the tick load node.

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
graph holds 64 nodes at the most (`AOTX_TICK_NODES_MAX`). The agent step comes after the tool
step, because an agent takes the result of its tool in the tick that result arrives. The reply of
the console comes after the agent step, so a reply that no agent streams shows nothing.

The pump launches the graph, records an event, and waits on the event (`cuda/sched/pump_host.cu`,
`aotx_pump_tick`). The page requests of a tick are answered after that wait, when no kernel of the
tick graph runs. The pump then sleeps the rest of the tick period.

The period is the setting `tick.period_ms` (`cuda/settings/keys.h`, `AOTX_SET_TICK_PERIOD_MS`).
It is 10 ms unless the settings file or a `set` line changes it. The pump therefore makes
100 ticks in one second at the most. The pump reads the period from the control page
(`cuda/settings/settings.cuh`).

## The raster graph

The raster graph is captured once and holds seven nodes (`cuda/ui/raster_host.cu`,
`aotx_ui_graph_build`). Six panel kernels write the cells of their own panel: the console, the
agents, the bus, the arena, the tick and the seam. Each panel kernel takes one block of 128
threads. The raster kernel then composes the cell grid into the pixel buffer with 1,024 blocks of
256 threads.

The graph takes a stream of the highest priority that the device gives. A long tick on the pump
stream therefore does not hold the display. The grid is 160 columns by 50 rows of cells, and a
cell is 8 pixels by 16 pixels. The pixel buffer is 1,280 by 800 pixels. The figures come from
`AOTX_UI_COLS`, `AOTX_UI_PANEL_THREADS` and `AOTX_UI_RASTER_BLOCKS`, in `cuda/ui/ui.cuh`.

A run with a window puts the tick pump on its own thread and draws on the first thread
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

The drain writes the journal segments and the derived files. The derived files are outputs only,
and no program reads them back as inputs. The requests file is the one exception. The drain writes
it (`disk/drain/derive_manifest.c`, `open_requests`) and the feeder reads it, to find the file
read requests it must answer (`cuda/boot/children_host.cu`, `aotx_boot_start_feed`).

Nothing hashes on the device. The drain computes the CRC-32C of each block and the digest chain of
the turns. The disk copy is the chain that outlives a lost context.
