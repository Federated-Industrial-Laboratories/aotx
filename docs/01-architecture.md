<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Architecture

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

AOTX-1 is a local inference runtime whose authoritative state resides in GPU memory.
CUDA executes agents, model inference, memory selection, policy decisions and the message bus.
The disk side maintains an asynchronous journal replica and supplies verified input bytes.

One instance uses one GPU. Local window and terminal clients expose its state.
An optional HTTP gateway connects application clients through a scoped service protocol.
The gateway does not own inference or persistent cognitive state.

<details>
<summary>On this page</summary>

- [State ownership](#state-ownership)
- [The seam](#the-seam)
- [The device modules](#the-device-modules)
- [Device memory](#device-memory)
- [Model execution](#model-execution)
- [The tick graph](#the-tick-graph)
- [The raster graph](#the-raster-graph)
- [The raw PTX modules](#the-raw-ptx-modules)
- [The disk side](#the-disk-side)
- [Memory and persistence](#memory-and-persistence)
- [Applications and trust](#applications-and-trust)

</details>

## State ownership

| Boundary | Responsibilities | Source |
| --- | --- | --- |
| Device | Models, agents, typed memory, scheduling, admission and recorded decisions. | `cuda/` device files |
| Host glue | Allocation, module loading, transport registration and graph launch. | `*_host.cu` |
| Disk | Files, journal, verified model input, restore and transport framing. | `disk/` |
| Local clients | Terminal rendering and graphical instance controls. | `disk/tui/`, `ctrl/` |
| HTTP gateway | HTTP, TLS, credentials, bounded uploads and network transfer. | `gateway/` |

Host glue moves bytes and starts device work. It does not search memory, interpret source text or calculate model results.
The disk side contains no CUDA dependency.
The gateway carries versioned requests and device results without importing a host inference engine.

## The seam

The seam is the mapped host-device transport boundary.
Bounded rings carry records, requests, results and larger payloads.
Publication fields and explicit ordering prevent readers from accepting incomplete writes.
The [seam contract](03-seam-contract.md) defines its original journal and input layouts.
Specialized service and memory transports define their own versioned layouts.

`cuda/seam/wire.h` contains shared record layouts and constants in plain C.
Other narrow wire headers describe memory, media and service messages.
Shared headers define bytes and bounds; they do not give disk code access to live device objects.

`tools/seam-gate.py` checks forbidden cross-boundary calls.
`tools/size-gate.py` limits source files to 1000 lines and host-glue files to 300 lines.
These source checks supplement the architectural contract.

## The device modules

| Modules | Owned state or work |
| --- | --- |
| `boot`, `profile`, `mem` | Runtime admission, region layout and capacity configuration. |
| `time`, `rng`, `settings` | Tick state, random streams and recorded device settings. |
| `seam`, `bus`, `sched` | Record publication, messages and ordered tick execution. |
| `text`, `model`, `kvcache` | Tokenization, model layers, sampling and cache pages. |
| `embed`, `rerank` | Embeddings, note search and reranking. |
| `catalog`, `agent`, `tool` | Installed modules, agent turns and tool requests. |
| `cognitive` | Typed objects, scoped recall, source interpretation, checkpoints and cold-memory decisions. |
| `appraisal`, `reflection` | Supported source assessments and task-scoped review cues. |
| `policy` | Bounded idle-work selection, control and persistent policy state. |
| `media`, `vision`, `audio` | Source ownership, native preprocessing and model feature rows. |
| `service`, `shared` | Scoped request admission and persistent shared-resource state. |
| `affect`, `quality` | Optional control state and measurement streams. |
| `cli`, `ui` | Command parsing, the cell grid, display mirror and raster output. |

`AOTX_SLOTS` binds the main agent and sequence tables to the profile.
An agent uses its corresponding sequence slot.
Persistent service objects have separate capacities; they do not require one permanently occupied execution slot each.
The [build guide](06-build.md) lists profiles and independent limits.

## Device memory

The runtime reserves virtual regions and maps physical storage where required.
Guard gaps separate the record ring, scratch arena and weights range.
A write into an unmapped guard gap faults instead of overwriting another region.

| Region | Base size or rule |
| --- | --- |
| Record ring | 16 MiB. |
| Scratch arena | 64 MiB, including the bulk staging area. |
| Guard gap | 2 MiB between guarded regions. |
| Weights | Profile-defined virtual range; tensors map in 2 MiB pieces. |
| Key/value cache | Separate physical page pool with per-slot maps. |
| Typed memory and media | Separate configured stores, scratch buffers and workspaces. |

The virtual weights range is not a physical-memory reservation for every possible model.
The runtime must also fit cache, state, transport and execution workspaces.
Model metadata determines page occupancy and recurrent-state dimensions.

## Model execution

A model descriptor selects compiled layer types from checked tensor metadata.
Each layer row declares its tensor slots, capture path and workspace requirements.
Offsets remain relative to the weights region and use 64-bit values.
Unsupported layouts are refused before execution.

Attention, biased attention, routed experts, gated attention and linear-delta state use their respective device paths.
Expert routing selects per-token slices and combines their weighted outputs on the GPU.
Hybrid models combine paged layers with fixed-size recurrent state.
Only paged layers contribute to key/value page occupancy.

Each recurrent sequence retains its own F32 matrix and convolution history.
Prompt replay rebuilds them in token order.
A model with no paged layers still needs a separate admission path; hybrid support does not imply that path exists.

Tensor readers support F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and Q2_K through Q6_K.
A file still needs compatible tensor shapes, tokenizer metadata and turn wraps.

[Model files](16-model-files.md) describes inspection and checked examples.
[Accuracy](17-accuracy.md) separates structural support from unresolved numerical comparison results.

## The tick graph

The runtime captures and instantiates its scheduling graph during setup.
Each tick executes ordered admission, command, inference, tool, agent and publication work.
Optional modules add bounded nodes for memory, media, shared service and idle policy work.
The graph shape does not grow with each submitted conversation.

The scheduling contract is in `cuda/sched/sched.cuh` and the capture code in `cuda/sched/capture_host.cu`.
Named node capacities describe the current compiled graph.
Model-role batches share the total token budget and restore the default role between selections.

The pump launches the graph, waits for its completion event and services page requests between ticks.
It then follows `tick.period_ms`, whose default is 10 ms.
The period limits normal pacing; a slow tick does not prove that the configured deadline was met.

At tick start, the device reads output capacity and decides whether new recorded work can proceed.
A journal hold prevents new mutations that require unavailable record space.
Kernels do not spin while waiting for the disk writer.
[Time and recovery](02-temporal-model.md) describes record order and durable completion.

## The raster graph

The raster graph draws six panels into a 160-by-50 cell grid.
Each cell is 8 by 16 pixels, producing a 1280-by-800 pixel buffer.
A high-priority stream separates raster scheduling from the main pump stream.

The device also publishes a two-slot shared-memory mirror.
The terminal receives a read-only descriptor and refuses torn snapshots through the sequence protocol.
That mirror is a display copy; it cannot become authoritative state.

A windowed instance uses a pump thread and a display thread.
Window key events reach the feeder as framed input, then become device records.
The [terminal](11-terminal.md) and [control client](13-control.md) provide their own presentation and lifecycle controls.

## The raw PTX modules

The build embeds the exact clock and Q8 matrix PTX text in the runtime.
The driver loads those bytes with `cuModuleLoadData`.
Execution does not reopen their source-tree files.

| Module | Entry | Loader |
| --- | --- | --- |
| `ptx/clock.ptx` | `aotx_clock_sample` | `cuda/boot/clock_host.cu`, `aotx_boot_clock_check` |
| `ptx/gemv_q8.ptx` | `aotx_gemv_q8` | `cuda/model/decode_module_host.cu` |

Boot samples the device clock before normal work.
The matrix module supplies a graph kernel node; device `.cu` functions do not call it as a linked function.
A failed module or symbol load follows the loader's explicit failure path.

## The disk side

| Program | Function |
| --- | --- |
| `aotx_drain` | Write synchronized journal blocks, bulk payloads and derived files. |
| `aotx_feed` | Publish input, imports and completed host-tool results. |
| `aotx_restore` | Read and replay authoritative records through the input transport. |
| `aotx_journal` | Inspect records and verify derived turn manifests. |
| `aotx_models`, `aotx_manifest` | Inspect, download, activate and verify model files. |
| `aotx_service` | Broker scoped local service packets. |
| `aotx_ccir_pack` | Create complete runtime containers from verified assets. |
| `aotx_tui` | Display device snapshots and send terminal input. |

File checksums and cryptographic asset digests run on the disk side.
The device maintains its own ordered state hash and protocol identities.
The journal outlives a lost CUDA context, but can lag behind current GPU state.

The boot program supervises essential children during execution and shutdown.
Unexpected writer, feeder or broker termination fails the run visibly.
A failed child cannot produce a successful shutdown result by being ignored.
See [operation](07-operation.md#how-a-run-stops).

## Memory and persistence

Local transcript memory supports hot, warm and summarized turns.
Optional typed memory stores exact sources, vectors, assertions, selections and appraisals with owner and scope fields.
Recall records its selected references so recovery does not repeat semantic selection.

Cold-memory offload retains exact versioned extents and validates them before retrieval.
Task reviews retain supported task outcomes with all required source references.
A complete [CCIR runtime](28-runtime-files.md) carries assets and state needed for file-only recovery.
A memory-only checkpoint has a narrower contract.

## Applications and trust

The [HTTP gateway](31-http-gateway.md) provides standard chat requests and native application resources.
Ordinary requests use submitted messages without creating learned conversation history.
The [shared service](33-shared-service.md) separately records participants, scopes, persistent conversations and operation receipts.
Neither interface depends on a particular frontend.

Host tools run with the operating-system account's rights and have no general sandbox.
Built-in file tools enforce their own root boundary.
Native device tools and creator policies are trusted executable code admitted by explicit identity checks.
[Security](../SECURITY.md) and the [tool SDK](10-tool-sdk.md) describe these boundaries.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
