<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Documentation

[Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

These manuals describe the current source tree. Tagged releases retain the documentation at their own revision.
Start with the architecture, then build and operation. Application developers can continue with the HTTP and shared service guides.

<details>
<summary>On this page</summary>

- [Choose a task](#choose-a-task)
- [Start and operate](#start-and-operate)
- [Connect applications and media](#connect-applications-and-media)
- [Use persistent memory](#use-persistent-memory)
- [Extend and configure behavior](#extend-and-configure-behavior)
- [Read formats and measurements](#read-formats-and-measurements)
- [Shared terms](#shared-terms)
- [Conventions](#conventions)

</details>

## Choose a task

| Task | Guide |
| --- | --- |
| Build and start a first instance | [Build](06-build.md), then [operation](07-operation.md) |
| Use an existing GGUF file | [Model files](16-model-files.md#use-a-file-already-on-disk) |
| Connect a standard chat client | [HTTP gateway](31-http-gateway.md#standard-requests) |
| Build a persistent conversation client | [Shared service](33-shared-service.md) |
| Enable source-linked memory | [Live memory](20-live-memory.md), then [semantic memory](27-semantic-memory.md) |
| Copy an instance to another machine | [Complete runtime files](28-runtime-files.md#copy-a-completed-runtime) |
| Inspect a failed run | [Operation](07-operation.md#failures-and-recovery) and [journal format](04-journal-format.md) |
| Check a capability before enabling it | [Support and qualification](support.md) |

## Start and operate

| Manual | Contents |
| --- | --- |
| [Architecture](01-architecture.md) | State ownership, execution and component boundaries. |
| [Build](06-build.md) | Requirements, card profiles and capacity settings. |
| [Operation](07-operation.md) | Start, inspect, stop and restore an instance. |
| [Settings](settings.md) | Startup options and values that affect later turns. |
| [Console commands](commands.md) | Agents, tools, models and memory commands. |
| [Terminal](11-terminal.md) | Attach, screens, keys and terminal settings. |
| [Graphical control](13-control.md) | Local instance and conversation windows. |
| [Model files](16-model-files.md) | Inspection, catalogs, wrappers and model selection. |
| [Support and qualification](support.md) | Current feature, model and hardware boundaries. |
| [Testing](testing.md) | Source gates, focused checks and runtime acceptance. |

## Connect applications and media

| Manual | Contents |
| --- | --- |
| [HTTP gateway](31-http-gateway.md) | Deployment, credentials and standard chat requests. |
| [Native service resources](32-service-wire.md) | Request handles, events, capabilities and broker framing. |
| [Shared service](33-shared-service.md) | Participants, spaces, persistent conversations and retries. |
| [Image input](29-image-input.md) | Image models, source formats and device capacity. |
| [Audio input](30-audio-input.md) | Audio models, source formats and combined media capacity. |

## Use persistent memory

| Manual | Contents |
| --- | --- |
| [Live memory](20-live-memory.md) | Bind conversations to scoped typed state. |
| [Text requests](21-text-memory.md) | Prepare queries through the embedding model. |
| [Input retention](22-memory-retention.md) | Save exact source text, vectors and working focus. |
| [Automatic retention](23-automatic-memory.md) | Retain accepted input and its recorded selection. |
| [Semantic memory](27-semantic-memory.md) | Qualified interpretation, source spans and corrections. |
| [Contextual recall](24-contextual-memory.md) | Task, participant, evidence and source-group selection. |
| [Automatic appraisal](35-automatic-appraisal.md) | Supported benefit, harm and relationship evidence. |
| [Task reviews](38-task-reviews.md) | Task-scoped outcome cues with exact supporting evidence. |
| [Checkpoints](25-memory-checkpoints.md) | Live durability, checkpoint pressure and resume. |
| [Maintenance](26-memory-maintenance.md) | Reclamation, retry state and file shrinking. |
| [Cold memory](36-cold-memory.md) | Explicit offload, retrieval and portable cold extents. |

## Extend and configure behavior

| Manual | Contents |
| --- | --- |
| [Modules](09-modules.md) | Role, skill and tool installation with effective tool selection. |
| [Tool SDK](10-tool-sdk.md) | Batched CUDA tools and trusted host tools. |
| [Creator policies](34-creator-policy.md) | Data or native scheduling rules and compatible updates. |
| [Control bindings](37-control-bindings.md) | Exact model identity, accepted doses and application selection. |
| [Conduct controls](12-conduct.md) | Vector authoring, calibration and local conduct commands. |
| [Affect and quality streams](14-affect.md) | State equations, probes, actuation and recovery. |
| [Quality measurement](15-quality.md) | Task scoring, paired conversations and fixture interpretation. |

## Read formats and measurements

| Manual | Contents |
| --- | --- |
| [Time and recovery](02-temporal-model.md) | Ticks, authoritative records and failure semantics. |
| [Host-device boundary](03-seam-contract.md) | Ring layouts, publication and mapped memory. |
| [Journal format](04-journal-format.md) | Segments, records, payloads and derived files. |
| [Message bus](05-bus-schema.md) | Message kinds, provenance and JSON line output. |
| [CCIR container](17-ccir.md) | Bounded file transactions, sections and data-state manifests. |
| [Complete runtime files](28-runtime-files.md) | Pack, activate, copy and recover complete instances. |
| [Typed state](18-typed-state.md) | Object records, payload schemas and configured limits. |
| [Prepared recall](19-prepared-memory.md) | Query records, selection references and exact replay. |
| [Historical measurements](08-measured.md) | Version-bound performance and control measurements. |
| [Architecture accuracy](17-accuracy.md) | Fixed numerical references, unresolved results and reproduction. |
| [Writing conventions](00-writing.md) | Terminology, sentence structure and source references. |

## Shared terms

| Term | Meaning |
| --- | --- |
| Runtime | One running AOTX instance on one GPU. |
| Profile | Build-time table capacities for a class of GPU. |
| Slot | A bounded device row used by an agent or sequence. |
| Tick | One ordered execution of the device scheduling graph. |
| Seam | The host-device boundary, crossed through bounded mapped transport. |
| Ring | A single-producer, single-consumer buffer that carries records or payloads. |
| Journal | Ordered records used to recover the last complete durable tick. |
| Drain | The disk program that writes and synchronizes journal blocks. |
| Feeder | The disk program that publishes input and completed host-tool results. |
| Replay | Reapplication of recorded authoritative inputs and decisions. |
| Mirror | A display snapshot or disk copy; it does not own live state. |
| Catalog | The device registry of imported roles, skills and tools. |
| Model store | Disk files and manifests that identify model assets. |
| Arena | A contiguous memory region with offset-based allocation. |
| Pump | Host glue that launches the device graph and services its transport. |
| Source | An exact retained input with identity, owner, scope and version. |
| Appraisal | A source-backed interpretation of benefit, harm or relationship evidence. |
| Qualification | Acceptance for an exact model, processor or control package and declared behavior. |
| Principal | An authenticated service identity with explicit grants. |
| CCIR | The container format for typed state or a complete runtime package. |

## Conventions

Commands run from the repository root unless a guide gives another directory.
Uppercase paths and IDs are placeholders. Replace them with the intended local values.
File offsets are byte offsets. Binary layouts state their byte order and schema version.

A successful build establishes compilation. A passed structural check does not establish a model's semantic behavior.
Read [testing](testing.md) and [support](support.md) before applying measured results to another configuration.
[Security](../SECURITY.md) describes trusted files, executable modules and service credentials.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
