<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Complete runtime files

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

A complete runtime uses one `.aotxccir` file for its components and durable state.
It contains original model files, model metadata, data modules, device settings and prepared memory.
After activation, it also contains the runtime recovery log and complete live memory checkpoints.
Active memory and state remain on the GPU. The disk file is the durable mirror.

This profile requires a compatible installed AOTX build and CUDA device.
The original component directories and network access are not required after packaging.
Ordinary model stores, base conversations and memory-only checkpoints remain available.

<details>
<summary>On this page</summary>

- [Package contents and limits](#package-contents-and-limits)
- [Create a runtime](#create-a-runtime)
- [Inspect and activate](#inspect-and-activate)
- [State and durability](#state-and-durability)
- [Copy a completed runtime](#copy-a-completed-runtime)
- [Capacity and write cost](#capacity-and-write-cost)
- [Format](#format)
- [Required runtime features](#required-runtime-features)

</details>

## Package contents and limits

| Included | Supplied by deployment |
| --- | --- |
| Selected exact model and data-module bytes | Compatible installed AOTX binaries and CUDA driver. |
| Device settings and persistent scope state | Journal path, terminal settings and service credentials. |
| Required policy and control components | Explicit trust for required native policy code. |
| Typed checkpoints and complete recorded history | Enough GPU memory, disk capacity and supported file semantics. |

A container digest verifies stored bytes, not their publisher's trustworthiness.
Inspection is inert; activation can load trusted native components.

## Create a runtime

Prepare a complete memory checkpoint with the [typed state tools](18-typed-state.md).
The input must contain the complete prepared state, without an unapplied tail.
Select a local model store with verified manifest entries and data-only role modules.
Include the conductor role in the module directory.

```text
aotx_ccir_pack --memory prepared.aotxccir --models STORE --roles language \
  --modules MODULES --settings SETTINGS --output identity.aotxccir \
  --phrases refusal-phrases.txt
```

`--settings` is optional. Omission selects the device defaults.
The packager copies device settings only. Deployment paths, credentials and terminal settings remain outside the file.
An affect build requires `quality/refusal-phrases.txt` in the store or a file supplied with `--phrases`.

The packager creates a new runtime from the selected components and prepared memory.
Its initial state has no bound users or inherited affect history.
The prepared memory is retained. The selected modules and settings define the initial identity.
Use a completed runtime file to transfer existing live identity and affect state.

Add `--shared` for [persistent shared conversations and operation receipts](33-shared-service.md).
An existing complete file can enable this required profile while its runtime is stopped.

The model manifest carries source, revision, license, byte count and SHA-256 fields.
The packager preserves original GGUF bytes and relocates their names within the container.
It checks model metadata, required files, data module references and selected model roles.
An existing output pathname is refused.

Data-only role and skill modules are supported.
This profile refuses imported tool programs.
Optional [creator policies](34-creator-policy.md) can contain a required native CUDA or PTX entry.
Built-in tools remain part of the installed AOTX code.
An operator can supply a deployment tool root with the normal boot option.

## Inspect and activate

```text
aotx_ccir inspect identity.aotxccir
aotx_ccir verify identity.aotxccir
aotx_boot --ccir identity.aotxccir --journal JOURNAL
```

Inspection verifies container digests and the required text component references.
It loads no CUDA code and executes no component.
Optional [image input](29-image-input.md) adds required vision weights, a paired manifest and device capacities.
Optional [audio input](30-audio-input.md) adds an independent audio parent, encoder and capacity profile.
Activation checks the runtime ABI, wire layout, resource requirements, architecture and affect build option.
An incompatible required profile is refused.

`--ccir` supplies models, model roles, modules, settings, restore state and the complete mirror path.
Do not combine it with the separate options for those components.
The runtime selects creation or recovery from the file's replay header.
New input starts only after restored state reaches a complete durable generation.

File validation and the first durable write have no fixed time limit.
Startup waits while the disk programs work. A stop request, writer exit or disk error ends the wait.

Model consumers read bounded extents directly from the container.
Settings and data modules use private temporary metadata files under the journal directory.
An included creator policy bundle uses the same private directory.
Those temporary files contain no model weights and are removed on normal exit.
An abrupt process kill can leave them in the old journal directory.

## State and durability

The runtime captures a completed-work boundary across all active agents.
Pending tasks, inference, tools, module imports, image work and memory transfers delay capture.
An incomplete creator policy decision also delays capture until its final recorded fragment.
Active [appraisal work](35-automatic-appraisal.md) delays capture until its complete result is recorded and applied.
A continuously busy runtime can delay its complete checkpoint.

The file includes a typed memory image and the matching complete journal prefix.
The journal restores settings, catalog state, affect, agent progression, sampled tokens and memory operations.
It is a recovery representation. Memory-bound prompts still use selected GPU memory and current input.
Unbound conversations retain the base transcript policy.

The drain waits for the captured tick's journal commit before publishing a generation.
Settings and identity changes request publication even when the accepted memory operation count does not change.
Status-only CLI reads remain in the journal but do not request another checkpoint.

The `memory` command adds runtime progress to the existing mirror counters:

```text
memory mirror: committed 6 durable 6 generation 5 pending 0 error 0 runtime source 2500 durable 2500
```

The first pair counts accepted memory operations.
`runtime source` names the latest record that requests complete runtime publication.
The final `durable` value names the source record covered by the acknowledged file generation.
These record numbers apply to the current boot.

Before stopping, wait for both durable values to cover their observed source values, with zero error.
An abrupt stop can lose work after the last acknowledged boundary.
A disk error retains the pending image and reports persistence pressure.
New memory operations can be refused while the transport is full or the disk reports an error.

Recovery must reproduce the recorded input count and state hash on the GPU.
Replayed records retain their original source identity through later saves and restores.
It must also reproduce the file's typed memory and bindings before replacing the saved replay state.
A mismatch stops activation before new input.

## Copy a completed runtime

Wait for the required state to become durable.
Stop the runtime and let its drain close.
Copy the selected generation into a new file:

```text
aotx_ccir compact identity.aotxccir portable.aotxccir
aotx_boot --ccir portable.aotxccir --journal NEW_JOURNAL
```

The destination must not exist.
Compaction preserves all selected sections, including unknown optional sections.
The new file can start without the old journal or component directories.
Its restored conversations accept new input with their saved user bindings and memory.

The running mirror keeps a writer ownership lease. Other writers and source compaction are refused during that lease.
Readers can inspect between publications. A conflicting lease returns file-in-use status.
File leases are advisory and require cooperating processes.

The first mirror write must use the generation checked at startup.
If another writer changes the file during startup, activation stops before new input.

## Capacity and write cost

The configured CCIR file cap applies to the complete file, including models and recovery data.
The default is 16 GiB. Set `AOTX_CCIR_FILE_BYTES` through CMake; zero selects the system offset limit.
Individual sections use the same default byte cap. Explicit library limits can be smaller.
The directory holds at most 256 sections; the runtime index binds each required asset to a section.

Active memory capacity remains controlled by the existing GPU memory configuration.
Changing a disk file cap does not increase GPU capacity.

This profile retains the complete replay history and copies its prefix during each publication.
The temporary replay file is bounded by the configured section cap.
Normal append reuses unchanged model extents and does not reread their bytes after a successful write.
File opens verify all selected payload digests.

Memory reclamation or excess appended data can cause replacement of the container.
Replacement copies the retained components, including model weights, into a new file incarnation.
This needs temporary disk space and can delay the drain on a large file.
The old generation remains authoritative until replacement succeeds.

## Format

The container uses [CCIR manifest schema 3](17-ccir.md).
It names required checkpoint, live state and runtime index sections.
Section type 5 holds the runtime index, type 6 holds an asset, and type 7 holds replay state.
Asset sections can repeat. Their logical names and section IDs must be unique.
Each recorded length and digest must match its section.

The runtime index uses a 256-byte header and 384-byte asset rows.
Runtime index schemas extend the same header and asset rows.
Required feature bit 32 selects appraisal, which needs schema 4 or a later required-feature schema.
The packager selects this profile when prepared memory contains appraisal configuration, queues or generated evidence.
The mirror also selects it when retained memory or recorded appraisal work requires the profile.
Its processor and selected language model digests occupy the reserved header tail, preserving the shared-runtime capacity fields.

All retained historical model digests must refer to packaged language assets.
Missing dependencies refuse publication and preserve the previous complete generation.

The replay header is 128 bytes, followed by framed complete journal blocks for a saved runtime.
Creation mode has no replay blocks. Integer fields are explicitly encoded as little endian bytes.
The exact offsets are defined in `disk/runtime/runtime.h` and `disk/runtime/replay.h`.

## Required runtime features

| Feature bit | Meaning | Required index schema |
| --- | --- | --- |
| 1 | Affect build support. | Compatible base profile. |
| 2 | Vision assets and capacities. | Compatible base profile. |
| 4 | Audio assets and capacities. | Compatible base profile. |
| 8 | Persistent shared tables. | At least 2. |
| 16 | Creator-policy state. | At least 3. |
| 32 | Appraisal state and processor identity. | At least 4. |
| 64 | Compatible policy-update history. | At least 5. |
| 128 | Cold extents. | At least 6. |
| 256 | Task review state and recovery. | 7. |

The writer chooses the schema required by the complete feature set.
A reader must support every required feature; an older reader cannot discard one and continue.
The source constants are in `disk/runtime/runtime.h`; `aotx_runtime_schema` selects the index schema.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
