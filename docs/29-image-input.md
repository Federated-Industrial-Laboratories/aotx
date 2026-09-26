<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Native image input

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

The optional image component accepts JPEG files and packed RGB8 pixels.
CUDA validates, decodes, resizes and encodes the image into native language model input rows.
The file reader transfers bytes only. It uses no host image codec or image inference.

The supported trained pair is Qwen3.5-0.8B with its F16 vision GGUF component.
Other text model stores can continue without the image component.
Native sound input uses the separate [audio component](30-audio-input.md).
Video decoding, camera drivers and network retrieval are not supported by this interface.
A producer can submit camera frames as separate JPEG or RGB8 sources.

<details>
<summary>On this page</summary>

- [Before starting](#before-starting)
- [Prepare the model store](#prepare-the-model-store)
- [Submit a file](#submit-a-file)
- [Image formats](#image-formats)
- [Device capacity](#device-capacity)
- [Producer transport](#producer-transport)
- [Recovery and storage](#recovery-and-storage)

</details>

## Before starting

Build the image path, prepare the exact model pair, and reserve source, feature and workspace capacity.
Use [the gateway](31-http-gateway.md) for authenticated remote uploads.
The local commands below read files on the runtime host.

The paired workflow checks source ownership, input routing and copied-file recovery.
It does not qualify general image recognition or visual reasoning accuracy.

## Prepare the model store

Use a verified language model entry in `manifest.jsonl`.
Place the matching vision GGUF in the same store.
Create `vision.jsonl` with exactly two entries, in this order:

1. Copy the selected language entry from `manifest.jsonl`.
2. Add the vision entry with role `vision`.

Both entries require name, role, path, source, revision, license, SHA-256 and byte count.
Their names, paths and digests must differ.
The language entry must match the selected model entry exactly.
The vision component has 154 tensors, 12 blocks, width 768 and output width 1024.
The reader checks each tensor type, shape and extent before allocation.

The supported component pair uses these source files:

| Field | Value |
| --- | --- |
| Source | `unsloth/Qwen3.5-0.8B-GGUF` |
| Revision | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| Language file | `Qwen3.5-0.8B-Q8_0.gguf` |
| Language bytes | 811843840 |
| Language SHA-256 | `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c` |
| Vision file | `mmproj-F16.gguf` |
| Vision bytes | 204987232 |
| Vision SHA-256 | `56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453` |
| License | Apache-2.0 |

Use the manifest wrap fields described in [model files](16-model-files.md).
Select the paired language role at startup.
A model change cannot replace that role with a different digest while the image component is active.

## Submit a file

Use the terminal console or a complete command line through the local attach interface:

```text
image load 0 private jpeg /absolute/path/image.jpg
image load 0 private rgb8 640 480 /absolute/path/pixels.rgb
```

The RGB8 file contains packed red, green and blue bytes, without row padding.
Its exact length must equal width times height times three.
The remainder of the command is the pathname; spaces do not require quotes.
The reader refuses symlinks, nonregular files and files that change during transfer.
It reads the file on the runtime host.

`private` limits a source to its bound principal and room.
Without a memory binding, it limits the source to the addressed base conversation slot.
`room` permits readers with the same bound room.
`shared` permits readers throughout the running instance.
A transfer command does not change conversation bindings or memory policy.

The console displays a source reference and a transfer ID:

```text
image: [image:SHA256] transfer TRANSFER_ID
```

Replace `SHA256` with the displayed 64 lowercase hexadecimal characters.
Use the reference at the required position in a normal message:

```text
say [image:SHA256] Describe the visible animal.
```

Several references can occur in one message, in the required order.
The source can also appear in a bound text-memory request.
CUDA resolves every reference under the requesting slot's scope.
A request waits for an admitted source to finish encoding.
Missing, refused, out-of-scope or malformed references refuse the prompt.
The digest and pathname do not become ordinary model prompt text.

The reply uses the image rows at the reference position.
Visual rows count toward the sequence token limit.
A plain conversation can retire older hot turns when its expanded prompt exceeds that limit.
The current turn and system block remain intact.
Required cognitive context is not removed to fit an image.
An individual request that cannot fit is refused.

Cancel or retire an owned source with its 32-character transfer ID:

```text
image cancel 0 TRANSFER_ID
```

The device refuses cancellation from another slot or a changed private binding.
An active sequence keeps its source and feature reservation until completion.
Cancellation after completion releases those reservations.
Later references to a retired source refuse; upload it again before reuse.

Refusal notices include a numeric status:

| Status | Meaning |
| --- | --- |
| 1 | Invalid source or transport |
| 2 | Capacity limit |
| 3 | Source digest mismatch |
| 4 | Canceled source |
| 5 | Unsupported or invalid JPEG/RGB8 data |
| 6 | Vision processing failure or extent refusal |
| 7 | Missing component, transfer or owner permission |
| 8 | Active feature lease |

## Image formats

JPEG support covers 8-bit Huffman baseline and progressive scans.
Supported color forms are grayscale and JFIF YCbCr with 4:4:4, 4:2:2 or 4:2:0 sampling.
The decoder supports progressive AC refinement and restart intervals.
Unsupported color profiles, orientation metadata and codec variants are refused.
The source file SHA-256 identifies a JPEG without conversion.

The RGB8 transport representation starts with a 24-byte header.
It contains `AOTXRGB1`, width at byte 8, height at byte 12 and pixel byte length at byte 16.
Dimensions use unsigned 32-bit little-endian values; length uses an unsigned 64-bit value.
The source digest covers this header and all pixels.
The file command supplies the header automatically.
A custom producer must include it in the source bytes.

## Device capacity

Each capacity is independent and configurable through CMake:

| Option | Default | Resource |
| --- | --- | --- |
| `AOTX_MEDIA_OBJECTS` | 128 | Resident source descriptors |
| `AOTX_MEDIA_BYTES` | 67108864 | Total encoded source bytes |
| `AOTX_MEDIA_FEATURE_ROWS` | 131072 | Resident rows of 1024 float values |
| `AOTX_MEDIA_WORKERS` | 1 | Simultaneous codec and encoder workspaces |
| `AOTX_MEDIA_PIXELS` | 4194304 | Decoded pixels per workspace |
| `AOTX_MEDIA_DIMENSION` | 8192 | Maximum decoded width or height |
| `AOTX_MEDIA_PATCHES` | 8192 | Resized vision patches per workspace |
| `AOTX_MEDIA_HORIZONTAL` | 16777216 | Float values for horizontal resize |

Workspace scheduling processes the admitted batch within these capacities.
It does not allocate maximum image scratch for every conversation slot.
Feature storage uses 4096 bytes per row, plus source, descriptor, weights and workspace storage.
Startup reports its device allocation and refuses requirements that exceed available VRAM.
A source-byte refusal is acknowledged before the file reader streams its payload.

The trained processor resizes to multiples of 32, with at least 65536 pixels.
Its trained maximum is 16777216 resized pixels.
A configured workspace can impose a smaller explicit limit.
Each group of four patches produces one language input row.
The conversation token budget, source storage and processor extent are separate limits.

An optional `media.profile` file records smaller capacities for a portable runtime.
Its exact length is 80 bytes. All numbers are little-endian.

| Byte | Field |
| --- | --- |
| 0 | Eight bytes `AOTXIM01` |
| 8 | Schema 1, unsigned 32-bit |
| 12 | Object count, unsigned 32-bit |
| 16 | Source bytes, unsigned 64-bit |
| 24, 28, 32, 36, 40 | Feature rows, workers, pixels, dimension, patches; unsigned 32-bit |
| 44 | Four zero bytes |
| 48 | Horizontal float capacity, unsigned 64-bit |
| 56 | 24 zero bytes |

Omission from a model directory selects the build defaults.
A complete runtime includes this profile and requires a build that can hold every capacity.
Active source bytes and features remain on the GPU; no automatic disk offload occurs.

## Producer transport

`cuda/media/wire.h` defines the mapped ring and canonical source records.
One producer releases complete frames; CUDA releases consumed frame slots and a result status.
The ring has 16 frames of 65536 bytes, with a 64-byte frame header.
Begin, chunk, end and cancel operations use distinct transfer IDs and contiguous source offsets.
The disk producer API is `disk/feed/media_io.h`.
It provides interruption-aware publication and acknowledgment waits.

CUDA captures scope from the current slot binding and verifies the complete source digest.
Gaps and premature end terminate the transfer. A closed producer cancels unfinished sources.
The source descriptor and byte capacities bound admission.
Numerical decoding and vision work yield between finite graph steps.

## Recovery and storage

[Complete runtime files](28-runtime-files.md) include the vision weights, pair manifest and media capacities.
Their recovery log retains the exact canonical source bytes.
Recovery rebuilds image features before admitting new prompts.
A copied complete file therefore accepts fresh image questions without the original image or model paths.

Retirement releases GPU space. It does not remove historical source records from the recovery log.
Container compaction preserves the selected recovery section, including these records.
Its storage and rewrite costs grow with retained source history.
The complete file cap and completed-work durability rules still apply.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
