<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Cold payload storage

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Cold memory keeps object metadata on the GPU and selected payload bytes in the complete CCIR file.
The GPU controls selection, access, dependency checks and publication.
Offload releases usable capacity in the resident payload arena.
It does not reduce the CUDA allocation reserved for that arena.

<details>
<summary>On this page</summary>

- [Enable and operate](#enable-and-operate)
- [Readiness and failures](#readiness-and-failures)
- [Control file](#control-file)
- [Storage and recovery](#storage-and-recovery)
- [Verify storage and recovery](#verify-storage-and-recovery)

</details>

## Enable and operate

Start the runtime with `--ccir FILE` or a durable `--memory-mirror FILE`.
Wait until `memory` reports equal committed and durable revisions with no error.
Submit a control file with `memory tier FILE`.
Each control takes the current store lineage and sequence.
A stale sequence refuses the complete operation.

| Mode | Value | Operation |
| --- | --- | --- |
| Enable | 4 | Enable explicit offload. Take no object rows. |
| Offload | 1 | Remove the selected resident payloads from the arena. |
| Fetch | 2 | Restore selected cold payloads and their required cold sources. |
| GPU only | 3 | Check the complete retained size, fetch all cold payloads, then disable offload. |

Offload and fetch take 1 to 64 distinct object rows.
Each row names an exact current version, principal and room.
One failed row refuses the complete batch.
Fetch can restore more objects than the input count when dependencies require them.
The result reports the number of changed objects.

Offload accepts typed text sources, assertions, cues, participant records and media payloads.
It refuses protected objects, tombstones, current conversation selections and payloads needed by resident objects.
Offload dependent objects before their sources, or submit the complete group together.
Policy, appraisal, query and selection payloads stay resident.
No pressure rule enables offload without an explicit control.

## Readiness and failures

`memory` reports the mode, cold object count, cold byte count and pending read state.
An operation that requires cold evidence returns status 12 with `unavailable`.
It does not silently omit that evidence or replace a current correction with an old version.
Fetch the required objects before retrying the operation.
`memory cancel` cancels a pending read without waiting for disk progress.

Reads use a separate disk worker and one bounded transport slot.
Console commands continue while a read waits.
Other memory operations refuse while the residency operation is pending.
An unavailable file, damaged payload, changed generation, cancellation or read timeout leaves the retained store unchanged.
A late response cannot publish a cancelled operation.

GPU-only mode checks all retained payload bytes before it starts retrieval.
If capacity is too small, the complete store stays in tiered mode.
Use an explicit retention policy or a build with enough payload capacity before retrying.
The GPU-only control is a local operator action and covers all retained scopes.

## Control file

Integers use little-endian encoding. Unused bytes are zero.
The file has a 64-byte header followed by 64-byte rows.

| Header offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `AOTXTIR1` |
| 8 | 4 | Schema 1 |
| 12 | 4 | Mode |
| 16 | 4 | Input row count |
| 20 | 4 | Row size, 64 |
| 24 | 16 | Store lineage |
| 40 | 8 | Current store sequence |
| 48 | 16 | Zero |

| Row offset | Bytes | Field |
| --- | --- | --- |
| 0 | 16 | Object ID |
| 16 | 8 | Object version |
| 24 | 16 | Principal ID |
| 40 | 16 | Room ID, or zero for private scope |
| 56 | 8 | Zero |

The console returns `memory: operation 18 status S rows N`.
Status 0 means the GPU accepted the operation.
Wait for its durable revision before copying the complete file or stopping its disk writer.

## Storage and recovery

Tiered checkpoints use object schema 3 and require CCIR section type 8.
Cold payloads require runtime index schema 6 or later and feature bit 128.
Earlier readers refuse these required schemas.
The cold section contains exact object metadata, extent offsets, payload lengths and per-payload SHA-256 digests.
The generation contains every referenced cold extent before its root is published.

Each catalog read checks the complete cold section against its committed digest in bounded chunks.
The read worker then checks each requested payload against that verified catalog.

File-only recovery restores cold metadata without loading every cold payload into the GPU arena.
Retrieval checks the exact source generation, object metadata, digest, scope and payload schema before publication.
Cold payload schema checks occur when those bytes become resident.
No consumer may use unavailable payload bytes before that check.

Class A result records carry the exact retrieved bytes.
Journal replay uses those records without an external cold read.
An interrupted result records an unavailable outcome and preserves the previous store.
Repeated recovery accepts that interruption marker.

## Verify storage and recovery

Run `aotx_cold_memory_test`, `aotx_cold_guards_test` and `aotx_cold_closure_test`.
Each executable uses distinct batches at N=1 and N=64.
Run the console and complete-file workflow with an existing model store:

```text
python3 tests/cold_boot_test.py BUILD SOURCE STORE NEW_OUTPUT 1
python3 tests/cold_boot_test.py BUILD SOURCE STORE NEW_OUTPUT_64 64
```

These checks use prepared payloads and do not qualify model interpretation.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
