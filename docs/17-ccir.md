# CCIR data-state files

`aotx_ccir` stores typed state sections in one `.aotxccir` file.
The data-state profile holds a manifest, an encoded checkpoint and an optional recorded tail.
The [live memory profile](25-memory-checkpoints.md) also stores required conversation bindings.
Neither profile contains a complete model runtime.

Packing checks file framing and section references.
The GPU state reader must check the encoded objects before use.
Inspection never loads code or fetches a URL.

Build with the normal CMake configuration.
The `aotx_ccir` target links the C-only `aotx_ccir_store` library.
The library header is `disk/ccir/ccir.h`.

## Commands

```
aotx_ccir pack state.aotxccir checkpoint.bin 00000000000000000000000000000001 64 64 100
aotx_ccir append state.aotxccir checkpoint.bin 64 66 105 tail.bin
aotx_ccir inspect state.aotxccir
aotx_ccir verify state.aotxccir other.aotxccir
aotx_ccir compact state.aotxccir small.aotxccir
```

Pack takes a destination, checkpoint, lineage ID, checkpoint sequence, durable sequence and source tick.
Append takes the same arguments without the lineage ID.
A final argument names a tail file.
The durable sequence cannot precede the checkpoint sequence.
A greater durable sequence requires a tail.
The input files contain bytes from the state encoder.

Create and compact refuse an existing destination.
Append keeps unknown optional sections from the selected generation.
It checks the source incarnation and commit under the writer lease before it writes.
If another writer changes that identity, append refuses with exit code 7.
Repeat the command after inspecting the new generation.
Directory order does not affect retention, and a new tail gets an unused section ID.

Compact copies the selected complete generation to a new file and keeps the source.
It preserves lineage and section IDs and creates a new incarnation ID.
The new file starts at generation 1 and records the source commit digest as history.

Inspect and verify check all selected section digests.
Output gives the selected generation, checkpoint sequence, durable sequence, tick, fallback state and trailing byte count.
Fallback means that one nonempty root or its generation failed validation.
Trailing bytes belong to data beyond the selected complete generation.
The reader does not scan those bytes for another commit.

Exit codes are 0 for success, 1 for file IO failure, and 2 for invalid input.
Code 3 means that a required format is not supported.
Code 4 means that a size limit is exceeded.
Code 5 means that the file is in use.
Code 6 means that the destination exists.
Code 7 means that the source file changed after the command read its directory.

An append IO error can occur after publication; reopen the file to check the selected generation.

## File framing

All integers use explicit little-endian encoding.
No native C structure is stored as framing.
Format 1.0 has no public compatibility guarantee.
All unspecified bytes are zero.
SHA-256 detects corruption; it does not identify a publisher.

The prologue occupies bytes 0 through 4095.

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 8 | `AOTXCCIR` |
| 8 | 2 | Major version, 1 |
| 10 | 2 | Minor version, 0 |
| 12 | 4 | Prologue length, 4096 |
| 16 | 8 | Required features, 0 |
| 24 | 16 | Nonzero lineage ID |
| 40 | 16 | Nonzero incarnation ID |
| 56 | 8 | Root A offset, 4096 |
| 64 | 8 | Root B offset, 8192 |
| 72 | 4 | Root length, 4096 |
| 76 | 4 | Directory row length, 128 |
| 80 | 8 | First data offset, 12288 |
| 4064 | 32 | SHA-256 of bytes 0 through 4063 |

Each root occupies 4096 bytes.
An all-zero root is empty.

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 8 | `AOTXROOT` |
| 8 | 2 | Root schema, 1 |
| 16 | 8 | Generation |
| 24 | 8 | Commit offset |
| 32 | 8 | Commit length, 256 |
| 40 | 8 | Declared file end |
| 48 | 32 | SHA-256 of all commit bytes |
| 80 | 32 | Prologue digest |
| 4064 | 32 | SHA-256 of root bytes 0 through 4063 |

The commit occupies 256 bytes and ends at the declared file end.
Its complete directory ends at the commit offset.

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 8 | `AOTXCMT1` |
| 8 | 8 | Generation |
| 16 | 32 | Previous commit digest; history only |
| 48 | 8 | Checkpoint sequence |
| 56 | 8 | Durable sequence |
| 64 | 8 | Source tick |
| 72 | 8 | Directory offset |
| 80 | 4 | Directory row count |
| 84 | 4 | Row length, 128 |
| 88 | 8 | Directory byte length |
| 96 | 32 | SHA-256 of all directory bytes |
| 128 | 8 | Declared file end |
| 136 | 16 | Manifest section ID |
| 152 | 16 | Checkpoint section ID |
| 168 | 16 | Tail section ID; zero when absent |

Directory rows retain this layout:

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 4 | Section type |
| 4 | 2 | Section schema |
| 6 | 2 | Flags; bit 0 means required |
| 8 | 16 | Nonzero section ID |
| 24 | 8 | Absolute file offset |
| 32 | 8 | Stored byte length |
| 40 | 8 | Logical byte length; equal to stored length |
| 48 | 4 | Alignment; power of two from 1 through 4096 |
| 52 | 2 | Codec, 0 for uncompressed bytes |
| 54 | 2 | Reserved |
| 56 | 32 | Stored-payload SHA-256 |
| 88 | 40 | Reserved |

The reader refuses duplicate IDs, overlap, wrapped ranges, unknown flag bits and nonzero reserved bytes.
Every selected extent lies between the prologue area and the selected directory.
Directory and commit offsets align to 128 bytes.
An extent can reuse validated bytes from an older generation.
Recovery needs no previous directory or commit.

## Data-state manifest

Types 1, 2 and 3 name the manifest, checkpoint and tail.
Each uses schema 1 and the required flag.
There is exactly one manifest and one checkpoint, and at most one tail.
Unknown optional types retain their IDs, schema and bytes during compaction.
Unknown required types or required schemas refuse use of an intact newer generation.
They do not cause silent fallback to an older mode.

The manifest is 96 bytes.
Bytes 0 through 7 contain `AOTXDATA`.
Four-byte values at offsets 8, 12, 16 and 20 are 1, 1, 256 and 1.
They name the manifest schema, object ABI, object record length and encoded-state representation.
The checkpoint ID starts at offset 24, and the tail ID starts at offset 40.
A missing tail has a zero ID.

Bytes 56 through 95 are zero.

## IO and resource limits

The library takes complete section batches and bounded read batches.
Default limits are 256 sections, 8 GiB per section and 16 GiB per file.
A caller can supply lower or higher byte limits within signed 64-bit file offsets.
The section-count ceiling remains 256.
Checksums and file copies use a fixed 64 KiB buffer.
No file-supplied length controls an allocation.

An open view holds a shared nonblocking file lease.
Close the view before an append or compaction call on the same file.
Use `aotx_ccir_append_if` for a directory made from a previously opened view.
Copy its prologue and commit digests into `aotx_ccir_revision` before closing the view.
The ordinary `aotx_ccir_append` API still accepts an intentional complete replacement directory.

Writers hold exclusive nonblocking leases on regular local files.
These advisory leases exclude cooperating writers; they are not access control.
Network filesystem and distributed writer behavior are not supported.

Append writes payloads and a complete directory, then calls `fsync`.
It writes the commit and calls `fsync` again.
It writes the other root slot and calls `fsync` before success.
Create and compact also verify the result and sync its parent directory before success.
An interrupted new file can be invalid; it has no acknowledged generation.
Process-interruption checks do not prove power-loss behavior for every filesystem or storage device.
