<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Live memory bindings

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Bind a fresh conversation slot to a typed GPU store when its prompts must use scoped selected memory.
A binding fixes lineage, principal, room, conversation identity, scope and page cap.
It cannot replace an existing binding or adopt a slot with earlier turns.
Unbound conversations keep their transcript-based input path.

<details>
<summary>On this page</summary>

- [Choose a retention mode](#choose-a-retention-mode)
- [Commands and prerequisites](#commands-and-prerequisites)
- [Byte layout](#byte-layout)
- [State, prompts and recovery](#state-prompts-and-recovery)
- [Extended operations](#extended-operations)

</details>

## Choose a retention mode

| Bind value at offset 60 | Behavior |
| --- | --- |
| 0 | Retain input only through an explicit retention command. |
| 1 | Retain each accepted source and prepared vector automatically. |
| 2 | Add qualified semantic interpretation to automatic source retention. |

Mode 2 requires an exact accepted model, wrapper and processor combination.
See [semantic memory qualification](27-semantic-memory.md#qualified-automatic-memory).
Text query preparation also requires the embedding model; prepared queries supply their own vectors.

## Commands and prerequisites

The local operator can use these commands from standard input or an attached input file:

```text
memory load PATH
memory apply PATH
memory bind PATH
memory query PATH
memory text PATH
memory retain PATH
memory resume PATH
```

`resume` restores a [live memory checkpoint](25-memory-checkpoints.md), including its required conversation bindings.

`load` reads the verified checkpoint and tail from a CCIR file. `apply` reads a canonical
typed tail. `bind` and `query` read raw batch files with the layouts below. A path is the
rest of the line, with spaces kept as path bytes. There is no shell expansion or quote
processing.

`text` reads the same batch shape with zeroed vector fields and a 2,048-byte input limit.
The GPU prepares those vectors with the loaded embedding model.
[Text requests](21-text-memory.md) defines its required model roles and exact byte layout.

`retain` names the last accepted input of each idle binding. The device retains its
exact text and prepared vector. See [Retain accepted input](22-memory-retention.md) for
the request, recorded mutation and optional working focus.

Files must be bounded regular files. The reader refuses symbolic links,
pipes, directories, incomplete reads and extra bytes after the declared file size.

Read and framing refusals appear on standard error and as a console note.

The disk reader checks framing and transmits exact bytes. The device checks the full
batch before it changes state. It checks IDs, versions, bindings, ordinals, scope, the
declared store cut and the prepared embedding space. The reader does not search memory
or construct prompts. `memory choice` is not an input command for typed choices.

## Byte layout

All integers are unsigned and little-endian. IDs are 16 bytes. The shared constants
and offsets are in `cuda/cognitive/live.h`. A batch has 1 to 64 rows. One query batch
waits for its recorded choice before the next typed transfer is admitted.

Each class A record of type 33 has a 32-byte prefix and at most 160 data bytes:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 4 | Schema 1 |
| 4 | 4 | Operation: load 1, update 2, bind 3, query 4, choice 5, text 6, text choice 7, retain 8, retained 9, automatic choice 10, resume 11 |
| 8 | 16 | Nonzero transfer ID |
| 24 | 4 | Total transfer bytes |
| 28 | 4 | Data offset in the transfer |
| 32 | 1 to 160 | Exact data bytes |

Parts are in order. Each part has 160 data bytes except the final part. The feeder
publishes at most 32 records per group and waits for ring space between groups. Each
file transfer gets a new random ID. A load has two 64-bit lengths at offsets 0 and 8,
then the exact checkpoint bytes and exact tail bytes.

An update is one typed tail.
The maximum load is `16 + 2 * image_bytes`, using the compiled image capacity.
It is 37,749,008 bytes with the default capacity, including the 16-byte length prefix.

Bind, query and choice have a 64-byte header. Their magic bytes are `AOTXBND1`,
`AOTXLIV1` and `AOTXCHO1`. Count and schema are 32-bit fields at 8 and 12; lineage is
at 16. The 64-bit store sequence is at 32, and the 32-bit row size is at 40. Bind and
query bytes 44 through 63 are zero. A choice has a 32-bit status at 44 and zero bytes
48 through 63.

Status 0 means success; a nonzero status from the [typed status table](18-typed-state.md#status-codes) is a refusal with count 0
and no rows. A choice has the same transfer ID as its query. Only the device emits it.

| Row | Bytes | Fields |
| --- | --- | --- |
| Bind | 64 | Slot at 0, scope at 4, principal at 8, room at 24, conversation at 40, page cap at 56, automatic retention at 60 |
| Query | 8,256 | Slot at 0, focus flag at 4, zero 8 through 15, conversation at 16, 64-bit ordinal at 32, zero 40 through 63, prepared query at 64 |
| Choice | 592 | Exact 64-byte query prefix, then the 528-byte ordered selection buffer |

Slots, scope, page caps and automatic retention are 32-bit values.
Automatic retention is 0 for explicit retention, 1 for source retention or 2 for qualified semantic intake.
See [automatic input retention](23-automatic-memory.md) for combined decisions and admission. The prepared query and selection layout
are defined in [prepared memory](19-prepared-memory.md).

Ordinals start at 1 for each
binding. The largest bind is 4,160 bytes; the largest query is 528,448 bytes; the
largest choice is 37,952 bytes. A choice keeps zero bytes in unused selection slots.

The 32-bit focus flag is 0 for caller references only. Flag 1 appends the binding's
working references after explicit focus, as defined in [retention](22-memory-retention.md).

## State, prompts and recovery

The resident typed store uses the [configured capacity](18-typed-state.md#configured-capacity).
The defaults are 8,192 immutable object versions and 16 MiB of payload per store.
Explicit updates require idle cognitive conversations. Each binding keeps bounded
current request and choice state. Later queries replace this current state; the
journal keeps prior requests. The 64-pair saved-query limit of the offline recall
file does not limit the number of live turns.

Prepared queries supply their own vectors;
text requests use the GPU embedding service. Explicit retention creates memory from
accepted input. Automatic bindings retain each input during admission.
Other memory changes require typed operations.
[Cold memory](36-cold-memory.md) supplies an explicit offload and retrieval path; binding alone does not enable it.

The device uses the model's loaded prompt format, labelled selected memory and current
input. It checks context and page limits. Tool continuations retain the bound context
with the current call and result, and check selected dependencies again. Bound prompts
do not take their memory from the audit transcript.

Load, update, bind, query and choice bytes are in the class A journal. Choice output
is limited to 64 records per tick and completes before prompt admission. Restore uses
the recorded choices without a new search. It does not need the source CCIR or request
paths. Incomplete transfers cannot publish state or prompts. Missing or invalid choices
at restore completion refuse affected work.

Each choice completes one pending query. Repeated choices refuse restore.

The journal preserves this interface's authoritative requests and choices.
Use [live checkpoints](25-memory-checkpoints.md) for a continuous memory mirror, or [complete runtime files](28-runtime-files.md) for packaged assets and history.

Per-agent audit JSONL contains the exact accepted input and a selection row.
The selection names lineage, conversation, principal, room, scope, full ordinal,
request ID, selection ID, store cut and selected object IDs with versions.
A complete matching device choice
is required before accepted input is written. A refused choice produces a selection
status row for a safely framed request, without an accepted input row. Malformed or
incomplete batches do not become accepted text. These files are derived audit data.

Typed audit turns follow the last device manifest, including prior tool turns.
Ordinary input does not advance typed audit turns. The request ordinal is separate from the model turn number.

See [memory maintenance](26-memory-maintenance.md) for retention policies, GPU reclamation and automatic file shrinking.

## Extended operations

The prefix table above covers the original live request family.
Operation 12 records direct admission, 13 carries maintenance and 14 carries semantic results.
[Appraisal](35-automatic-appraisal.md), [cold memory](36-cold-memory.md) and [task reviews](38-task-reviews.md) define their additional operation layouts.
Unknown operations or required schemas are refused.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
