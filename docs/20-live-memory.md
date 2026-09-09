# Live memory

The device can load one typed CCIR store and bind fresh conversation slots to it.
Each binding names a lineage, principal, room, conversation ID, scope and fixed page
cap. A binding cannot replace an existing binding or adopt a slot with prior turns.
Unbound conversations keep the base input and transcript policy. Bound conversations
accept typed requests and use selected memory plus the current input for their prompts.

The local operator can use these commands from standard input or an attached input file:

```
memory load PATH
memory apply PATH
memory bind PATH
memory query PATH
memory text PATH
```

`load` reads the verified checkpoint and tail from a CCIR file. `apply` reads a canonical
typed tail. `bind` and `query` read raw batch files with the layouts below. A path is the
rest of the line, with spaces kept as path bytes. There is no shell expansion or quote
processing.

`text` reads the same batch shape with zeroed vector fields and a 192-byte input limit.
The GPU prepares those vectors with the loaded embedding model.
[Text requests](21-text-memory.md) defines its required model roles and exact byte layout.

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
| 4 | 4 | Operation: load 1, update 2, bind 3, query 4, choice 5, text 6, text choice 7 |
| 8 | 16 | Nonzero transfer ID |
| 24 | 4 | Total transfer bytes |
| 28 | 4 | Data offset in the transfer |
| 32 | 1 to 160 | Exact data bytes |

Parts are in order. Each part has 160 data bytes except the final part. The feeder
publishes at most 32 records per group and waits for ring space between groups. Each
file transfer gets a new random ID. A load has two 64-bit lengths at offsets 0 and 8,
then the exact checkpoint bytes and exact tail bytes.

An update is one typed tail.
The maximum load is 2,228,496 bytes, including the 16-byte length prefix.

Bind, query and choice have a 64-byte header. Their magic bytes are `AOTXBND1`,
`AOTXLIV1` and `AOTXCHO1`. Count and schema are 32-bit fields at 8 and 12; lineage is
at 16. The 64-bit store sequence is at 32, and the 32-bit row size is at 40. Bind and
query bytes 44 through 63 are zero. A choice has a 32-bit status at 44 and zero bytes
48 through 63.

Status 0 means success; status 1 through 11 is a refusal with count 0
and no rows. A choice has the same transfer ID as its query. Only the device emits it.

| Row | Bytes | Fields |
| --- | --- | --- |
| Bind | 64 | Slot at 0, scope at 4, principal at 8, room at 24, conversation at 40, page cap at 56; zero 60 through 63 |
| Query | 8,256 | Slot at 0, zero 4 through 15, conversation at 16, 64-bit ordinal at 32, zero 40 through 63, prepared query at 64 |
| Choice | 592 | Exact 64-byte query prefix, then the 528-byte ordered selection buffer |

Slots, scope and page caps are 32-bit values. The prepared query and selection layout
are defined in [prepared memory](19-prepared-memory.md). Ordinals start at 1 for each
binding. The largest bind is 4,160 bytes; the largest query is 528,448 bytes; the
largest choice is 37,952 bytes. A choice keeps zero bytes in unused selection slots.

## State, prompts and recovery

The resident typed store keeps the existing 256-version and 1 MiB payload limits.
Explicit updates require idle cognitive conversations. Each binding keeps bounded
current request and choice state. Later queries replace this current state; the
journal keeps prior requests. The 64-pair saved-query limit of the offline recall
file does not limit the number of live turns.

Prepared queries supply their own vectors;
text requests use the GPU embedding service. Memory writing and deletion require
explicit typed updates. This interface does not provide disk offload.

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

The journal is the recovery file for this
interface; a continuous CCIR mirror and portable active-conversation export are not
part of these commands.

Per-agent audit JSONL contains the exact accepted input and a selection row.
The selection names lineage, conversation, principal, room, scope, full ordinal,
request ID, selection ID, store cut and selected object IDs with versions.
A complete matching device choice
is required before accepted input is written. A refused choice produces a selection
status row for a safely framed request, without an accepted input row. Malformed or
incomplete batches do not become accepted text. These files are derived audit data.

Typed audit turns follow the last device manifest, including prior tool turns.
Ordinary input does not advance typed audit turns. The request ordinal is separate from the model turn number.
