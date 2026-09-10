# Live memory checkpoints

A runtime can maintain a `.aotxccir` mirror of its GPU memory state.
Start the runtime with `--memory-mirror PATH`. The normal disk drain writes the file.
The GPU remains authoritative. This option does not move active memory to disk.

Without the option, the checkpoint transport is not allocated and no mirror is written.
Unbound conversations keep the base conversation policy.

```
aotx_boot --models STORE --roles language --journal JOURNAL --memory-mirror state.aotxccir
```

Load and bind memory with the [live memory commands](20-live-memory.md).
The mirror is created after the first complete memory operation.
Its parent directory must exist. File creation refuses an existing destination with different state.
An existing mirror must match the lineage and operation history of the GPU checkpoint.
Use a new pathname for a new memory instance.

## Durability and pressure

A checkpoint contains the complete typed store and all bound conversations at one completed-work boundary.
All bound conversations must be idle. Partial input transfers, prompts, tools and pending quality work prevent capture.
A continuously busy binding can delay checkpoints for the whole store.
The GPU encodes one immutable image, then copies at most 256 KiB to the transport per tick.
Later operations can proceed while this copy completes if another transport slot is available.

The drain publishes complete CCIR generations in order.
It releases a transport slot only after the file update is synchronized.
An acknowledgement from another boot cannot release a slot.
The `memory` command reports these counters:

```
memory mirror: committed 6 durable 6 generation 5 pending 0 error 0
```

`committed` counts accepted memory operations. It does not count typed objects.
`durable` names the accepted operation boundary in the synchronized mirror.
`generation` names the selected CCIR generation. Several operations can share one checkpoint.

`pending` counts published or partly copied transport slots.
It does not count dirty state that is waiting for an idle boundary.
`error` is zero or the reported persistence error.
An equal committed and durable count with zero pending and error means that the mirror has caught up.

Check these counters before stopping a runtime when the latest completed memory work must survive.
An abrupt stop can lose work after the last durable boundary.

Disk errors preserve the unacknowledged snapshot and retry after one second.
A full transport or disk error refuses new memory operations with capacity status.
Already admitted work can finish. No snapshot slot is overwritten before acknowledgement.

Direct load, update, bind, resume and maintenance inputs carry journal flag `0x0008`.
The device writes a class A memory admission record before the operation can publish.
Replay waits for that record and uses its recorded pressure result.
A missing or mismatched admission record makes recovery fail.
Older unmarked direct inputs retain their original replay path.

The admission uses memory operation 12 with a single 64-byte payload.
Its outer transfer ID matches the original input. The payload contains `AOTXADM1` at 0.
Schema, original operation, input length and pressure (0 or 1) occupy 32-bit fields at 8, 12, 16 and 20.
Original input sequence, prior accepted count and prior store sequence occupy 64-bit fields at 24, 32 and 40.
The prior store lineage occupies bytes 48 through 63.

The mirror keeps a writer lease between updates. Other writers and compaction refuse that source while it is open.
Readers can inspect complete generations between updates. A conflicting reader or writer can return file-in-use status.

## Resume from the file

Start a fresh runtime with the compatible external model, module and settings configuration.
Create the same agent slots with the required roles, then send:

```
memory resume state.aotxccir
```

Resume requires fresh idle slots. It refuses existing bindings, prior turns and pending tasks on those slots.
It validates the entire store and binding batch on the GPU before publishing either.

The original input files and prior runtime journal are not required.
A new journal records the resume transfer and later activity.

The GPU emits class B record 34 only after successful import.
This audit record restores the disk transcript turn counters.
Its schema/count fields are at 0/4, transfer ID at 8 and operation revision at 24.
Up to 20 slot/turn pairs follow at 32; each pair has two little-endian 32-bit values.


Use `--memory-mirror` with the same file to continue its durable generations.
An unchanged resumed snapshot does not create another generation.
Capture time and executed-search counts are diagnostic observations; a change to these fields alone does not create a generation.

The file preserves principal, room, conversation ID, scope, page cap, input ordinal and automatic retention mode.
It also preserves exact working focus, the last prepared query, selection and rendered context.
Completed turn and prompt-open counts preserve the next conversation turn number.
Resume performs no vector search. Fresh input uses the restored memory and normal recall rules.
A context invalidated by a later store update remains stale and cannot authorize a continuation.

This memory-state profile requires external runtime components.
It does not package model weights, module code, affect state, settings or conversation audit logs.
It does not resume an interrupted response, tool action or model cache.
Use normal journal restore when the required recovery includes that journal's runtime history.
`memory load` and offline typed-state commands refuse this profile because they cannot restore its required bindings.

## File and transport layout

The CCIR prologue, roots and directory rows keep [format 1](17-ccir.md).
Required sections are manifest type 1/schema 2, typed checkpoint type 2/schema 1 or 2 and live bindings type 4/schema 1.
The 96-byte manifest keeps the data-state fields, with schema 2 at offset 8.
Bytes 40 through 55 are zero; there is no tail section.
Bytes 56 through 71 name the required live section ID. Bytes 72 through 95 are zero.

Unknown optional sections remain opaque and survive mirror updates and file compaction.
An optional type 4 section is not the required live section.

The live section has a 128-byte header followed by 17,416-byte binding rows.
Integers are little-endian. IDs are 16 bytes. The typed checkpoint remains a separate extent.
The resume transfer concatenates the live section and the typed checkpoint without changing their bytes.

| Header offset | Bytes | Value |
| --- | --- | --- |
| 0 | 8 | `AOTXLCP1` |
| 8, 12, 16 | 4 each | Schema 1, row bytes, binding count |
| 20 | 4 | Zero |
| 24 | 8 | Typed checkpoint byte count |
| 32 | 16 | Lineage |
| 48, 56 | 8 each | Object sequence and object tick |
| 64, 72 | 8 each | Accepted operation count and capture runtime tick |
| 80 | 48 | Zero |

| Row offset | Bytes | Value |
| --- | --- | --- |
| 0, 4, 8, 12 | 4 each | Slot, page cap, scope, memory context byte count |
| 16 | 8 | Input ordinal |
| 24, 40, 56 | 16 each | Principal, room, conversation |
| 72, 76 | 4 each | Focus count, automatic retention |
| 80, 84 | 4 each | Completed turn count, prompt-open count |
| 88 | 40 | Zero |
| 128 | 8,192 | Prepared query |
| 8,320 | 8,904 | Encoded result |
| 17,224 | 192 | Eight focus ID/version pairs |

The result stores status, count, context bytes and searches as 32-bit values at 0, 4, 8 and 12.
Its 64-bit store cut is at 16. Request and selection IDs are at 24 and 40.
Sixteen 32-bit object indices start at 56; sixteen 32-bit reasons start at 120.

The 528-byte selection starts at 184. The 8,192-byte context starts at 712.
No process pointer or device address is stored in the file.

The transport is a separate mapped file descriptor, with layout constants in `cuda/cognitive/checkpoint.h`.
Its boot ID and monotonically increasing slot serial identify each publication.
This descriptor is local transport, not a portable file format.

## Capacity and checks

`AOTX_MEMORY_OBJECTS` and `AOTX_MEMORY_BYTES` size the complete GPU store and its checkpoint capacity.
`AOTX_MEMORY_SNAPSHOTS` sets the transport slot count; the default is two.
Each slot fits the configured complete typed image plus every profile binding.
There is no smaller checkpoint object limit.

The default 64-slot profile uses a 19,989,248-byte maximum snapshot and a 39,981,056-byte mapped transport.
The GPU also reserves one complete checkpoint image and one staged binding table.

File writes use actual occupied bytes. Unchanged sections reuse their existing extents.

The file grows as new generations append. Optional [memory maintenance](26-memory-maintenance.md) releases GPU state and shrinks lifecycle mirrors.
Stop the writer before using `aotx_ccir compact` to produce a smaller file with the same selected state.
The compiled CCIR file cap is controlled by `AOTX_CCIR_FILE_BYTES`; zero removes that application cap.
Filesystem limits and available storage still apply. A limit failure retains pending state and reports pressure.

Run `ctest -R '^checkpoint$'` for device and disk failure checks at N=1 and N=64.
Run `ctest -R '^checkpoint_replay$'` for direct-operation pressure decisions and recovery refusal controls.
For bounded sanitizer runs, `aotx_checkpoint_test --state-only` omits full-payload capacity checks.
It still checks coherent copies, state validation, recovery and persistence failures at both batch sizes.
The default test always includes the full configured payload.
Run the real model check with an existing store and a new output directory:

```
python3 tests/checkpoint_boot_test.py BUILD SOURCE STORE OUTPUT vector-64
```

Use `vector-1` or `text-1` for the corresponding singleton checks.
The check deletes its original memory inputs and journal before resuming from the mirror.
It verifies exact private identities, corrected recall, focus and new automatic retention through actual model replies.
