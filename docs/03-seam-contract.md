# The seam contract

The seam is the boundary between device memory and pinned host memory. Rings cross it and nothing
else crosses it. A program on the far side can be written from this document and
`cuda/seam/wire.h` alone. Every number is little-endian, and each structure takes the standard
alignment of its fields. `cuda/seam/wire.h`, `aotx_wire_check_record` checks each total at build
time.

## The four rings

| ring | memory | producer | consumer | size |
| --- | --- | --- | --- | --- |
| device record ring | device | every kernel | the flush node | 65,536 slots at the most |
| host ring | pinned host, one memfd | the flush node | the drain | 64 MB data area |
| bulk ring | pinned host, one memfd | the bulk flush node | the drain | 256 MB data area |
| inbound ring | pinned host, one memfd | the feeder or the restore program | the apply node | 4,096 slots |

The sizes come from `AOTX_DEVICE_RING_SLOTS` and `AOTX_HOST_RING_DATA_BYTES` in
`cuda/seam/seam.cuh`, and `AOTX_MEM_RING_BYTES` in `cuda/mem/mem.cuh`. `aotx_seam_bind` gives the
device ring one slot for each 256 bytes of its region, and cuts the count to 65,536
(`cuda/seam/seam_host.cu`). That ring never crosses the seam. The host ring and the bulk ring
carry blocks; the inbound ring carries records. A data area is a power of two, so a position is a
mask.

## The fourth crossing

The display mirror is the fourth crossing. It is not a ring and carries no authoritative
state. Host glue publishes a fixed cell snapshot to a memfd. The feeder reopens that descriptor
read-only and sends it to `aotx_tui` over `<journal>/aotx.sock` with `SCM_RIGHTS`. Both ends bind
or connect through an open descriptor for the journal directory. Only the socket file name counts
against the Unix socket address bound.

The attach socket is a byte stream. A frame starts with one kind byte:

| kind | direction | bytes after the kind |
| --- | --- | --- |
| `K` | terminal to feeder | one 16-byte `aotx_key_body` |
| `L` | terminal to feeder | one 4-byte little-endian length, then that many line bytes |
| `R` | feeder to terminal | one 4-byte little-endian length, then that many reason bytes |

The mirror descriptor travels with one `M` payload byte. A terminal sends whole frames. The
feeder may receive a partial frame or several frames in one read and keeps the unused bytes for
the next read. A key becomes one KEY record. A line becomes one INPUT_LINE record. The terminal
never writes an inbound ring and the feeder remains its one producer.

## The record

One record fills one slot of 256 bytes: a header of 64 bytes and a body of 192 bytes. The header
is `aotx_record_header`, in `cuda/seam/wire.h`.

| offset | bytes | field | value |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | `0x58544F41` |
| 4 | 2 | `layout` | 1 |
| 6 | 2 | `header_bytes` | 64 |
| 8 | 8 | `boot_id` | the run that wrote the record |
| 16 | 8 | `tick` | device time |
| 24 | 8 | `seq` | the publish field; from 1, contiguous |
| 32 | 8 | `globaltimer` | device clock sample in nanoseconds |
| 40 | 4 | `writer` | the writer identity |
| 44 | 1 | `cls` | 1 for class A, 2 for class B |
| 45 | 1 | `type` | 0 to 25 |
| 46 | 2 | `flags` | `0x0001` replayed, `0x0002` fragment |
| 48 | 4 | `body_len` | bytes of the body that carry data, 192 at the most |
| 52 | 8 | `source_seq` | source record sequence in a replay ring, else zero |
| 60 | 4 | `reserved` | zero |
| 64 | 192 | the body | the layout of the type |

The `seq` field is the publish field of a slot, and zero means unpublished or under rewrite. The
flag `0x0001` states that a restore applied the record again, and `0x0002` states that the record
continues the line before it (`cuda/seam/wire.h`, `AOTX_FLAG_REPLAYED`).
The restore program puts the source record sequence in `source_seq` on the inbound ring. A
device record puts zero there. This field keeps transcript provenance stable after a replay.

A writer identity below 1,024 is a system writer (`cuda/seam/wire.h`, `AOTX_WRITER_AGENT_BASE`).
Identity 0 is the system, 1 the feeder, 2 the restore program and 3 the console. Agent `i` writes
as `1024 + i`.

## The block

A block holds the records of one tick. The block header is `aotx_block_header` in
`cuda/seam/wire.h`, and it is 64 bytes.

| offset | bytes | field | value |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | `0x4B4C4241` |
| 4 | 2 | `layout` | 1 |
| 6 | 2 | `kind` | 0 records, 1 pad, 2 payload |
| 8 | 8 | `block_seq` | the publish field; from 1, no gaps |
| 16 | 8 | `boot_id` | the run that wrote the block |
| 24 | 8 | `tick` | the tick the block carries |
| 32 | 8 | `first_seq` | the sequence of the first record, or a payload handle |
| 40 | 4 | `record_count` | records that follow the header |
| 44 | 4 | `byte_len` | bytes of the block, this header included |
| 48 | 16 | `reserved` | zero |

A block of records satisfies `byte_len = 64 + 256 * record_count`. A block never wraps. A pad
block fills the tail when the next block does not fit there. The block after it starts at offset
zero. A pad block and a payload block hold no record.

## The ring preambles

The host ring and the bulk ring carry `aotx_host_ring_preamble` of `cuda/seam/wire.h`, which is
256 bytes. Each shared field has a cache line of its own.

| offset | bytes | field | writer |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | the producer, once at creation |
| 4 | 2 | `layout` | the producer, once at creation |
| 6 | 2 | `closed` | the producer, at the end of the run |
| 8 | 8 | `boot_id` | the producer, once at creation |
| 16 | 8 | `data_bytes` | the producer, once at creation |
| 24 | 8 | `preamble_bytes` | the producer, once at creation |
| 64 | 8 | `head` | the producer; bytes written, monotonic, not masked |
| 128 | 8 | `cursor` | the consumer; bytes drained to disk, monotonic |
| 192 | 8 | `last_block_seq` | the producer; the last published block sequence |

The inbound ring carries `aotx_inbound_preamble` of the same header, which is 192 bytes. It holds
`magic`, `layout` and `closed` at the same three offsets, `slot_count` at offset 8 and
`preamble_bytes` at offset 16. The feeder writes `head` at offset 64 and the device writes
`consumed` at offset 128, each on a line of its own. A data area begins at `preamble_bytes` from
the first mapped byte, and the padding bytes hold zero.

## The one consumer field

`cursor` is the only field a consumer writes in a host ring. The drain publishes it with a release
store, after the bytes reach the disk (`disk/wire/ring.c`, `aotx_host_ring_advance`). An earlier
store would let the producer count a lost block as safe. The producer reads `cursor` with an
acquire load and never spins on it. It reads that cursor at tick start for the hold decision
(`cuda/sched/step.cu`, `aotx_sched_tick_start`), and again in the flush for the pad decision
(`cuda/seam/flush.cu`, `aotx_seam_flush`).

The bulk path reads the cursor of its own ring once at tick start (`cuda/seam/seam.cuh`,
`aotx_bulk_tick_start`). `consumed` is its mirror in the inbound ring, and the device is its one
writer. The apply publishes it with a release store when its last block ends
(`cuda/seam/inbound.cu`, `aotx_seam_apply_inbound`). The feeder never writes over a slot the
device has not taken.

## Publication and acquisition

The producer of a block writes in this order (`cuda/seam/flush.cu`, `aotx_seam_flush` and
`aotx_flush_publish`).

1. Store zero into `block_seq` of the place the block will take.
2. Fence with system scope.
3. Write the block header and the records.
4. Fence with system scope.
5. Release-store the true `block_seq`, which is the last one plus one.
6. Release-store `head`, which grows by the byte length of the block, then `last_block_seq`.

The consumer reads in the opposite order (`disk/wire/ring.c`, `aotx_host_ring_take`).

1. Acquire-load `head`. The ring is empty when `cursor` is not below `head`.
2. Take the block at `cursor & (data_bytes - 1)`.
3. Acquire-load `block_seq`. Zero means a block under write, and the ring reads as empty.
4. Read `byte_len`, check it, and copy the whole block into a private buffer.
5. Acquire-load `block_seq` again. A value that differs means a torn read; pause and read again.
6. Check the copy, then move `cursor` by `byte_len`.

The consumer moves its cursor over a pad block as over any other. Before it accepts a block it
checks these things (`disk/wire/ring.c`, `aotx_host_ring_take`, and `disk/wire/record.c`,
`aotx_block_valid`). The cursor stands on an 8-byte boundary, and a block header fits before the
end of the data area. `byte_len` is 64 at the least, is inside the buffer, does not run past the
data area, and matches the copy.

The magic and the layout version are the ones this build reads, `block_seq` is not zero, and `head
- cursor` is not below `byte_len`. A pad block reaches the end of the data area, and a payload
  block length is a count of 8 bytes. Every record header holds the record magic, the layout
  version, a header size of 64 and a body length of 192 at the most.

## Sequences and loss

Block sequences start at 1 and have no gaps, so a consumer knows which sequence comes next. A
block left behind by an earlier pass holds a smaller sequence, and the double-load rule refuses
it. The drain reports a gap and goes on, because the blocks that follow are still whole
(`disk/drain/drain.c`, `drain_pass`).

Record sequences also start at 1 and are contiguous. A sequence that carries nothing takes a pad
record, so one tick leaves no hole in the sequence space. The flush checks that every record
carries the sequence its position gives it. It cuts the block in front of the first record that
fails (`cuda/seam/flush.cu`, `aotx_seam_flush`). Loss therefore reaches the consumer as a gap in
the record sequence, and never as silence. The count of dropped runs goes in the next stall
record, in the high bit of `held_count` (`cuda/seam/seam.cuh`, `AOTX_STALL_OVERRUN`).

## Making and attaching a ring

The device process makes each ring (`cuda/seam/seam_host.cu`, `aotx_seam_make`). It calls
`memfd_create` with the close-on-exec flag, named `aotx-host-ring`, `aotx-bulk-ring` or
`aotx-inbound-ring`. It grows the file to the preamble size plus the data area, rounded up to
4,096 bytes. It maps the file with `PROT_READ | PROT_WRITE` and `MAP_SHARED`, and sets every byte
to zero. It then registers the mapping with `cudaHostRegister` and `cudaHostRegisterMapped`.

`aotx_seam_open` writes the preamble once, so a late reader still knows the layout, and
`aotx_seam_bind` takes the device address from `cudaHostGetDevicePointer`
(`cuda/seam/seam_host.cu`).

A disk-side program receives the descriptor of the ring it must map and no other. The spawn clears
the close-on-exec flag on the named descriptors and sets it on every other one
(`cuda/seam/seam_host.cu`, `aotx_seam_only`). It maps the whole descriptor with the size from
`fstat`, then attaches. An attach fails when the magic, the layout version or a preamble size is
not the one that build reads (`disk/wire/diskwire.h`, `aotx_host_ring_attach`). A consumer takes
one more pass after `closed` goes to 1, and stops.

## The inbound ring

The producer writes one slot and publishes it (`disk/wire/ring.c`, `aotx_inbound_put`). An
input line can use more than one slot. The feeder and the restore program write all parts,
and then advance the head one time (`disk/feed/line.c`, `aotx_line_publish_records`). A reader
therefore sees the complete line or no part of it.

1. Take `head` with an acquire load. The slot is `head & (slot_count - 1)`.
2. Release-store zero into `seq` of that slot.
3. Write the header fields and the body, and zero the rest of the body.
4. Release-store `head + 1` into `seq`.
5. Release-store `head + 1` into `head` of the preamble.

The producer waits for a free slot before step 1. A slot is free when `head - consumed` is below
`slot_count` (`disk/wire/ring.c`, `aotx_inbound_wait`). The producer stamps `boot_id` zero,
because the inbound preamble carries no boot identity (`disk/feed/feed.c`, `AOTX_WRITER_FEEDER`).
The device stamps its own boot identity when it writes the record to the journal. It takes ten
types and refuses every other one (`cuda/seam/inbound.cu`, `aotx_apply_takes`). A refused slot is
counted, and its two sequences take pad records.

| type | number | class | least `body_len` | body |
| --- | --- | --- | --- | --- |
| `TICK_START` | 2 | A | 0; the wall clock is read at 8 | `aotx_clock_body`, 8 bytes: 0 `wall_ns` |
| `INPUT_LINE` | 4 | A | 0 | UTF-8 bytes; `body_len` gives the count |
| `RESTORE` | 8 | B | 32 | `aotx_restore_body`, 32 bytes: 0 `restored_boot_id`, 8 `last_tick`, 16 `replayed_count`, 24 `state_hash` |
| `KEY` | 10 | A | 16 | `aotx_key_body`, 16 bytes: 0 `key`, 4 `codepoint`, 8 `action`, 12 `mods` |
| `TOKEN` | 14 | A | 192 | `aotx_token_body`, 192 bytes: 0 `slot`, 4 `token`, 8 `position`, 12 `flags`, 16 `seed`, 24 `draw`, 32 `role`, 36 `text_len`, 40:152 reply bytes |
| `TOOL_REPLY` | 17 | A | 192 | `aotx_tool_reply_body`, 192 bytes: 0 `agent`, 4 `request`, 8 `status`, 12 `part`, 16 `parts`, 20 `len`, 24 the bytes |
| `SETTING` | 21 | A | 80 | `aotx_setting_body`, 80 bytes: 0 `value`, 8 `scale`, 12 `key_len`, 16:64 key bytes |
| `IMPORT` | 23 | A | 184 | an import head of 184 bytes or a part of 192 bytes |
| `REMOVE` | 24 | A | 64 | `aotx_remove_body`, 64 name bytes |
| `SELECTION` | 25 | A | 192 | `aotx_selection_body`: agent, turn, count, pages, summary sequence, at most 20 recalled sequences and the source sequence |

The magic and the layout version must match, and `body_len` must not be above 192. A key body
carries the codes of the window library. The action is 1 for a press, 0 for a release and 2 for a
repeat, and a code point event holds `key` zero. A token body carries the seed, the draw and the
reply bytes of a sampled token. A restore applies the token and samples nothing again.

selection is refused when its count is above 20 or its pages are above the profile limit. It
is also refused when a sequence is not before its source sequence.

## The bulk ring

The bulk ring carries the same preamble and the same block header, and a payload never enters a
record. A bulk block has `kind` 2 and a `record_count` of zero, and the handle stands in
`first_seq`. The `byte_len` is 64 plus the payload rounded up to 8 bytes (`cuda/seam/bulk.cu`,
`aotx_seam_bulk_flush`). A `BULK` record in the journal carries the same handle and the exact
length.

A bulk block takes a place only when the space after it is zero, or holds a block header of 64
bytes (`cuda/seam/bulk.cu`, `aotx_bulk_fits`). A pad block goes in first when it does not. A
payload block rounds to 8 bytes and a record block rounds to 256. The tail of this ring therefore
reaches a state the host ring cannot. One tick stages 256 payloads at the most, in a staging
region of 8 MB (`cuda/seam/seam.cuh`, `AOTX_BULK_STAGE_MAX`). A payload that finds no room is
refused and counted.
