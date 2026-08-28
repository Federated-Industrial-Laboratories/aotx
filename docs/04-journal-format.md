# The journal format

The journal is the disk copy of the records that the device writes. The drain reads one block
for each tick from the host ring and writes that block to a segment file. The segment files are
the only input of a restore. The other files of a journal are derived.

This document gives every layout of the journal. A reader can decode a journal with this
document and no other. The layouts come from `cuda/seam/wire.h`, which both sides include, and
from `disk/wire/diskwire.h`, which the disk side adds to it.

## The journal directory

The drain takes the journal directory from `--journal`. It makes one directory for each boot,
and the name of that directory is the boot identity in 16 hexadecimal digits.

| path | content |
| --- | --- |
| `<boot>/seg-<n>.seg` | the segment files of one boot; `<n>` is 6 digits |
| `<boot>/console.log` | the text of the console records of that boot |
| `bus/<date>-aotx.jsonl` | one message line for each message record and each derived event |
| `requests.jsonl` | one line for each tool request that the feeder executes |
| `manifest/<boot>.jsonl` | one line for each completed turn, with the digest chain |
| `bulk/<handle>` | one bulk payload; the name is the handle in 16 hexadecimal digits |
| `bulk/index.tsv` | one row for each payload |

The segment files hold every record. The derived files hold the part of the journal that a
person reads. The option `--derive` names the types that make derived lines, and it changes no
segment file.

## The segment file

A segment file is a run of frames. One frame carries one block. The frame header is 8 bytes.

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | the byte count of the block that follows |
| 4 | 4 | the CRC-32C of those bytes |

The checksum covers the block bytes only, and not the frame header. The seed of the checksum is
zero. The polynomial is the Castagnoli polynomial 0x1EDC6F41, which the table path holds in the
reflected form 0x82F63B78 (`disk/wire/crc32c.c`, `aotx_crc32c_table`). The drain opens the next
segment when a frame does not fit under the limit `AOTX_SEGMENT_LIMIT` of 64 * 1024 * 1024 bytes
(`disk/wire/diskwire.h`). The names hold a fixed count of digits, so the order of the names is
the order of the writes.

A reader finds the torn tail of a segment by four tests, in this order (`disk/wire/segment.c`,
`aotx_segment_get`). A read that gives no byte is the clean end of the file. A read that gives
fewer than 8 bytes is torn. A block length below 64, or above the buffer of the reader, is torn.
A block that the file cuts short, or a block whose checksum does not agree, is torn. A torn
frame ends the part of the journal that reads, and the walk stops at it (`disk/restore/scan.c`,
`aotx_journal_walk`).

## The block

A block starts with a 64-byte header and holds the records of one tick after it.

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic 0x4B4C4241, the letters ABLK |
| 4 | 2 | layout version, 1 |
| 6 | 2 | kind: 0 records, 1 pad, 2 bulk |
| 8 | 8 | block sequence, from 1; the publish field |
| 16 | 8 | boot identity |
| 24 | 8 | tick |
| 32 | 8 | the record sequence of the first record |
| 40 | 4 | record count |
| 44 | 4 | byte count of the block, this header included |
| 48 | 16 | reserved, zero |

A block never wraps the data area of the ring. A pad block fills the tail of that area, holds no
record, and reaches the end of the area. The drain does not write a pad block to a segment,
because it holds nothing (`disk/drain/drain.c`, `drain_pass`). A block of records satisfies the
rule `record_count * 256 + 64 == byte_len`. A bulk block holds a payload and no record, and its
byte count is a multiple of 8. The checks are in `disk/wire/record.c`, `aotx_block_valid`.

The last record of a complete tick is a tick-commit record. The block that ends with that record
is the block of a complete tick (`cuda/sched/step.cu`, `aotx_sched_commit`). A held tick writes
no commit record, so the block of a held tick ends with another type.

## The record

A record fills one slot of 256 bytes: a header of 64 bytes and a body of 192 bytes.

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic 0x58544F41, the letters AOTX |
| 4 | 2 | layout version, 1 |
| 6 | 2 | header byte count, 64 |
| 8 | 8 | boot identity of the run that wrote the record |
| 16 | 8 | tick |
| 24 | 8 | record sequence, from 1, contiguous; the publish field |
| 32 | 8 | device clock sample in nanoseconds |
| 40 | 4 | writer identity |
| 44 | 1 | class: 1 for class A, 2 for class B |
| 45 | 1 | type |
| 46 | 2 | flags |
| 48 | 4 | body byte count, at most 192 |
| 52 | 12 | reserved, zero |

A record of class A is authoritative and a restore replays it. A record of class B is derived
and a restore does not replay it. The flag 0x0001 marks a record that a restore applied again.
The flag 0x0002 marks a console record that continues the line of the record before it. The
flag 0x0004 marks a record that the device wrote while a replay ran.

The writer identity names the writer. The values below 1024 are system writers: 0 system, 1
feeder, 2 restore, 3 console. The value 1024 is the first agent, so an agent identity is 1024
plus the number of the agent.

## The record types

| number | name | class | writer | body |
| --- | --- | --- | --- | --- |
| 0 | pad | B | system | none |
| 1 | boot | A | system | boot |
| 2 | tick start | A | feeder | clock |
| 3 | tick commit | A | system | commit |
| 4 | input line | A | feeder | UTF-8 text |
| 5 | console | B | console | UTF-8 text |
| 6 | stall | B | system | stall |
| 7 | stats | B | system | stats |
| 8 | restore | B | restore | restore |
| 9 | note | B | system | UTF-8 text |
| 10 | key | A | feeder | key |
| 11 | command | B | console | UTF-8 text |
| 12 | bus | B | an agent | bus |
| 13 | bulk | B | system | bulk |
| 14 | token | A | system | token |
| 15 | sequence | B | system | sequence |
| 16 | tool request | B | an agent | tool request |
| 17 | tool reply | A | feeder, or an agent for a late verdict | tool reply |
| 18 | manifest | B | an agent | manifest |
| 19 | task | B | an agent | task |
| 20 | agent | B | an agent | agent |

A body of UTF-8 text carries the bytes alone, and the body byte count gives the count.

A tool reply of the status late comes from the device and not from the feeder. The tool step
writes it as a class A record, and the writer is the agent that made the request
(`cuda/tool/device_tools.cu`, `aotx_tool_step`). The body folds into the state hash, so a replay
applies the same verdict at the same place in the order. The feeder writes no late status,
because the feeder holds no tick.

## The record bodies

Each offset below is from the first byte of the body, which is byte 64 of the record. A field is
given as offset, colon, size, name.

| body | fields |
| --- | --- |
| boot | 0:8 boot identity; 8:8 boot identity of the run before, zero at a cold start; 16:8 wall clock at boot, in nanoseconds |
| clock | 0:8 wall clock when the feeder wrote the record |
| commit | 0:8 state hash, FNV-1a 64 over the applied class A bodies in order; 8:8 class A records applied since boot; 16:8 inbound slots consumed since boot; 24:8 records in the block that ends with this record |
| stall | 0:8 free bytes of the host ring when the tick was held; 8:8 ticks held since boot, with the top bit 0x8000000000000000 set when the flush dropped records |
| stats | 0:8 device time the tick took, in nanoseconds; 8:8 records written this tick; 16:8 inbound slots consumed this tick |
| key | 0:4 key code; 4:4 code point of a character event, or zero; 8:4 action, 1 press, 0 release, 2 repeat; 12:4 modifier bits |
| bus | 0:1 kind, 1 to 7; 1:1 provenance, 1 to 4 for a finding, else zero; 2:2 reserved; 4:4 writer sequence, from 1; 8:8 the record sequence this message refers to, or zero; 16:8 the record sequence this message corrects, or zero; 24:4 score, a 32-bit float from 0 to 1; 28:4 text byte count; 32:160 text |
| bulk | 0:8 handle, equal to the first sequence of the bulk block; 8:8 payload byte count; 16:4 kind of the payload, 1 for a text export; 20:4 reserved |
| token | 0:4 sequence slot; 4:4 token identity; 8:4 position in the sequence, from 0; 12:4 flags, 1 prompt, 2 sampled, 4 last; 16:8 seed of the random stream, zero for a prompt token; 24:8 draw count of the stream at this token; 32:4 model role; 36:4 reserved |
| sequence | 0:4 sequence slot; 4:4 event, 1 opened, 2 done, 3 stopped, 4 released; 8:4 prompt tokens; 12:4 sampled tokens; 16:8 ticks from the open to this event; 24:4 model role; 28:4 reserved |
| tool request | 0:4 agent; 4:4 turn; 8:4 tool, 1 memory recall, 2 memory write, 3 file read; 12:4 request identity; 16:8 the tick after which the request fails, or 0xffffffffffffffff while the request waits for the operator; 24:4 authorization, 0 none, 1 pending, 2 granted, 3 refused; 28:4 argument byte count; 32:160 argument |
| tool reply | 0:4 agent; 4:4 request identity; 8:4 status, 0 ok, 1 error, 2 refused, 3 late; 12:4 part, from 0; 16:4 count of parts; 20:4 byte count of this part; 24:168 bytes |
| manifest | 0:4 agent; 4:4 turn; 8:8 input hash, FNV-1a 64 over the prompt bytes; 16:8 output hash, FNV-1a 64 over the reply bytes; 24:4 output tokens; 28:4 finish, 0 stop, 1 tool, 2 limit; 32:4 the tool called, or zero; 36:4 the request made, or zero |
| task | 0:4 task; 4:4 agent; 8:4 state, 0 pending, 1 assigned, 2 running, 3 verifying, 4 done, 5 failed; 12:4 verification, 0 none, 1 sibling; 16:4 tries; 20:4 text byte count; 24:8 ticks since the task opened; 32:160 text |
| agent | 0:4 agent; 4:4 role; 8:4 parent, or the agent itself for a root; 12:4 state; 16:4 event, 1 spawned, 2 turn, 3 released; 20:4 turn; 24:8 ticks since the agent spawned |
| restore | 0:8 the boot identity of the journal that was replayed; 8:8 the last complete tick that was applied; 16:8 class A records replayed; 24:8 the state hash after the replay |

The top bit of the held count in a stall body is the mark `AOTX_STALL_OVERRUN`, which states
that the flush dropped records (`cuda/seam/seam.cuh`).

## The derived files

The console log takes the text of each console record. A control byte becomes a space. A record
that carries the fragment flag continues the line before it. A record without that flag ends
that line and starts a new one. The drain writes the end byte of the last line when it closes
the file, and the same text goes to its standard output.

The message file takes one line for each message record. It takes one line for each console
record, note record, task event, agent event and sequence end. `05-bus-schema.md` gives the
seven kinds and the fields of each one. The drain opens a new file when the day of its clock
changes. One note line:

```
{"v":1,"run":"aotx","agent":"system","seq":1,"ts":"2026-08-28T14:41:15.875+01:00","type":"note","body":{"text":"sequence done slot 0 role 2 prompt 353 sampled 256 ticks 195","tick":196,"boot":"0772fbccf1e3666e","lag_ms":72.560}}
```

The requests file takes one line for each tool request that the feeder can execute. A request
that needs no authorization makes its line at once. A request that waits for the operator makes
its line when the record that grants it comes. A request that the operator refuses makes no
line. A record with the flag 0x0004 makes no line, because the device wrote it while a replay
ran and the journal already answers its request.

The tool, the agent, the turn and the argument of a line come from the request. The deadline
comes from the record that grants it (`disk/drain/derive_manifest.c`, `aotx_derive_request`).
The field `tick` is the tick that the deadline counts from, for every line. A tool without
authorization counts from the request, and a tool with one counts from the grant.

A request that waits for the operator holds no deadline. The deadline field of its record
carries the mark `AOTX_TOOL_NO_DEADLINE` of 0xffffffffffffffff, which is above every tick that a
run reaches (`cuda/tool/tool.cuh`). The deadline of `tool.deadline_ticks` (`cuda/settings/keys.h`,
500 ticks unless a setting changes it) starts at the grant (`cuda/agent/table.cu`,
`aotx_agent_authorize`).

```
{"request":2,"agent":1,"turn":1,"tool":"fs_read","arg":"one.txt","deadline":525,"auth":"granted","tick":25}
```

The chain file takes one line for each completed turn of an agent. The two hashes are 16
hexadecimal digits. The field `finish` and the field `tool` are words.

```
{"agent":1,"turn":1,"input_hash":"3c6c28436fe74706","output_hash":"c0223947f65f82b2","tokens":21,"finish":"tool","tool":"fs_read","request":2,"prev":"0000000000000000000000000000000000000000000000000000000000000000"}
```

The bulk files take the payloads that no record can hold. The name of a payload file is the
handle in 16 hexadecimal digits. The index holds one header row and one row for each payload,
with tab characters between the fields.

```
handle	tick	length	crc
```

A row gives the handle in 16 hexadecimal digits, the tick, the payload byte count and the
CRC-32C in 8.

## The chain rule

The field `prev` of a chain line is the SHA-256 digest of the bytes of the line before it
(`disk/drain/derive_manifest.c`, `aotx_derive_turn`). The end byte of that line is in the
digest. The first line of a file carries 64 zeros. A line that is taken out, and a byte that
changes, therefore break every line after them.

The chain is not in the `--derive` list. A turn that makes no line makes a gap, and a chain with
a gap proves nothing. A drain that opens a chain file which already holds lines takes the digest
of the last one. A file that ends without an end byte gets one first, so the line that a crash
cut stays a line of its own.

## Reading a journal

`aotx_journal` prints a journal as text. The directory is a boot directory when it holds
segments. If it holds none, it is a journal directory, `--boot` names the boot in it, and with
no `--boot` the newest complete boot is read.

```
aotx_journal tokens|manifest|requests <dir> [--boot <id>]
```

`tokens` prints one line for each token record. The line gives the slot, the position, the
token, the flags, the seed, the draw and the role. It gives the tick, the record sequence, a
sampled field and a replayed field after them. The first four fields are the token itself, so a
comparison of two runs cuts each line after them.

```
slot=0 position=0 token=151644 flags=0x0001 seed=64673fed7e48d689 draw=0 role=2 tick=1 seq=5 sampled=0 replayed=1
```

`manifest` prints one line for each turn and recomputes every digest. With no `--boot` it reads
every file that ends in `.jsonl` under `<dir>/manifest`. A report names the count of turns and
the state of each chain. A broken chain reports the line, the digest that the line carries and
the digest that the line before it gives.

```
line=1 agent=1 turn=1 input=3c6c28436fe74706 output=c0223947f65f82b2 tokens=21 finish=tool tool=fs_read request=2
```

`requests` prints the requests file as fixed fields.

```
request=2 agent=1 turn=1 tool=fs_read auth=granted deadline=525 tick=25 arg=one.txt
```

The three disk-side programs share four exit statuses. Status 0 is a clean run. Status 2 states
that a ring preamble names another layout version. Status 3 states that the journal, the boot or
the file is not there.

Status 1 is a fault. A fault is bad arguments, a walk that did not finish, or a chain that
breaks. A line of the requests file that does not read is also a fault.

## What a restore reads

A restore reads the segment files and nothing else, because the derived files are outputs.

1. It reads each directory of the journal whose name is 16 hexadecimal digits.
2. It keeps each boot that holds at least one tick-commit record, and it stops the read of a
   boot at the first torn frame.
3. It ranks the boots by the wall clock of the boot record. A boot that holds no boot record
   ranks by the change time of its directory. It takes the highest.
4. It walks that boot again, up to and including the block of the last tick-commit record.
5. It sends every class A record of those blocks to the inbound ring, in record order. Each
   record goes out with the replayed flag set and the writer set to restore. It leaves out the
   boot records and the tick-commit records, because the device makes both again.
6. It publishes one restore record, and then waits until the device consumes every slot.

The apply of the device takes the records of one journal tick in one tick of the restored run
(`cuda/seam/inbound.cu`, `aotx_seam_replay_take`). A replay therefore takes as many ticks as the
run that wrote the journal. An input of the operator then reaches the device at the place in the
flow of the agents that it had before.

The clock of the replay moves only when the apply saw a record of a later tick. A journal tick
with more records than one apply takes spills into the ticks after it. It never merges with the
tick that follows it. The restore report of the run states a `paced` count: the ticks of the replay that took no record of the journal
(`cuda/boot/children_host.cu`, `aotx_boot_replay`).

A replay whose ring makes no progress for a million turns of that loop ends the run with a line
that names it. A run therefore never goes on from a part of the journal as if it were the whole.

A torn tail is reported and the tick before it is restored. The last complete tick is the tick
of the last block that ends with a tick-commit record.
