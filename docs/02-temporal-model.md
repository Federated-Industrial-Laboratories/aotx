# The temporal model

The device is ahead and the disk is behind. This rule holds every module boundary in place.
This document gives the rules, the two record classes, what a restore does, and what a crash
loses.

## The rules

1. Device memory holds the authoritative state while a run goes on. The disk is a copy that
   lags. The disk is authoritative at a cold start only.
2. Device time is the tick. The tick start kernel adds one to the tick counter, and every
   other kernel reads it (`cuda/sched/step.cu:176`). The counter starts at zero, so the first
   tick is tick 1 (`cuda/time/tick.cu:25`).
3. Every record carries a header of the boot identity, the tick, the record sequence and a
   device clock sample (`cuda/seam/wire.h:59-73`). Order comes from the tick and the sequence.
   The clock sample is in nanoseconds from an origin that a driver load resets, and it
   measures lag only.
4. The feeder writes a tick start record that carries the wall clock, ten times a second
   (`disk/feed/feed.c:17`). Device time therefore pairs with wall-clock time on disk. The
   device reads no host clock.
5. At the end of each tick the flush node copies the records of that tick into the host ring
   as one block (`cuda/seam/flush.cu:48`). One tick makes one block.
6. The drain writes blocks to journal segments. It synchronizes the segments before it moves
   its cursor, so a crash never loses a block that the producer counts as safe
   (`disk/drain/drain.c:139-147`).
7. No kernel waits on the host. Backpressure is decided at tick start from one read of the
   drain cursor. Nothing spins on a host field.
8. A restore replays the class A records of the newest complete journal, up to the last
   complete tick. A tick that is not complete on disk is left out.
9. The model files are read once at the start of a run, before the first tick
   (`cuda/boot/boot_host.cu:151`). Nothing reads them again while the run goes on.

## The two record classes

A class A record is authoritative and a restore replays it. A class B record is derived from
class A records and a restore does not replay it. The class stands in the header of every
record, and the wire header gives the class of each type.

| number | type | class | body |
| --- | --- | --- | --- |
| 0 | `PAD` | B | none |
| 1 | `BOOT` | A | `aotx_boot_body` |
| 2 | `TICK_START` | A | `aotx_clock_body` |
| 3 | `TICK_COMMIT` | A | `aotx_commit_body` |
| 4 | `INPUT_LINE` | A | UTF-8 bytes |
| 5 | `CONSOLE` | B | UTF-8 bytes |
| 6 | `STALL` | B | `aotx_stall_body` |
| 7 | `STATS` | B | `aotx_stats_body` |
| 8 | `RESTORE` | B | `aotx_restore_body` |
| 9 | `NOTE` | B | UTF-8 bytes |
| 10 | `KEY` | A | `aotx_key_body` |
| 11 | `COMMAND` | B | UTF-8 bytes |
| 12 | `BUS` | B | `aotx_bus_body` |
| 13 | `BULK` | B | `aotx_bulk_body` |
| 14 | `TOKEN` | A | `aotx_token_body` |
| 15 | `SEQUENCE` | B | `aotx_sequence_body` |
| 16 | `TOOL_REQUEST` | B | `aotx_tool_request_body` |
| 17 | `TOOL_REPLY` | A | `aotx_tool_reply_body` |
| 18 | `MANIFEST` | B | `aotx_manifest_body` |
| 19 | `TASK` | B | `aotx_task_body` |
| 20 | `AGENT` | B | `aotx_agent_body` |

The table comes from `cuda/seam/wire.h:24-44`. The pad record takes class B at the one place
that writes it (`cuda/seam/seam.cuh:270`). Seven types are class A and fourteen are class B.

The inputs of the operator are class A: a line, a key event and the reply of a host tool. The
tokens of a sequence are class A, because a replay must apply the token that was drawn and
never draw again. The state of an agent, the text of the console, the messages of the bus and
the statistics of a tick are class B. A run derives them again from the same inputs.

## The state hash

The state hash is FNV-1a over 64 bits. The basis is `0xcbf29ce484222325` and the prime is
`0x100000001b3` (`cuda/seam/seam.cuh:44-45`). Two runs that applied the same inputs in the
same order carry the same hash.

Two places fold the hash. The apply folds the body of every class A record it takes from the
inbound ring, in slot order, from one thread (`cuda/seam/inbound.cu:134-137`). The commit of
the decode folds the body of every token record of the tick, in the order of the claimed run
(`cuda/model/commit.cu:185-193`). One thread does each fold, so the order never changes.

The tick commit record carries four fields (`cuda/seam/wire.h:87-92`). They are the state hash
and the count of class A records applied since the start. The other two are the count of
inbound slots consumed and the record count of the block. A replay that gives the same hash
proves the inputs and the token lists together.

## The tick

The tick start kernel reads the drain cursor once and the inbound head once
(`cuda/sched/step.cu:169`). It then takes the worst case of the tick that follows.

| part | records |
| --- | --- |
| the records of the tick itself | 4 |
| the journal record and the echo of each input | 2 for each input |
| the answer of the command layer to each input | 32 for each input |
| the tick load | the load of the tick |
| the decode | 704 |
| the agents and the tools | 512 |

The constants are in `cuda/seam/seam.cuh:17-35` and `cuda/sched/sched.cuh:39-45`. One tick
takes 256 inbound slots at the most, and writes 32,768 records at the most. The backlog is the
records that no block holds yet. The tick is held when the free bytes are below twice the block
size of the backlog and the worst case. It is also held when the backlog and the worst case do
not fit in the device ring.

A held tick takes no input, writes no record of its own, and writes no commit record. A held
tick is not a complete tick, and the window that a crash loses grows by it. A hold writes one
stall record when it begins and one when it ends. The second record carries the count of ticks
that were held. The display and the echo of the command line are never held.

## What a restore does

The restore program finds the newest boot directory of the journal that holds one complete
tick at the least (`disk/restore/scan.c:146`). The wall clock of the boot record orders one
boot against another. A block is a complete tick when the last record of the block is a tick
commit record (`disk/restore/scan.c:102-108`).

The replay then walks the segments of that boot in order, up to and including the block of the
last complete tick. It sends every class A record of those blocks into the inbound ring. It
leaves out the boot record and the tick commit record (`disk/restore/restore.c:48-54`). The
device is the only writer of those two types, and it makes them again on its own.

Each replayed record carries the replayed flag and the writer identity of the restore
(`disk/restore/restore.c:56-57`). The device applies each one and writes it to the new journal
with its own boot identity. Each journal is therefore self-contained, and a later restore
reads the newest complete boot only.

The last record of a replay is one restore record. The device puts its own hash in the body of
that record, at the place the record takes in the order (`cuda/seam/inbound.cu:128`). The
restore program then waits until the device consumed every published slot, and exits.

## What a restore does not do

- A restore never samples again. A replayed token joins its sequence at its position and no
  draw is taken (`cuda/model/decode.cu:225-236`). The pages of the slot are rebuilt by the
  prefill of the ticks that follow.
- A replayed prompt token must be the token that stands at its position. A token that differs
  is refused and counted (`cuda/model/decode.cu:211-215`).
- The decode makes no rows while a replay runs (`cuda/model/plan.cu:56`). The decode resumes
  when the replay ends, and takes each sequence up from the state its records leave.
- No deadline passes while a replay runs (`cuda/tool/device_tools.cu:289`). A request that
  still waits when the replay ends takes a new deadline from that tick. Its record goes in
  again with the same number, so the operator sees the request that waits
  (`cuda/tool/device_tools.cu:255-262`).
- The file read tool runs nothing again. The reply of a host tool is class A, so a restore
  applies the recorded reply. A restore is therefore not changed by the state of the file
  system.
- A replayed line has no echo on the console (`cuda/seam/inbound.cu:174`). The keys of the
  journal build the same command line again.
- The command that closes the run holds while a replay runs (`cuda/cli/parse.cu:754`). A run
  does not stop on a command that a past run typed.

## Crash semantics

The last tick that is not complete is lost. The lost window is the records that no block
holds, and the blocks that the drain had not synchronized. A held tick writes no commit
record, so a run against a drain that stopped loses more, and the journal states that.

A driver call or a runtime call that fails stops the program at once, and names the call
(`cuda/boot/check.h:14-33`). The run then holds no last flush and no clean close, and the
recovery is a restore.

The restore reports the boot it replayed, the last tick, the count of records replayed and the
hash after the replay (`disk/restore/restore.c:115`). A run that restores to the same hash as
the run before it lost nothing that the journal held.
