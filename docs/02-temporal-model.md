# The temporal model

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| ring | a single-producer, single-consumer ring buffer in pinned host memory |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| drain | the disk-side process that writes the outbound ring to the journal (a log writer) |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| bus | an append-only message log between agents (a message bus) |

The device is ahead and the disk is behind. This rule maintains every module boundary. This
document gives the rules, the two record classes, what a restore does, and what a crash loses.

## The rules

1. The authoritative state resides in device memory while the system operates. The disk maintains a replica that lags.
   The disk is authoritative at a cold start only.
2. Device time is the tick. The tick start kernel adds one to the tick counter, and every other
   kernel reads it (`cuda/sched/step.cu`, `aotx_sched_tick_start`). The counter starts at zero, so
   the first tick is tick 1 (`cuda/time/tick.cu`, `aotx_time_tick`).
3. Every record carries a header of the boot identity, the tick, the record sequence and a device
   clock sample (`cuda/seam/wire.h`, `aotx_record_header`). Order comes from the tick and the
   sequence. The clock sample reads the device nanosecond timer. Its origin is target-specific.
   It measures elapsed device time and lag, not wall-clock time.
4. The feeder writes a tick start record that carries the wall clock, ten times a second
   (`disk/feed/feed.c`, `AOTX_TICK_NS`). Device time therefore pairs with wall-clock time on disk.
   The device reads no host clock.
5. At the end of each tick the flush node copies the records of that tick into the host ring as
   one block (`cuda/seam/flush.cu`, `aotx_seam_flush`). One tick makes one block.
6. The drain writes blocks to journal segments. It synchronizes the segments before it moves its
   cursor, so a crash never loses a block that the producer counts as safe (`disk/drain/drain.c`,
   `aotx_host_ring_advance`).
7. No kernel blocks on the host. Backpressure is decided at tick start from one read of the drain
   cursor. Nothing spins on a host field.
8. A restore replays the class A records of the newest complete journal, up to the last complete
   tick. A tick that is not complete on disk is left out.
9. The model files are read once at the start of a run, before the first tick
   (`cuda/boot/boot_host.cu`, `aotx_boot_models`). Nothing reads them again while the run goes on.
10. A flow decision the device makes on its own is a class A record, and the device writes it
    itself. The late verdict of a tool request is the one decision of this kind today.

The [PTX timer specification](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)
defines the device timer. Its behavior depends on the target.

## The two record classes

A class A record is authoritative and a restore replays it. A class B record is derived from class
A records and a restore does not replay it. The class is in the header of every record, and
the wire header gives the class of each type.

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

The table comes from the names that begin `AOTX_REC_` in `cuda/seam/wire.h`. The pad record has
class B at the one place that writes it (`cuda/seam/seam.cuh`, `aotx_seam_pad`). Seven types are
class A and fourteen are class B.

The inputs of the operator are class A: a line, a key event and the reply of a host tool. The
tokens of a sequence are class A, because a replay must apply the token that was drawn and never
draw again. The state of an agent, the text of the console, the messages of the bus and the
statistics of a tick are class B. A run derives them again from the same inputs.

## The decisions the device makes

A record from outside is class A because the device cannot derive it. A decision the
device makes on its own is class A for a second reason: a replay must not make it again. The late
verdict of a tool request is the one decision of this kind in this version.

The tool step writes that verdict as a tool reply record of the late status
(`cuda/tool/device_tools.cu`, `aotx_tool_late_body`). The record is class A, and the same kernel
folds it into the state hash (`cuda/tool/device_tools.cu`, `aotx_tool_step`). A restore applies it
at its position in the order, so the request fails again in the same place. The slots claim
one run of sequences in slot order, so two runs of the same inputs write the same records in the
same places.

## The state hash

The state hash is FNV-1a over 64 bits. The basis is `0xcbf29ce484222325` and the prime is
`0x100000001b3` (`cuda/seam/seam.cuh`, `AOTX_FNV_BASIS`). Two runs that applied the same inputs in
the same order carry the same hash.

Two places fold the hash. The apply folds the body of every class A record from the
inbound ring, in slot order, from one thread (`cuda/seam/inbound.cu`, `aotx_seam_apply_inbound`).
The commit of the decode folds the body of every token record of the tick, in the order of the
claimed run (`cuda/model/commit.cu`, `aotx_decode_commit`). One thread does each fold, so the
order never changes.

The tick commit record carries four fields (`cuda/seam/wire.h`, `aotx_commit_body`). They are the
state hash and the count of class A records applied since the start. The other two are the count
of inbound slots consumed and the record count of the block. A replay that gives the same hash
proves the inputs and the token lists together.

## The tick

The tick start kernel reads the drain cursor once and the inbound head once (`cuda/sched/step.cu`,
`aotx_sched_tick_start`). It then calculates the worst case of the next tick.

| part | records |
| --- | --- |
| the records of the tick itself | 4 |
| the journal record and the echo of each input | 2 for each input |
| the answer of the command layer to each input | 32 for each input |
| the tick load | the load of the tick |
| the decode | 704 |
| the agents and the tools | 512 |

The constants are `AOTX_TICK_RECORDS_OWN` and `AOTX_CLI_RECORDS_EACH` in `cuda/seam/seam.cuh`. The
decode and agent parts are `AOTX_DECODE_RECORDS_MAX` and `AOTX_AGENT_RECORDS_MAX` in
`cuda/sched/sched.cuh`. One tick accepts 256 inbound slots at the most, and writes 32,768 records at
the most. The backlog is the records that are not yet in a block. Backpressure blocks the tick when
the free bytes are below twice the block size of the backlog and the worst case. It also blocks the tick when the backlog
and the worst case do not fit in the device ring.

A blocked tick accepts no input, writes no record of its own, and writes no commit record. A blocked tick
is not a complete tick, and the window that a crash loses grows by it. The system writes one stall
record when backpressure starts and one when it ends. The second record carries the count of ticks that were
blocked. Backpressure never blocks the display or the command-line echo.

## What a restore does

The restore program finds the newest boot directory of the journal that contains one complete tick at
the least (`disk/restore/scan.c`, `aotx_journal_latest`). The wall clock of the boot record orders
one boot against another. A block is a complete tick when the last record of the block is a tick
commit record (`disk/restore/scan.c`, `AOTX_REC_TICK_COMMIT`).

The replay then walks the segments of that boot in order, up to and including the block of the
last complete tick. It sends every class A record of those blocks into the inbound ring. It leaves
out the boot record and the tick commit record (`disk/restore/restore.c`, `replay_block`). The
device is the only writer of those two types, and it makes them again on its own.

Each replayed record carries the replayed flag and the writer identity of the restore
(`disk/restore/restore.c`, `AOTX_FLAG_REPLAYED`). The device applies each one and writes it to the
new journal with its own boot identity. Each journal is therefore self-contained, and a later
restore reads the newest complete boot only.

The apply processes the records of one journal tick in one tick of the restored system
(`cuda/seam/inbound.cu`, `aotx_seam_replay_take`). The restore duration therefore equals the tick count of the
run it replays. The reason is the order of the inputs against the turns. A line that reaches an
agent which still runs the turn before it is refused (`cuda/cli/parse.cu`, `aotx_cli_say_text`). A
replay that processed every available record would put a line where the original system never
had one.

The clock of the replay moves only when the apply saw a record of a later tick. That record
proves the tick of the clock complete. A tick of the journal with more records than the apply
processes therefore spills into the ticks after it. It never merges with the tick that follows it.

A record the device writes while the replay runs carries the replay flag (`cuda/seam/wire.h`,
`AOTX_FLAG_REPLAY`). The journal already answers the request such a record names. The drain
therefore derives no request line from it, and the feeder executes no tool a second time. A
request that is still pending when the replay ends appears again in a record without the flag.

The last record of a replay is one restore record. The device writes its own hash in the body of
that record, at the position of the record in the order (`cuda/seam/inbound.cu`,
`aotx_restore_body`). The restore program exits after the device consumes every published slot.

A restore gives back every turn that the run before it completed. Each one comes back with the
same token count, finish, tool, request and output digest. The token count comes from the read
that processed the reply, because the slot of the sequence is used again (`cuda/agent/records.cuh`,
`aotx_agent_manifest`). The replay gate compares those fields of every completed turn, one by one
(`tests/replay_test.sh`, `compare_turns`).

## What a restore does not do

- A restore never samples again. A replayed token joins its sequence at its position and no draw
  occurs (`cuda/model/decode.cu`, `aotx_seq_apply`). The pages of the slot are rebuilt by the
  prefill of the ticks that follow.
- A replayed prompt token must be the token at its position. A token that differs is
  refused and counted (`cuda/model/decode.cu`, `AOTX_TOKEN_PROMPT`).
- The decode makes no rows while a replay runs (`cuda/model/plan.cu`, `aotx_decode_plan`). The
  decode resumes when the replay ends, and continues each sequence from the state its records
  leave.
- A request that requires operator authorization has no deadline (`cuda/tool/tool.cuh`,
  `AOTX_TOOL_NO_DEADLINE`). The deadline of `tool.deadline_ticks` (`cuda/settings/keys.h`,
  500 ticks unless a setting changes it) starts at the tick of the grant (`cuda/agent/table.cu`,
  `aotx_agent_authorize`). No deadline passes while a replay runs (`cuda/tool/device_tools.cu`,
  `aotx_tool_step`). A pending request receives a new deadline from
  that tick. Its record goes in again with the same number, so the operator sees it
  (`cuda/tool/device_tools.cu`, `aotx_tool_note_request`).
- The file read tool runs nothing again. The reply of a host tool is class A, so a restore applies
  the recorded reply. A restore is therefore not changed by the state of the file system.
- A replayed line has no echo on the console (`cuda/seam/inbound.cu`, `aotx_cli_echo`). The keys
  of the journal build the same command line again.
- Replay blocks the command that closes the system (`cuda/cli/parse.cu`, `aotx_cli_act`).
  A run does not stop on a command that a past run typed.

## Crash semantics

The last tick that is not complete is lost. The lost window is the records outside a block,
and the blocks that the drain had not synchronized. A blocked tick writes no commit record, so a system
against a drain that stopped loses more, and the journal states that.

A driver call or a runtime call that fails stops the program at once, and names the call
(`cuda/boot/check.h`, `aotx_check_driver`). The system then has no last flush and no clean close,
and the recovery is a restore.

The restore reports the boot it replayed, the last tick, the count of records replayed and the
hash after the replay (`disk/restore/restore.c`, `report`). A run that restores to the same hash
as the system before it lost nothing that the journal contained.
