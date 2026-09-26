<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Time and recovery

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

GPU memory owns live state. The journal preserves an ordered replica that can lag behind it.
Recovery starts from the last complete durable tick and reapplies recorded authoritative inputs and decisions.
A visible result can therefore be newer than the saved recovery point.

<details>
<summary>On this page</summary>

- [The rules](#the-rules)
- [The two record classes](#the-two-record-classes)
- [The decisions the device makes](#the-decisions-the-device-makes)
- [The state hash](#the-state-hash)
- [The tick](#the-tick)
- [What a restore does](#what-a-restore-does)
- [What a restore does not do](#what-a-restore-does-not-do)
- [Crash semantics](#crash-semantics)

</details>

## The rules

1. Device time advances through ticks of the scheduling graph.
2. Records carry a boot identity, tick, sequence and device timer sample.
3. Tick and sequence establish order; the device timer is not wall-clock time.
4. The feeder supplies wall-clock samples as recorded input.
5. The drain writes complete blocks and synchronizes storage before advancing its durable cursor.
6. Backpressure can hold new recorded work when transport capacity is unavailable.
7. Recovery uses recorded outputs and decisions instead of making them again.

The layouts reside in `cuda/seam/wire.h`.
The device timer has a target-specific origin and supplies elapsed-time observations.
It does not replace the feeder's wall-clock samples.

## The two record classes

Class A records carry authoritative state transitions and replay inputs.
Class B records carry derived output or observations that replay can reconstruct or omit.
The following table covers the current `AOTX_REC_*` definitions:

| ID | Type | Class |
| --- | --- | --- |
| 0 | `PAD` | B |
| 1 | `BOOT` | A |
| 2 | `TICK_START` | A |
| 3 | `TICK_COMMIT` | A |
| 4 | `INPUT_LINE` | A |
| 5 | `CONSOLE` | B |
| 6 | `STALL` | B |
| 7 | `STATS` | B |
| 8 | `RESTORE` | B |
| 9 | `NOTE` | B |
| 10 | `KEY` | A |
| 11 | `COMMAND` | B |
| 12 | `BUS` | B |
| 13 | `BULK` | B |
| 14 | `TOKEN` | A |
| 15 | `SEQUENCE` | B |
| 16 | `TOOL_REQUEST` | B |
| 17 | `TOOL_REPLY` | A for external replies and late verdicts; B for derived device results |
| 18 | `MANIFEST` | B |
| 19 | `TASK` | B |
| 20 | `AGENT` | B |
| 21 | `SETTING` | A |
| 22 | `CARD` | B |
| 23 | `IMPORT` | A |
| 24 | `REMOVE` | A |
| 25 | `SELECTION` | A |
| 26 | `MODEL` | A |
| 27 | `TOKEN_STATS` | B |
| 28 | `PAGE_STATS` | B |
| 29 | `AFFECT_TRACE` | B |
| 30 | `QUALITY` | B |
| 31 | `AFFECT` | A |
| 32 | `TOOL_POLICY` | B |
| 33 | `COGNITIVE` | A |
| 34 | `COGNITIVE_RESUME` | B |
| 35 | `MEDIA` | A |
| 36 | `SERVICE_TOKEN` | B |
| 37 | `SHARED` | A |
| 38 | `POLICY` | A |
| 39 | `POLICY_CONTROL` | A |


BOOT and TICK_COMMIT describe recovery boundaries; the restored runtime produces its own boundary records.
The replay stream omits those two record types while applying the selected authoritative body records.
[Journal format](04-journal-format.md) defines the byte framing and record bodies.

## The decisions the device makes

Some authoritative records originate on the GPU.
A sampled token, memory selection, source interpretation or idle-work result must retain its original decision during recovery.
A host-tool timeout also becomes a recorded reply with the `late` status.

Recorded memory operations include exact model and processor identities where their contract requires them.
Recovery validates recorded bytes and dependent references without rerunning interpretation.
An interrupted operation cannot turn partial output into accepted state.

Model files load at startup and can change through explicit recorded runtime replacement.
Recovery verifies the file identity needed by each recorded placement.
It does not silently substitute a different file with the same model name.

## The state hash

The state hash is a 64-bit FNV-1a fold over applied authoritative bodies in their defined order.
Its basis is `0xcbf29ce484222325`; its prime is `0x100000001b3`.
The constants reside in `cuda/seam/seam.cuh`.

The tick commit stores the hash, applied-record count, consumed inbound count and block record count.
Matching replay hashes provide a consistency check for the recorded input stream.
They are not cryptographic proof of arbitrary runtime equivalence or semantic model accuracy.
Separate file digests identify assets and recorded containers.

## The tick

At tick start, the scheduler reads the drain cursor and available input.
It calculates the bounded record demand for input, decode, agents, tools and optional modules.
The current limits are named in `cuda/sched/sched.cuh` and the relevant module headers.
They depend on the configured slot and page capacities.

Not enough journal space holds new recorded work.
A held tick does not become a completed durable mutation merely because the display continues to refresh.
The system records the start and end of the hold, including held-tick counters.
No GPU kernel waits in a loop for a host cursor to advance.

The final publication nodes copy completed records into the mapped output transport.
The drain checks block framing and writes synchronized segments.
The durable cursor advances only after that storage operation completes.

## What a restore does

The disk reader selects the newest boot containing at least one complete tick.
The boot wall-clock field orders candidates.
The reader accepts records through the last complete TICK_COMMIT boundary and omits an incomplete trailing tick.

It publishes class A replay records through the inbound ring with their original ordering metadata.
The new runtime applies them under its own boot identity and writes a self-contained new journal.
A later restore can therefore use the newest complete boot.

The device preserves journal tick boundaries during replay.
A large input batch can span several apply passes, but cannot merge with the following journal tick.
This preserves the position of input relative to active agent turns.

Replay runs without the normal tick-period delay.
Its elapsed duration is not equal to the original run duration or to a fixed wall-clock cost per recorded tick.
Workload, recorded state and model cache reconstruction determine the actual cost.

At completion, the runtime reports the applied records, state hash, refusal counts, mapped pages and paced ticks.
Nonzero rejected-input counts stop startup.
A refusal that correctly reproduces an earlier operation can appear in `decode_refused` without making replay invalid.

## What a restore does not do

| Operation | Recovery behavior |
| --- | --- |
| Token sampling | Apply the recorded token at its recorded position. |
| Prompt tokenization check | Refuse a recorded prompt token that does not match its expected position. |
| Completed host tool | Apply the saved reply without executing the tool again. |
| Memory selection | Apply the saved references and validate their exact versions. |
| Model interpretation | Validate recorded bounded outputs; do not generate a replacement. |
| Completed idle review | Restore the accepted cue and processed frontier. |
| Old `quit` command | Preserve its record without stopping the restored instance. |

Cache reconstruction can run model prefill to rebuild execution state.
That reconstruction is different from generating replacement conversation output.
Decode resumes new work only after recorded replay completes.

A pending operator authorization remains pending.
Its execution deadline starts only after a grant; completed recorded replies remain complete.
A pending host request can be republished after replay without repeating an already answered request.

## Crash semantics

A failed run can lose records outside the last complete synchronized tick.
The loss window includes unpublished records and unsynchronized blocks.
Backpressure can extend the interval since the last durable progress.

An essential child failure stops the runtime and produces a failure status.
Normal shutdown also checks essential child exit results.
A writer failure during shutdown cannot be treated as a successful final save.

A CUDA context failure or forced process termination can prevent the final flush.
Preserve the journal and recover its durable prefix.
Compare the restored hash with the saved commit, not with an uncommitted later GPU observation.

[Operation](07-operation.md#failures-and-recovery) describes diagnosis and restart.
[Complete runtime files](28-runtime-files.md) extends recovery to packaged assets and persistent shared state.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
