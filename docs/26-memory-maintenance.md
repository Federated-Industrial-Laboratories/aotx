<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Memory maintenance

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

`memory maintain PATH` applies a complete retention policy to the GPU store and releases eligible object slots and payload bytes.
The operation covers all bound conversations. It requires completed work and no partial input transfer.
The base conversation mode is unchanged. Memory remains on the GPU; maintenance does not enable disk offload.

<details>
<summary>On this page</summary>

- [Before applying a policy](#before-applying-a-policy)
- [Policy input](#policy-input)
- [Retained state](#retained-state)
- [Schema 2 and retries](#schema-2-and-retries)
- [File shrinking](#file-shrinking)
- [Capacity and checks](#capacity-and-checks)

</details>

## Before applying a policy

Use the current lineage, sequence and root from the intended store.
Maintenance can permanently release eligible history outside its retained roots and retry window.
Preserve a completed checkpoint when that older history must remain available.
The command does not silently enable offload as an alternative to reclamation.

## Policy input

The input file has exactly 64 bytes. All integers are unsigned and little endian.
Use the lineage and sequence from the current typed checkpoint.
A different lineage, sequence or root returns stale status without changing memory.

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 8 | `AOTXMNT1` |
| 8 | 4 | Schema 1 |
| 12 | 4 | Origin: explicit 0, automatic 1 |
| 16 | 4 | Exact retry window, in update sequences |
| 20 | 4 | Maximum ordinary age, in update sequences; zero disables age removal |
| 24 | 4 | Automatic pressure maintenance: disabled 0, enabled 1 |
| 28 | 4 | Used object or payload threshold, 1 through 100 percent |
| 32 | 8 | Expected current store sequence |
| 40 | 8 | Expected current root sequence; zero before activation |
| 48 | 16 | Expected lineage |

An explicit request uses origin zero. Each successful request selects typed checkpoint schema 2 and applies its complete policy.
A zero automatic flag leaves explicit maintenance available. It does not restore previously released memory.
Age is a sequence distance, not elapsed time. Expiry uses its existing sequence rule.

For example, this disk-side command creates a policy for a checkpoint in `state.bin`:

```text
python3 - <<'PY'
import struct
from pathlib import Path
state = Path("state.bin").read_bytes()
policy = bytearray(64)
policy[:8] = b"AOTXMNT1"
struct.pack_into("<6I", policy, 8, 1, 0, 4096, 0, 1, 80)
struct.pack_into("<2Q", policy, 32, struct.unpack_from("<Q", state, 32)[0],
                 struct.unpack_from("<Q", state, 96)[0])
policy[48:64] = state[48:64]
Path("policy.bin").write_bytes(policy)
PY
```

This policy retains exact retries for 4,096 sequences, disables age removal, and enables automatic maintenance at 80 percent usage.
The retry window is configurable. It does not impose a separate object-capacity limit.

Automatic maintenance uses the same recorded request and admission path as explicit maintenance.
It runs at most once per changed store sequence while the selected threshold is reached.
It waits for completed work and, with a mirror, a durable prior revision.
Each graph pass is finite. No kernel waits for disk progress.

An explicit request returns capacity status if the mirror has not durably stored the prior accepted revision.
Wait for equal committed and durable counters, with no pending slot or error, then send a current policy.

## Retained state

Maintenance retains protected objects, explicit retained or pending state, and pending intentions.
It retains every object inside the exact retry window.
Bound working focus, query pins and saved selections are also roots.
Latest ordinary objects remain unless they are deleted, expired, superseded or old enough under the selected age policy.

Each retained root keeps its exact source, embedding, selection and supersession dependencies.
The current version of every retained ID remains as an access and deletion guard.
Correction records remain when needed to keep an old retained target superseded.
Benefit and harm values remain separate and exact. Valence alone never selects an object for removal.

The GPU copies retained rows and payloads, then remaps physical selection indices before any consumer can run.
Logical IDs, exact versions, scope, focus, query bytes and context bytes remain unchanged.
A stale context remains stale.
A store filled with protected dependencies can still refuse new input. It does not discard protected state to admit a batch.

The `memory` command reports occupied object slots and payload bytes against configured capacity.
It also reports the current root, retry floor, automatic mode, and object and payload counts released by the last maintenance plan.
These released counts describe that plan; they are not an estimate of future removable state.
The mirror counters report the durable operation revision separately.

## Schema 2 and retries

The typed header remains 128 bytes and each object row remains 256 bytes.
Header schema is 2. Bytes 92 through 95 remain zero.

| Offset | Bytes | Value |
| --- | --- | --- |
| 96 | 8 | Root sequence |
| 104 | 8 | Retry floor |
| 112 | 4 | Exact retry window |
| 116 | 4 | Maximum ordinary age |
| 120 | 4 | Automatic maintenance flag |
| 124 | 4 | Pressure threshold percent |

The retry floor never moves backwards. Maintenance sets it to at least that operation's sequence minus the selected window.
A tail with any covered record at or below the floor returns stale status for the complete batch.
Above the floor, covered rows and payloads must match exactly.
Checkpoint import requires every update above the retry floor.
Maintenance refuses a retry window if a legacy store lacks any required update.

A fully covered exact retry may keep its original header.
New or partly new tails must carry the current schema, root and policy.

After activation, each new object version equals its unique update sequence.
Updates to a surviving ID preserve immutable metadata and cannot widen scope.
A missing ID can be used again only with a new sequence version.
An old exact reference cannot resolve to that new incarnation.
Root records can omit retired predecessors; their retained references and current guards must still validate.
Root status does not permit content revisions of events, media or components.

Schema 1 files and journals remain readable. Their consecutive-version rules remain unchanged.
The required CCIR typed section schema and manifest representation field at offset 20 match the image schema.
Old readers refuse the required schema 2 section.

## File shrinking

With `--memory-mirror`, a maintained root causes the drain to replace the file with a compacted incarnation.
The drain also replaces accumulated lifecycle generations when the file exceeds twice the estimated packed size.
The replacement includes every unknown optional section and keeps the configured file cap.

The writer holds the old and new leases through atomic replacement and parent-directory synchronization.
Only then can the GPU receive a durable acknowledgement.
The new incarnation starts at file generation 1. The accepted operation revision remains monotonic.
A synchronization error after replacement retains the new leased file for retry.

Replacement requires a Linux filesystem with `O_TMPFILE` support and an accessible `/proc/self/fd`.
Temporary bytes have no pathname until the new file is complete and validated.
A process exit before that point releases the temporary inode.
A later interrupted replacement leaves at most one complete pending successor for the current incarnation.
The next writer validates its lineage and predecessor digest before removing it.
An unknown conflicting file is preserved and causes refusal.

The old and new complete files can require disk space at the same time.
A lack of space reports pressure and preserves pending GPU state.
See [live checkpoints](25-memory-checkpoints.md) for file resume and durability limits.
The file still requires external models, modules and settings.

## Capacity and checks

Maintenance uses four arrays sized by `AOTX_MEMORY_OBJECTS` and the existing candidate store.
Its workspace adds `16 * AOTX_MEMORY_OBJECTS + 32` bytes, plus a 96-byte request buffer.
The schema metadata adds 32 bytes to each store allocation.
No separate small candidate list limits reclamation.

Run `ctest --test-dir build -R '^(maintenance|maintenance_guards|disk_maintenance)$'` for repeated capacity use, protected-state guards and disk failures.
The capacity check admits cumulative objects and payload bytes beyond both configured resident bounds.
The disk check injects failures and process exits at every write, sync, link and rename boundary.
All three checks use distinct batches at N=1 and N=64.

Run the complete model workflow with an existing model store and a new output directory:

```text
python3 tests/maintenance_boot_test.py BUILD SOURCE STORE OUTPUT vector-64
```

Use `vector-1` and `text-1` for the other input paths.
The check joins actual input, corrected recall, GPU maintenance, file shrinking, file resume, fresh input and exact journal recovery.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
