# Live memory regression checks

These checks exercise explicit memory bindings, source admission and recovery.
Unbound conversations use the base path.
Use [the test guide](../docs/testing.md) to select checks for the changed contract.
Run the device checks with `AOTX_AFFECT=ON` and `AOTX_AFFECT=OFF`.

## Check matrix

| Check | Consumer and source dependencies | Fixture and failure condition |
| --- | --- | --- |
| `checkpoint` | Device snapshot, live admission and disk CCIR writer | N=1/N=64 distinct bindings, exact import, stale context, pressure and disk failures |
| `checkpoint_boot_test.py` | Boot, feeder, drain, CCIR and real model | Removed source files and old journal; exact bindings and fresh corrected recall after resume |
| `live_memory` | Real inbound, live state nodes and agent prompt; cognitive, seam, scheduler and agent code | Distinct N=1 and N=64 principals, exact prompt bytes, corrections, tool results and recorded replay |
| `live_feed` | File reader and feeder publication; disk feed and CCIR reader | N=1, N=64 and maximum load; malformed files, interrupted reads, exact bytes and bounded groups |
| `live_transcript` | Typed audit after complete choices; disk transcript reader | N=1 and N=64 distinct rows; partial or changed choices cannot produce accepted input |
| `live_boot_test.py` | Boot, model prompt, file feeder, drain and cold restore | Base reply, private memory, correction, removed input files and exact restored hashes, audit and replies |

## Required failure behavior

`live_memory` runs 70 successive requests in its singleton case, beyond the saved-query
file limit. Its wide case gives each of 64 bindings different input and selected memory.
The current memory budget stays fixed. Restore must reproduce exact prompt bytes with
zero searches. A foreign-principal fixture uses an otherwise valid foreign memory pin.
Removing the principal binding check must make this fixture fail at both batch sizes.

Duplicate choices must refuse restore after an accepted or refused request completes.
Audit fixtures place ordinary input before typed input and use distinct prior manifest turns.
The accepted input and selection must keep the device turn, independent of the request ordinal.

## Complete workflow

Run the boot check with an existing compatible language model store and a new output directory:

```text
python3 tests/live_boot_test.py BUILD SOURCE STORE OUTPUT
```

The driver records each command and verdict. Response wording checks are separate from
structural recovery checks. It removes only its own input files and stops only its own
boot processes. It does not download a model or run an encoder.

## Related regression checks

Affected base checks include `conversation`, `request_completion`, `wrap_prompt`,
`call_prompt`, `call_schema`, `result_bounds`, `tool_policy`, `prompt_policy`, `seam`,
`sched`, `settings` and disk feed, transcript, journal, restore and CCIR checks.
The prepared `recall`, `recall_cli`, `cognitive` and `cognitive_file` checks remain required.

Checkpoint checks establish the [memory-state mirror](../docs/25-memory-checkpoints.md).
They do not establish complete runtime packaging, media encoding, remote authentication or native module execution.
