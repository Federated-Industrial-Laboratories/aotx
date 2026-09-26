<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Creator policy components

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

A creator policy selects when opted-in memory work can start.
Its input contains GPU memory use, capacity, retention settings and the last accepted observation.
Its output requests maintenance, requests admitted appraisal or task review, or remains quiet.
Existing memory rules protect retained references and conversation state.
Explicit operator maintenance keeps its direct command path.

The supplied policy and data rules require no native code.
A native policy supplies a CUDA cubin or PTX image with the same batched interface.
Each policy has private GPU state. Complete decisions and state bytes enter the recovery journal.
Base conversations and stores without a selected policy keep their existing behavior.

<details>
<summary>On this page</summary>

- [Select an interface](#select-an-interface)
- [Create a policy](#create-a-policy)
- [Native interface](#native-interface)
- [Package and recover](#package-and-recover)
- [Update a policy](#update-a-policy)
- [Operator controls](#operator-controls)
- [Device capacity](#device-capacity)

</details>

## Select an interface

| ABI | Permitted work | Enable separately |
| --- | --- | --- |
| 1 | Memory maintenance | Automatic maintenance |
| 2 | Maintenance and admitted appraisal | Appraisal writes and background work |
| 3 | Maintenance, appraisal and task reviews | Task review control |

A policy proposal cannot increase scope or replace the consumer's admission checks.
The supplied data rules need no native-code trust grant.
Native components execute in the shared CUDA context and require explicit operator trust.

## Create a policy

Store the source and compilation details in a provenance file. Supply the component license in a separate file.
The packager preserves both files in the policy bundle.

```text
aotx_policy_pack --output policy.bin --mode rules \
  --pressure 70 --minimum-move 1 --backoff 8 \
  --provenance provenance.txt --license LICENSE
aotx_policy_pack --inspect policy.bin
aotx_boot --policy policy.bin --journal JOURNAL
```

The pressure value is a percentage of object capacity or payload capacity.
Zero uses the store pressure setting. The default is zero.
The minimum movement and backoff values count accepted memory sequence movement since the last maintenance proposal.
Both default to one. Data policies use state schema 1 and 16 state bytes.

The default bundle uses ABI 1, which supports maintenance only.
Add `--abi 2` to select appraisal work support.
Add `--abi 3` for [task review support](38-task-reviews.md). Inspection reports the exact version.

Existing ABI 1 bundles and their recorded decisions keep their byte format.
Unsupported versions are refused. Native code is never assigned a newer ABI during loading.

Maintenance requires its automatic option in the [memory controls](26-memory-maintenance.md).
ABI 2 or 3 can process admitted appraisal work with automatic maintenance disabled.
Selecting a policy does not enable either memory option.

A policy runs once for each eligible changed memory observation.
ABI 2 also observes the appraisal work revision, including configuration changes.
Changed object or payload byte counts also permit a new ABI 2 evaluation.
This includes completed memory reclamation at the same memory sequence and root.

Foreground work, scheduler hold, replay, pause and checkpoint pressure prevent a new evaluation.
Unchanged idle ticks do not enter the policy. No input or model generation is created to keep it active.

## Native interface

The public interface is `cuda/policy/abi.h`.
The example `examples/policy/maintenance.cu` supplies a finite maintenance entry.
Compile it for the target device:

```text
nvcc --ptx -std=c++17 -arch=sm_86 -Icuda \
  examples/policy/maintenance.cu -o maintenance.ptx
aotx_policy_pack --output policy.bin --mode native \
  --image maintenance.ptx --format ptx --kernel aotx_creator_maintenance \
  --architecture 86 --state-schema 1 --state-bytes 16 --threads 64 \
  --registers 255 --shared-bytes 0 --local-bytes 0 \
  --pressure 70 --minimum-move 1 --backoff 8 \
  --provenance provenance.txt --license LICENSE
```

Use `nvcc --cubin` and `--format cubin` for a compiled image.
Choose the architecture and measured resource bounds for the selected device and component.
The entry has six parameters in this order:

```c
const aotx_policy_input *input,
const unsigned char *prior,
aotx_policy_output *output,
unsigned char *next,
uint32_t count,
uint32_t stride
```

Input rows are 128 bytes. Output rows are 64 bytes.
Private state row `i` starts at byte `i * stride` in each state buffer.

Process each valid row once. Set reserved output fields to zero.
ABI 1 returns `AOTX_POLICY_QUIET` or `AOTX_POLICY_MAINTAIN` with status zero.
ABI 2 can also return `AOTX_POLICY_APPRAISE` for admitted pending work.

ABI 1 keeps all reserved input words zero.
ABI 2 sets `reserved0` to 2 and uses `enabled` for the maintenance option.
`reserved1[0]` contains the admitted appraisal count.
`reserved1[1]` and `reserved1[2]` contain the low and high words of the 64-bit work revision.
A zero pending count grants no appraisal work.
Maintenance-disabled input grants no maintenance work, even when appraisal is pending.

ABI 3 sets `reserved0` to 3 and permits `AOTX_POLICY_REVIEW`.
Bit 0 of `enabled` permits maintenance. Bit 1 reports eligible, enabled task review work.
The appraisal count stays in `reserved1[0]`. The other two words hold the control and review-work revision.

Source, root and capacity observations continue to bind every proposal.
A review proposal with bit 1 clear is invalid.
The supplied rules select maintenance, appraisal and review in that priority order.

State is opaque portable data. Do not store device addresses in it.
The supplied example uses two little-endian counters and preserves the remaining declared state bytes.

The live runtime supplies one observation for the shared cognitive identity, across all execution slots.
The entry accepts a batch of independent observations and private state rows.
One runtime observation does not limit the number of users or memory objects.

Native activation requires the exact complete bundle digest from inspection:

```text
aotx_boot --policy policy.bin --policy-trust SHA256 --journal JOURNAL
```

Only the local operator supplies this grant. A file cannot grant trust to itself.
Changing code, metadata or license bytes changes the digest.
Activation checks the target, parameter layout, entry name and declared resource bounds before graph capture.
A required native entry that fails admission stops activation.

Native code shares the CUDA context. The loader does not provide a sandbox or kernel preemption.
Only grant trust to code whose memory access and termination are acceptable for that deployment.
A finite graph condition skips native entry during quiet, paused, held and replay ticks.
Malformed output records an error, preserves prior accepted state and pauses further evaluation.

An appraisal proposal is consumed once. Its memory source, root and work revision must remain current.
The accepted object and payload byte counts must also match.
The appraisal option, pause, stop, foreground and recovery guards still apply at consumption.
The proposal does not grant memory access, change sharing scope or execute tool actions.

## Package and recover

Add `--policy policy.bin` to the [complete runtime pack command](28-runtime-files.md).
The container stores the full required bundle, including its image and metadata.
Packing and inspection execute no code and grant no trust.
Native activation still needs the external digest grant:

```text
aotx_boot --ccir identity.aotxccir --policy-trust SHA256 --journal NEW_JOURNAL
```

Do not combine `--ccir` with `--policy`. The complete file selects its own required component.
The stopped durable file can restart without the original component paths.
Replay restores recorded decisions and exact private bytes without running the creator entry again.
An undeclared policy revision, ABI or incompatible state schema refuses recovery.

The decision header records zero at byte 20 for ABI 1, two for ABI 2 and three for ABI 3.
Replay checks this marker against the selected bundle and input marker.

The accepted work revision survives recovery. Recorded replay issues no fresh appraisal proposal.
A compatible update can preserve state through the explicit conversion command below.
Arbitrary state-schema migration is not implemented.

Each decision is published in bounded journal fragments. Accepted state changes after the complete final fragment.
A complete runtime checkpoint waits for this transition. An incomplete raw journal candidate has no committed state or work action.
Memory-only exports do not contain the creator runtime.

## Update a policy

Stop the runtime after a durable save. Inspect the old and new bundles before conversion.

```text
aotx_policy_update --runtime identity.aotxccir --from OLD_SHA256 \
  --policy next-policy.bin --state-map preserve --output updated.aotxccir
```

The old digest must match the selected bundle in the input file.
The preserve mapping requires the same policy ABI, state schema and state size.
Use this mapping only when each state byte keeps its meaning in the new policy.
Different schemas, sizes and ABIs are refused. No implicit reset or conversion occurs.

The command creates a separate complete file. The input file stays unchanged.
An existing output path is refused. Publication occurs only after complete file validation.
The original replay, memory, models, identity and sharing sections remain unchanged.

The updated file includes every prior bundle and its exact last decision boundary.
Recovery validates historical records against their original policy parameters.
It restores the recorded bytes without executing historical native code.
New work uses the selected new policy. Native activation requires its new external trust digest.

```text
aotx_boot --ccir updated.aotxccir --policy-trust NEW_SHA256 --journal NEW_JOURNAL
```

The history asset is required. It uses runtime feature bit 64 and section schema 5.
Earlier readers refuse this required schema. Later saves retain the complete history.
At most eight prior revisions are supported. A repeated revision or an additional update is refused.

## Operator controls

```text
policy status
policy pause
policy resume
policy stop
```

Pause prevents new evaluations. Stop also records the stopped state. Resume clears either condition unless recovery has failed.
An in-progress decision finishes publication; paused or stopped state prevents its work action.

Status shows the decision, memory source, state bytes, error status, evaluation count, time and saved file generation.
ABI 2 also reports pending appraisal, current work revision and accepted work revision.
The state hash is FNV-1a over accepted private bytes. It is a recovery diagnostic, not a trust digest.

Evaluation counts and times apply to the current process. Accepted decisions and private state survive recovery.

## Device capacity

`AOTX_POLICY_STATE_BYTES` sets native private state capacity through CMake. Its default is 65536 bytes.
`AOTX_POLICY_IMAGE_BYTES` independently sets image capacity. Its default is 16777216 bytes.

These capacities do not set memory object or payload capacity.
GPU allocation and graph admission must fit the selected device.
The required policy asset uses runtime feature bit 16 and runtime section schema 3.
Existing files without policies retain their prior runtime schemas.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
