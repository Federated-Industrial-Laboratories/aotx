# Creator policies

A creator policy selects when opted-in automatic memory maintenance can start.
Its input contains GPU memory use, capacity, retention settings and the last accepted observation.
Its output requests maintenance or remains quiet. Existing memory rules protect retained references and conversation state.
Explicit operator maintenance keeps its direct command path.

The supplied policy and data rules require no native code.
A native policy supplies a CUDA cubin or PTX image with the same batched interface.
Each policy has private GPU state. Complete decisions and state bytes enter the recovery journal.
Base conversations and stores without a selected policy keep their existing behavior.

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

The store must have automatic maintenance enabled through its [memory controls](26-memory-maintenance.md).
Selecting a policy does not enable that store option.
A policy runs once for each eligible changed memory observation.
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
Return `AOTX_POLICY_QUIET` or `AOTX_POLICY_MAINTAIN` with status zero.

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
A different policy revision or incompatible state schema refuses recovery.
State migration requires explicit conversion to a new compatible revision.

Each decision is published in bounded journal fragments. Accepted state changes after the complete final fragment.
A complete runtime checkpoint waits for this transition. An incomplete raw journal candidate has no committed state or maintenance action.
Memory-only exports do not contain the creator runtime.

## Controls and capacity

```text
policy status
policy pause
policy resume
policy stop
```

Pause prevents new evaluations. Stop also records the stopped state. Resume clears either condition unless recovery has failed.
An in-progress decision finishes publication; paused or stopped state prevents its maintenance action.

Status shows the decision, memory source, state bytes, error status, evaluation count, time and saved file generation.
The state hash is FNV-1a over accepted private bytes. It is a recovery diagnostic, not a trust digest.

Evaluation counts and times apply to the current process. Accepted decisions and private state survive recovery.

`AOTX_POLICY_STATE_BYTES` sets native private state capacity through CMake. Its default is 65536 bytes.
`AOTX_POLICY_IMAGE_BYTES` independently sets image capacity. Its default is 16777216 bytes.

These capacities do not set memory object or payload capacity.
GPU allocation and graph admission must fit the selected device.
The required policy asset uses runtime feature bit 16 and runtime section schema 3.
Existing files without policies retain their prior runtime schemas.
