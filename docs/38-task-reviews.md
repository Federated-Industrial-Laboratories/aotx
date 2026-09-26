<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Task reviews

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

An enabled idle policy can save a review cue for a supported task outcome.
A later matching task receives the cue and its exact evidence.
The cue does not add advice, a causal claim, a user turn or a tool request.
No model call is required to create it.

<details>
<summary>On this page</summary>

- [Enable task reviews](#enable-task-reviews)
- [Evidence and scope](#evidence-and-scope)
- [Scheduling and recovery](#scheduling-and-recovery)
- [API controls](#api-controls)
- [Verify a later task](#verify-a-later-task)

</details>

## Enable task reviews

Use a [creator policy](34-creator-policy.md) with `--abi 3`.
Review is off by default.
Prepare retained sources, admitted task descriptors and completed supported appraisal before enabling reviews.
A policy without eligible evidence remains quiet.

```text
policy review on
policy status
policy pause
policy resume
policy review off
policy stop
```

Pause and stop prevent new work. Resume clears these conditions.
An admitted result can finish recording after pause or stop.
Work interrupted before construction records an interruption and creates no cue.
Turning review off preserves previously saved reviews and their evidence.

## Evidence and scope

A source must have a completed current appraisal with a known nonzero benefit or harm.
It must name a registered task and a participant.
The current source, appraisal, work record, relationship and task registration must remain accessible.
Generated sources, unsupported outcomes and absent task bindings are ineligible.

Each result contains an inferred selection and a typed review cue.
The selection retains five exact references: source, appraisal, completed work record, relationship and task registration.
The cue has object kind 13 and payload `AOTXMEM4`, version 4.
Its fixed text requests review of the supported outcome before repeating the task.

Recall requires the same task and an explicit matching participant.
It rechecks current versions, scope, expiry and withdrawal.
The cue and all five supporting records must fit together. A capacity limit refuses the query without partial context.

A correction invalidates the earlier cue through its dependency selection.
New supported evidence can produce a replacement. Old records remain available for inspection.
Withdrawn cues retain their type in complete files and require review support.
Reading or repeating a cue creates no new evidence or exposure.

## Scheduling and recovery

One batch contains at most 64 source revisions, in increasing revision order.
Foreground work, pause, stop, checkpoint pressure and journal hold prevent admission.
Unchanged observations cause no repeated policy entry or completed review.
Capacity refusal blocks retries against the same store contents.

Requests use live operation 20. Results use operation 21.
A complete result records its exact object tail before atomic publication.
The processed source frontier survives journal recovery and memory reclamation.

Interrupted work advances that frontier without adding a cue. It is not automatically repeated.
A later accepted source revision remains eligible.

A partial result never becomes a completed cue during recovery.
Recovery records an interruption before new idle work can start.
A complete runtime file carries required feature bit 256 and runtime section schema 7.
Earlier readers refuse this required feature. Memory-only exports omit policy control and scheduling state.

Each result contains 64 header bytes and at most 128 object rows with their bounded payloads.
Publication emits at most 64 journal fragments per graph tick.
The current 64-row maximum needs six publication ticks.
Construction is one finite GPU block. It cannot be preempted inside its kernel.

A queued memory query waits for the current result publication to finish.
Other active foreground work can proceed while the remaining result fragments are recorded.

## API controls

GET `/aotx/v1/policy` returns aggregate state, progress, result status and saved generation.
It requires `telemetry` or `policy_manage`. It contains no source text or private source identifiers.
The maximum duration measures new work in this process. The last duration can include replay.

POST `/aotx/v1/policy` requires the separate `policy_manage` grant and ABI 3.

```json
{"action":"pause","epoch":71,"control_revision":4}
```

Actions are `pause`, `resume`, `stop`, `review_on` and `review_off`.
Use the epoch and control revision from the latest GET response.
Each accepted mutation advances the revision once.
A stale revision returns 409; an old epoch returns 410; journal pressure returns 429.
A retry cannot apply an old command after a later accepted control.
Accepted native controls and local commands recover through the same device control state.

## Verify a later task

Submit a later task with the same admitted task and participant identifiers.
Inspect the selected review cue and all five evidence references.
A correction or withdrawal must remove the old cue from current recall.

Save and recover the complete runtime before repeating the read.
Recovery must preserve the processed frontier and exact evidence without creating another review or model call.
See [testing](testing.md) for selecting the affected contract and complete workflow checks.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
