# Automatic appraisal

Appraisal converts retained external reports into source-backed assessments and relationship evidence on the GPU.
It uses the resident language model, the typed memory store and the existing work scheduler.
The function is optional. New runs keep appraisal writes, recall defaults and background work off.

The source event keeps its exact text, owner, scope and admitted subject.
An inferred name does not change the subject ID.
A report about another person does not establish that person's authenticated identity.
Unsupported benefit, harm, regard and task trust remain unknown.

## Model limits

Qwen3-4B-Q4_0 can return unknown values for all dimensions when a report contains both helpful and harmful actions.
The same model can answer questions about those actions correctly.
An unknown result does not establish that the source lacks useful evidence.
An assessment with no known intensity does not add a recall priority or a derived evidence group.
Appraisal accuracy for this model remains unverified across the supported report types.

## Controls

Set appraisal after a memory store is ready.
The console records each control and applies settings at a quiet memory boundary.
`appraisal control: pending` means the new setting has not yet been applied.
Operation 15 reports the accepted configuration or a refusal status.

| Command | Effect |
| --- | --- |
| `appraisal` or `appraisal status` | Show accepted flags, pending work and process counters |
| `appraisal on` | Enable writes and recall defaults |
| `appraisal off` | Disable writes, recall defaults and background work |
| `appraisal writes on` or `off` | Select new queue writes without changing recall defaults |
| `appraisal recall on` or `off` | Select recall defaults without changing writes |
| `appraisal background on` or `off` | Select idle work; writes must also be enabled |
| `appraisal run` | Request one explicit batch, including retained refused or interrupted work |
| `appraisal limits PAGES TOKENS TICKS ROWS` | Set the execution limits for each batch |
| `appraisal priority FLOOR BOOST` | Set the optional recall floor and priority |

The flags are writes=1, recall=2 and background=4.
`appraisal on` preserves the current background setting.
An explicit run does not enable background work.
The policy pause and stop controls also suspend appraisal admission.
An active appraisal cancels when a new configuration or foreground work requires the device.
A scheduler hold suspends the whole tick path; cancellation can complete when ticks resume.

The initial limits are 160 pages per sequence, 512 output tokens, 16384 ticks and 64 source rows.
The output buffer is 4096 bytes per source.
These are work limits, not retained memory limits.
The existing memory object and payload settings determine retained capacity.

The row limit accepts 1 through 64; pages and ticks must be positive.
The token limit accepts 1 through 4096.
The recall floor and priority accept 0 through 1000000.

## Source and result

With writes enabled, a retained external source with a subject receives one pending queue object.
The source and queue enter memory in the same atomic write.
Automatic and explicit retention use the same queue format.
A source without a subject receives no relationship queue.
Pending work retains its exact source, configuration version and optional authored task descriptor.

The queue uses `AOTXAPQ1`, schema 1, with a 160-byte payload.
States are pending=0, complete=1, refused=2 and interrupted=3.
One successful result adds a completed queue version, one assessment and one relationship object.
A completed queue cannot create another exposure.
Internal generation and repeated recall create no external exposure.

The assessment uses appraisal schema 2 with a 128-byte payload.
Its existing prefix keeps separate benefit, harm, arousal, consequence and confidence fields.
Schema 1 remains readable.
The relationship uses `AOTXREL1`, schema 1, with a 192-byte payload.
It keeps separate regard gain/loss and task trust gain/loss, one exposure and exact evidence spans.
These values are model interpretations, not calibrated probabilities or access grants.

The model returns one JSON object for each source.
It uses these exact field names in this order, with no added or repeated fields:

```json
{"benefit":4294967295,"harm":4294967295,
 "arousal":4294967295,"consequence":0,"confidence":4294967295,
 "regard_gain":4294967295,"regard_loss":4294967295,
 "trust_gain":4294967295,"trust_loss":4294967295,
 "evidence":"","task":"","commitment":"","correction":0}
```

Numeric dimensions use 0 through 1000000, or 4294967295 for unknown.
These are ordinal model estimates. The source does not need to contain a numeric measurement.
Unknown records that the model supplied no supported conclusion for that dimension.
Explicit helpful and harmful outcomes each require a positive estimate; they do not cancel each other.
Consequence uses 0 through 4, with zero for no established consequence.

Quotes must be unique, exact UTF-8 substrings of the source.
The decoder emits dimensions before evidence.
If all scaled dimensions are unknown and consequence is zero, all quotes are empty and correction is zero.
Any known interpretation requires nonempty evidence.
Empty evidence requires unknown interpretations, consequence 0, no task or commitment quote and correction 0.

Commitment requires an explicit promise of future action. A completed action alone requires an empty commitment.

Known task trust requires an admitted task ID and its exact authored task descriptor.
The descriptor is a current `AOTXMEM1` cue whose object ID is the task ID.
Its full text must occur once in the source and equal the task quote.
An opaque task ID alone does not establish task trust.
An absent task restricts only task trust. Other dimensions can use supported actions, outcomes and feelings.

Corrections can supersede current inferred assessments for the same subject, owner and scope.
The paired relationship is superseded with the assessment.
Protected, withdrawn, expired or unavailable evidence cannot be a correction target.
The correction context contains up to 16 recent eligible assessments per source.
The original reports remain available as historical source events.

## Recall

Recall defaults add appraisal priority only when the caller has not supplied explicit appraisal controls.
Explicit task and participant constraints retain their meaning.
Scope, current versions, source availability and the similarity floor apply before optional priority.
For schema 2, the source, assessment, completed queue and relationship must fit as one selection group.
Semantic search can miss reports that differ only by numeric identifiers.

The memory prompt marks stored records as historical data.
Instructions within those records have no current authority.
The current caller request follows the memory segment.

The context identifies the source subject and keeps positive and negative values separate.
Task trust applies only to the matching task context.
Superseded interpretations do not contribute current relationship evidence.

Disabling writes preserves existing evidence and permits recall to continue.
Disabling recall defaults adds no default appraisal selection.
An explicit caller appraisal request still uses its supplied controls.
Existing conversations continue through their normal path when these functions are off.

## Work and recovery

The supplied policy can select background appraisal without a creator bundle.
A creator bundle must select policy ABI 2 to request appraisal.
ABI 1 remains maintenance-only. Required native code never silently selects the supplied implementation.

Foreground arrival cancels at a decoder boundary and releases internal sequence leases.
No partial interpretation is published.
A refused or interrupted queue requires an explicit retry.
An unchanged capacity refusal does not cause model calls on each tick.

The journal records the exact queue references, model digest, processor digest, output bytes and accepted object tail.
The `AOTXAPS1` result uses schema 1 and stores the bounded UTF-8 response without a format conversion.
The processor digest identifies the exact response contract. A different processor contract is refused.

Recovery validates and applies those bytes without new appraisal generation.
If a raw journal ends inside internal work, recovery records interruption before another result can publish.
The interruption marker can replace an incomplete recorded result without treating its partial output as accepted evidence.

Complete CCIR files use runtime schema 4 and required feature bit 32 when appraisal state or history is present.
The runtime header retains its existing directory and shared-runtime profile.
The appraisal profile is at byte 188, processor SHA-256 at byte 192 and selected model SHA-256 at byte 224.
Packaged language assets must cover recorded nonzero model digests, including retained historical evidence and replay results.
The host mirror contains the same accepted state; it does not replace GPU memory authority.

The processor contract SHA-256 is `c83fd9f8ca7e8fc4f218394ac1c4480f85cf29061200a441ddd2a79d5f3f3780`.
It covers the 4511 instruction bytes in the appraisal model module, without the terminating NUL byte.
