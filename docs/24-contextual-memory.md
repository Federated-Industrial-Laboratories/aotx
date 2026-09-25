# Contextual memory and appraisal

Prepared recall can select task requirements and use source-linked appraisals to rank optional memories.
The GPU applies these controls in offline CCIR recall and live `memory query` or `memory text` requests.
They also work with automatic input retention. Affect can be enabled or disabled.
Queries with a zero extension keep the existing recall policy. Unbound conversations keep the base conversation path.

The caller supplies task and participant IDs. Typed load and update operations admit claims, appraisals and corrections.
The system does not infer these IDs or assessments from natural language.
Automatic retention stores exact input; it does not extract facts or measure benefit and harm.

## Task and participant extension

The [prepared query](19-prepared-memory.md#request-file) remains 8,192 bytes, with schema 1.
Its last 1,504 bytes start at offset 6,688. All-zero bytes disable the extension.
Otherwise, this region has the following layout. Offsets are relative to the extension.
Integers use unsigned little-endian encoding.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | Magic `AOTXCTX1` |
| 8 | 4 | Schema 1 |
| 12 | 4 | Flags: 1 for task requirements, 2 for appraisal ranking |
| 16 | 16 | Task cue ID |
| 32 | 4 | Participant count, 0 through 64 |
| 36 | 4 | Minimum cosine relevance in millionths |
| 40 | 4 | Appraisal boost in millionths |
| 44 | 4 | Policy revision 1 |
| 48 | 1,024 | Up to 64 participant IDs, 16 bytes each |
| 1072 | 432 | Zero |

Set flags to 3 to use both controls. Unknown flags or revisions refuse the query.
Flag 1 requires a nonzero task cue ID. Participant IDs are nonzero and distinct; unused slots are zero.

Without flag 1, the task ID, count and participant slots are zero.
Count zero means unknown participants. It does not mean every known participant.

Both numeric controls accept 0 through 1,000,000. Without flag 2, both are zero.
The floor applies before a boost. A zero boost keeps cosine order and selects no appraisal records.
These controls are software policy values, not probabilities or biological measurements.

## Contextual text

An assertion, cue or intention can use an `AOTXMEM2` payload:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | Magic `AOTXMEM2` |
| 8 | 4 | Schema 2 |
| 12 | 4 | UTF-8 text byte count, 1 through 2,048 |
| 16 | 16 | Nonzero task cue ID |
| 32 | 4 | 1 for required memory; 0 for optional memory |
| 36 | 28 | Zero |
| 64 | Text count | Exact UTF-8 text |

The object requires an exact source reference. Assertions and intentions require a nonzero subject ID.
A general cue can have a zero subject. Store admission validates recognized payloads before publication.
The generic [object layout](18-typed-state.md) and CCIR directory layout retain their existing schemas.

A contextual object is eligible only when flag 1 is enabled and the task cue matches.
Its subject must be zero or one of the supplied participants. An intention also requires retention value 2, pending.
Set its retention to 0 in a valid new version when the intention is complete.
Explicit references and working focus cannot override applicability, visibility or source access.

For example, a member's meal requirement can name that member and a meal task cue.
A new member receives no other member's requirement. A general meal cue can request missing requirements.
The caller must identify the actual participants; the extension does not authenticate an identity.

## Required selection and optional appraisal

Selection order is explicit required references, applicable contextual requirements, working focus, then optional memories.
Contextual requirements use ascending object ID order. Duplicate references occur only once.
Requirements do not need embeddings. The scan covers the configured store capacity without a separate cue table.

All required and focus references must fit the caller's object and byte budgets.
If they cannot fit, the entire query refuses. The existing limits remain 16 selected objects and 4,096 memory context bytes.
These are per-query context limits; they do not limit the stored memory to 16 objects.
Task and participant annotations use the same byte budget.

Automatic recall uses the complete input as search text. Reply format instructions can affect the selected memories.
A topic name does not require its source to be selected. Prepared queries can require exact source references.

With flag 2, optional memories first meet the cosine floor. A current eligible appraisal can then increase the score.
The appraisal must share the memory's subject and refer to its exact ID and version.
For working memory, it can instead refer to that memory's exact source event.
Source and appraisal visibility, evidence and current-version checks still apply.

The policy uses the largest known value of benefit and harm. Unknown values remain distinct from known zero.
If neither value is known, or the largest is zero, the appraisal adds no priority.
Where several appraisals qualify, use the largest intensity; ties use ascending appraisal ID.
Duplicate appraisals cannot accumulate priority through addition.

```
score = cosine + (boost / 1000000) * (intensity / 1000000)
```

Benefit and harm never cancel. Ranking does not raise evidence, confidence, permission or source quality.
The selected memory and its exact appraisal must fit together. Otherwise, skip that optional candidate.
An appraisal already selected for the same source can support another selected working memory without another row.

Context includes the separate appraisal values, units, subject and exact assessed source.
Selection reasons are 1 required reference, 2 focus, 3 cosine, 4 contextual requirement, 5 appraisal and 6 significant memory.
The existing 32-byte selection rows retain their layout. Validated query and reference structure determine the reasons.

## Corrections and recovery

Use a new version when the source is unchanged. For new evidence, retain its source event and create a new claim ID.
Use an exact supersession reference to replace the old claim. Current recall excludes the superseded claim.
Corrections preserve source history. A withdrawn source also removes its dependent claims and appraisals from eligible recall.

Recorded decisions preserve the complete query extension and exact selected object versions.
Replay checks applicability, the complete required set and appraisal support without repeating similarity search.
Changed or missing required references, obsolete claims and unrelated appraisals refuse recovery.
Repeated recall does not write appraisals or strengthen stored evidence.

Automatic admission rechecks dependencies at the resulting store sequence.
An appraisal that expires at that cut refuses the complete input before memory, focus or messages are published.
Exact journal recovery retains the query controls, corrections, prepared vectors and selection audit.
Model files must remain available. Continuous CCIR mirroring and memory offload remain separate functions.

Completed idle cognitive conversations release their language cache pages after pending tool and quality consumers finish.
Sequence text, reply metadata and persistent memory remain available. Base conversations keep their existing cache policy.
Partly served language page requests can retry when the queue is served and physical pages become available.
Outstanding requests and a full physical pool do not produce repeated asks each tick.

## Resource cost and checks

Recall scratch uses 16 bytes per configured object slot per query, with up to 64 distinct query rows.
The appraisal index adds 2,097,152 GPU bytes at the default 8,192-object capacity.
No implicit paging, eviction or storage cap change occurs. See [memory capacity](06-build.md#memory-capacity).
Optional appraisal lookup scans eligible store objects for each candidate; full-store lookup costs remain significant.

`context_memory` checks task and subject applicability, ranking, corrections, pressure and exact offline replay.
`context_live` checks live prepared and text input, automatic retention, complete refusal and recorded recovery.
`context_cache` checks completed cache release and progress after partial page allocation.
All three tests use distinct batches of 1 and 64. Run the real file and model workflow with:

```
python3 tests/context_boot_test.py BUILD SOURCE STORE OUTPUT vector-1
python3 tests/context_boot_test.py BUILD SOURCE STORE OUTPUT vector-64
python3 tests/context_boot_test.py BUILD SOURCE STORE OUTPUT text-1
```

Use a new output directory for each command. The driver removes its original inputs before cold recovery.
It checks known requirements, an unknown participant, a corrected requirement and fresh input after restore.


## Source query extension

`AOTXCTX2` opts into source-diverse recall and exact actor labels.
It keeps the preceding task and appraisal fields at their existing offsets.
The extension magic is `AOTXCTX2`, its schema at offset 8 is 2, and its policy at offset 44 is 2.
Flags can be zero.

The 16 bytes at extension offset 1,072 hold the current source actor.
An all-zero actor means unknown. All bytes after extension offset 1,088 are zero.
The extension starts at query offset 6,688, so the actor starts at query offset 7,760.

The actor is separate from the query owner, task subject and participant list.
Task and appraisal defaults preserve the actor and extension version.
Zero extensions and `AOTXCTX1` keep their existing selection and rendering rules.
See [source memory and correction targets](27-semantic-memory.md).

`AOTXCTX3` keeps the same fields and uses schema 3 and rendering policy 3.
It permits a selected working record to refer to its exact selected source body.
The source ID, version and complete payload must match; all selected references remain.
Bodies of 20 bytes or fewer remain unchanged.

The 16-reference and 4096-byte bounds still apply to the complete selection.
Revision 3 also places the historical-data rule in the model system frame.
Previous formats retain their exact rendering. Unknown versions and policies are refused.

## Shared source labels

Use magic `AOTXCTX4`, schema 4 and policy revision 4 to share repeated source labels.
The extension retains the source actor field and all task and appraisal controls.
The GPU assigns a group to each exact source ID and version used by two or more selected records.
Groups have consecutive numbers in first occurrence order.

Each group has one header: `[source_group=N source_ref=ID@VERSION source_actor=ACTOR]`.
Its memory headers use `source_group=N` in place of the repeated source and actor fields.
A source used by only one record retains its full inline labels. An unknown actor remains `unknown`.
Different source IDs or versions never share a group because their text or actors match.

Every selected object ID, version, reason, subject, assessment and exact quote remains present.
Exact duplicate working bodies retain the `text: see source_ref` rule of revision 3.
The table and every memory row count toward the same 4,096-byte limit.
Complete evidence groups enter together after their full rendered size fits.
Required references, focus order, cosine ranking and the 16-reference limit do not change.

The table exists only in the context; it adds no stored object or selection reference.
Earlier query revisions retain their exact context and prompt bytes during recovery.
