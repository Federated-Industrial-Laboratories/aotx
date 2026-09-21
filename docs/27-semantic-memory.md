# Semantic memory intake

Set the automatic-retention field of a memory binding to `2` to enable semantic intake.
Use `memory text PATH` for GPU text preparation or `memory query PATH` for prepared vectors.
Value `0` keeps explicit retention. Value `1` keeps automatic source retention.
The model roles must include a language model and, for text requests, an embedding model.

The device recalls scoped memory and asks the resident language model to identify source
spans. It admits participant mentions, task mentions, assertions and corrections as inferred
memory. It records the result before it admits the input and memory changes together.
The normal agent then answers the input. Base conversations keep their existing path.

## Source and authority

Each candidate quotes an exact UTF-8 span in the input. The device records its source ID,
source version, byte offset and byte length. A quote must occur once in that source.
Missing or ambiguous quotes refuse the complete batch. The instruction requests complete
assertions with their negation and uncertainty. Source matching does not prove that the
model chose the correct span or candidate kind.

A participant mention does not identify an authenticated user. A task mention does not
create a trusted task ID or a required constraint. All interpretations have inferred source
kind, unknown evidence and unknown importance. The subject field stays zero.
Explicit task and participant descriptors retain their existing recall rules.

A correction can replace only a current inferred assertion from this intake format.
Imports and updates check the target at the correction's recorded sequence.
Later history does not invalidate an earlier correction. A stale target refuses the complete batch.

The old assertion must have the same owner and visibility scope. Protected assertions,
authored facts and another user's assertions cannot be replaced through this path.
The old source remains available as a dependency. The device removes the replaced optional
memory from the answer context. A required or focused reference that would become stale
refuses the complete input batch.

## Model response contract

The internal model pass uses greedy sampling, the resident chat wrap and no tools or
affect controls. It leases the input slots and releases their pages before publication.
Each internal sequence waits for its full page budget before decoding starts. Pending
sequences keep their complete requests, so partial contexts cannot fill the shared pool.
Their prompt tokens remain in the leased sequence storage while the shared tokenizer serves other work.

It does not open an ordinary conversation turn. Its temporary tokens are not replay inputs.

Each internal token must extend the JSON grammar and the source substring index.
Each token must contain structure or quoted content. Tokens that contain only space,
tab, CR or LF outside quotes are masked. These bytes remain valid within a token
that advances structure. Quoted spaces and JSON escapes for tabs and newlines remain valid.

The GPU index uses capacities derived from the complete source limit. A quote can close
only at a complete UTF-8 span that occurs once. Correction targets must be eligible.
The final parser independently checks the complete output before admission.

The response is one JSON array. Each item is `[kind, quote, target]`:

| Kind | Meaning | Target |
| --- | --- | --- |
| 1 | Participant mention | 0 |
| 2 | Task or plan mention | 0 |
| 3 | Assertion | 0 |
| 4 | Correction | One-based index of a supplied prior assertion |

An empty array means that the model proposes no candidates. Additional prose, malformed
JSON, extra fields, duplicate candidates and invalid targets refuse the batch.
The parser supports JSON string escapes and UTF-16 surrogate pairs in Unicode escapes.
The decoded quote must satisfy the source UTF-8 rules.

The output limit is 4,096 bytes. The descriptor allocation covers every item that can fit
that grammar. Reply tokens must fit the build's sequence capacity and the binding's page
limit. A deadline of 4,096 service ticks bounds generation. Resource failure and incomplete
output refuse the request; input is not truncated or partly admitted.

The 64-row profile allocates 15,863,552 bytes for source indexes and 1,489,256 bytes for
interpretation state. These GPU buffers are temporary and do not enter the cognitive file.
Object and payload capacity limits apply to the complete proposed batch, including every
source, vector, working object and interpretation. No candidate prefix is published on failure.

## Typed payload

`AOTXMEM3` contains a 96-byte header, then the exact quote:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 8 | `AOTXMEM3` |
| 8 | 4 | Schema 3 |
| 12 | 4 | Quote byte count |
| 16 | 4 | Candidate kind, 1 through 4 |
| 20 | 4 | Source byte offset |
| 24 | 32 | Language model digest |
| 56 | 32 | Extraction processor digest |
| 88 | 8 | Zero |

Generic object fields store the exact source and embedding references, owner, scope and
correction target. Participant mentions use identity objects; task mentions use cue objects.
Assertions and corrections use assertion objects. These cues do not use the mandatory
task-constraint format. Recall labels each interpretation and its exact source span.

The extraction processor digest is SHA-256 of two concatenated byte sequences:

1. The following ASCII contract line, then one LF byte.
2. The exact UTF-8 bytes of `aotx_intake_instruction` in `cuda/cognitive/intake_model.cu`, without its final NUL byte.

```
AOTX source interpretation 3; source-substring JSON token mask; each token advances structure or quoted content; exact unique UTF-8 source spans; inferred evidence; same-owner scoped inferred-assertion corrections; greedy resident decoder; no tools or affect
```

The hexadecimal digest is `0dd12d2329ab9fc0ee60591054125926749cbb58adb64a269b3b6477cba65149`.

## Decision and recovery

Recovery also accepts the previous processor digest,
`fd01a61c7c4c34a6bb64796432f5d55bf30e74c6c0f4abcea0ae3571604b61da`.
It preserves that digest in recorded metadata and inferred payloads. It does not repeat generation.
Legacy query generation uses processor 3. Submitted decisions remain refused outside recovery.

The device writes class A type 33, operation 14, with magic `AOTXICH1`. The header follows
the automatic-choice layout. Each row has 13,920 bytes:

| Row offset | Bytes | Value |
| --- | --- | --- |
| 0 | 9,168 | Existing automatic row, with the pre-write selection |
| 9,168 | 128 | Extraction metadata |
| 9,296 | 4,096 | Model response, followed by zero bytes |
| 13,392 | 528 | Effective selection used for the answer context |

Metadata has schema 1 at 0, response length at 4, model digest at 8, processor digest at
40 and interpretation count at 72. Bytes 76 through 127 are zero. For other binding modes,
metadata and response bytes are zero. Every successful row records its effective selection.
The canonical typed tail follows all rows and includes the retained source, vector,
working memory and admitted interpretations. A refusal has only a count-zero header.

Replay verifies the original request, recorded output, exact source spans, model identity,
correction targets, both selections and canonical mutation. It does not run the model,
embedding service or recall search again. The transcript audit reports admitted
interpretation counts and the effective memory selection.

Live checkpoints and the continuous CCIR memory mirror include these typed objects and
binding modes. The memory-only checkpoint profile uses external runtime assets.
The [complete text runtime profile](28-runtime-files.md) also packages the model and its required components.
See [memory checkpoints](25-memory-checkpoints.md).


## Source query mode

Use the explicit `AOTXCTX2` query extension to enable source diversity and actor labels.
Both `memory text PATH` and `memory query PATH` accept it. Shared input selects it through
recorded shared lease revision 1. Revision 0 reconstructs the old query bytes.
See [context query bytes](24-contextual-memory.md).

The owner identifies the memory scope. The actor identifies who supplied the current source.
These IDs can differ. Historical labels resolve the exact source event and its actor.
An unknown event actor is labelled `unknown`. Quoted names do not authenticate a participant.

The shared reply prompt checks its recorded query actor against the authenticated receipt.

Required references, task requirements and focus keep their exact order and meaning.
Optional selection groups rows by exact source ID and version. A prepared row without a
source event forms its own group. Each source gets a turn before further representations.

The existing cosine score and stable object-ID order choose among sources in the same round.
An eligible complete working source that fits has priority within its group.
Other representations remain available in later rounds. Distinct events with equal text remain distinct.

All references and labels pay the existing 16-reference and 4,096-byte context limits.
Appraisal groups remain complete or absent.

## Source correction targets

Source query mode supplies a separate correction table. It has at most 16 exact assertion
references and at most 4,096 rendered bytes. Each entry includes its one-based index,
exact source ID and version, source actor, and exact assertion quote.

Only current eligible inferred assertions from selected sources can enter the table.
A directly selected eligible assertion is also a candidate. Owner, scope, protection and
transitive access checks apply. Sources follow their first position in the pre-write selection.
Each source takes one turn per round. Stable object-ID order selects within each source.

The device reserves all fixed prompt bytes before it selects targets. These bytes include
the instruction, source text, actor frame, headings and selected model wrap.
The table cap is the smaller of 4,096 and the remaining prompt capacity.
A change of selected model role repeats this calculation before generation.
No recorded table is truncated. Fixed prompt bytes that exceed capacity refuse the request.

## Statement and classification calls

Source mode uses the same resident language model for two bounded calls.
The first prompt contains complete source spans and the authenticated actor. It has no prior assertions.
It places the actor before `<source>` and numbered quoted spans before `</source>`.
The actor is 32 hexadecimal digits, or `unknown` when no actor is known.

The GPU builds complete source spans before the first call. Source profile 1 uses Unicode 17.0.0 UAX29 revision 47.
It applies default sentence boundaries, then removes edge `Sp`, `CR`, `LF` and `Sep` code points from each interval.
It omits empty intervals. Every other byte remains at its original offset, including final fragments.

It does not normalize text or use locale, name, title or abbreviation dictionaries.
Default boundaries can split `Dr. Vale` into `Dr.` and `Vale`. They do not prove linguistic sentence completeness.

The source property table is separate from the tokenizer's Unicode 15 tables.
`tools/make-sentence.py` checks pinned file digests before it writes the property table and all 512 official test vectors.
The Unicode license is in `NOTICE`. Query admission still refuses CR, NUL, DEL and C1 controls.
It permits TAB and LF. Raw boundary conformance tests do not widen this input domain.

The profile digest is SHA-256 of these ASCII bytes, including the last LF:

```
aotx-source-span-profile-1
unicode=17.0.0
uax29=47
sentence-break-property-sha256=871c0c985ad95125e25b302414065a10839d068970bceb383ecec138f22a0a18
base=default-sentence-boundaries
trim-edge=Sp,CR,LF,Sep
empty-intervals=omit
text=original-utf8-bytes
offsets=original-byte-offsets
normalization=none
locale=none
fragments=retain
query-utf8=strict-scalars-no-overlong
query-controls=reject-C0-except-0009-000A-and-reject-007F-0080-009F
```

The profile digest is `af7743df1359d59a72c536c1109927ed2e04fa1ce69e76dc6374efb373c4b946`.
The first prompt renders the source once, as ordered `ordinal: "JSON-escaped exact span"` lines.
Ordinals start at 1 and remain metadata. The prompt escapes quote, backslash, TAB and LF; it preserves other admitted UTF-8 bytes.

The first response uses `[quote, "statement"]` or `[quote, "request"]` pairs.
Each label is an exact lowercase literal without escape forms.
The device maps these labels to internal kinds 3 and 0. Requests include questions and commands.

Every quote must match its complete supplied span at the exact ordered position.
Every source span needs one item, including identical text at different positions.
All raw items share the 512-item and 4,096-byte limits.
The device validates the complete response before it removes request items.
Only accepted statement spans reach classification. The first call publishes no memory.

The second prompt contains all accepted statements, their actor, and the correction table.
Its response first repeats each complete accepted span in order, including repeated text.
These required items use kind 3 or 4. Optional kind 1 and 2 quotes then come from accepted statements.

An optional quote cannot cross a statement boundary. It needs one exact occurrence in the original source.
Each optional quote can appear once per kind.

The token mask and complete parser enforce coverage independently. The parser rebuilds boundaries from the original source.
These checks do not prove that a label is correct or an assertion is true.
An empty accepted list skips the second call. Its final response is exactly `[]` and its correction table is empty.
A nonempty whitespace-only source still executes the first call, constrained to `[]`.

Each call has the existing 4,096-byte response and 4,096-tick limits.
Outside quotes, at most eight consecutive decoded space, TAB, CR or LF bytes are allowed across token boundaries.
Quoted content and structural bytes reset the count. Old queries retain their original per-token progress rule.

The slot remains owned while the first call releases its key and value pages.
The second call resets only per-call counters and scratch state.
Its resident model digest, semantic role and complete wrap table must match the captured values.
A failed call, cancellation or capacity refusal releases all held resources and refuses the whole input batch.

Before each call, output reservation uses compact escaped quote bytes E and span count K.

The first call reserves `1 + E + 17*K`; the second reserves `1 + E + 10*K` for its required items.
Zero spans reserve two bytes. A reservation over 4,096 bytes refuses the whole batch.

Actual escape forms and spaces still count against the actual response limit.

Both prompts count all exact rendered bytes against the existing 6,144-byte bound.
The second prompt reserves its fixed bytes before it selects correction targets. It never truncates a span.

Optional entry and prefixes need a remaining unique completion inside an accepted span for the selected kind.
Sparse consumed-prefix data removes emitted quotes outside vocabulary threads. Escape prefixes need a valid decoded continuation.

A token that completes an optional item cannot also include the next-item comma.
It can include trailing whitespace and the outer close. A subsequent token can start the next item.

This rule changes token admission. Complete parsing and replay check JSON bytes independently of token boundaries.

The statement processor digest covers the following two lines, including their final LF bytes.
It then covers `aotx_intake_statement_instruction` and `aotx_intake_statement_reminder`, without their NUL bytes.

```
AOTX statement processor 2
CTX2 source profile1 sha256=af7743df1359d59a72c536c1109927ed2e04fa1ce69e76dc6374efb373c4b946; Unicode17 UAX29rev47 default boundaries with edge Sp/CR/LF/Sep trim; complete ordered positional pairs [quote,label], exact lowercase unescaped statement/request labels map3/0; all source spans required including repeats; first schema2 profileID80 total84 digest88 zero120; validate all512 items before statement compaction; first compact output reserve1+E+17K<=4096; finite8 gaps; actual4096 output/ticks; actor32hex or unknown; source once as numbered JSON-escaped span lines; exact6144 prompt/wrapper capacity.
```

Its SHA-256 digest is `3e4b4380fedfa96164e0f7286c24fc8f7f7a808b36a78d483632ca346b41b8c9`.
The final processor digest starts with these two lines, including their final LF bytes.
It then covers these exact strings in `cuda/cognitive/intake_instruction.cuh`, without NUL bytes:
`aotx_intake_statement_instruction`, `aotx_intake_source_instruction`,
`aotx_intake_statement_reminder`, and `aotx_intake_source_reminder`, in that order.

```
AOTX interpretation processor 4
AOTXICH2 row18672; source profile1 sha256=af7743df1359d59a72c536c1109927ed2e04fa1ce69e76dc6374efb373c4b946; first schema2 profileID80 total84 digest88 zero120, final schema2 executed80; complete positional source pairs [quote,label] map3/0; required second spans positional as3/4 including repeats; optional1/2 globally unique original-source location inside accepted spans, once per kind, remaining completion entry/prefix/escape guard; optional completed-item token cannot include next-item separator, trailing gap/outer close allowed, parser/replay token-independent; zero-separated projection2559; finite8 gaps; first reserve1+E+17K and second reserve1+E+10A, each<=4096; actual4096 output/ticks; same resident role/model/wrapper; at most two calls; zero spans still first call; canonical empty accepted second skip; actor32hex or unknown; first numbered JSON-escaped spans only; atomic admission; source-diverse explicit targets16/4096; exact remaining6144 prompt capacity; exact recorded replay; target labels index/source_ref/source_actor/quote.
```

Its SHA-256 digest is `fa54a9e4c882459d866fc4ce2b165630020b09637d4aa4f09a52cc0a840a99d6`.
Typed inferred payloads retain the 96-byte header and this final processor digest.

## Recorded calls

Operation 14 uses `AOTXICH2` when any semantic row uses source query mode.
Its row is 18,672 bytes. The first 13,920 bytes retain the preceding automatic row layout.
The correction table occupies bytes 13,920 through 14,447 in selection-reference format.

Its schema is 1, its count is at offset 4, and exact 32-byte references start at offset 16.
Unused table bytes are zero. Each used reference has flag 1 and four final zero bytes.

The next 128 bytes hold first-call metadata. The final 4,096 bytes hold its exact response and a zero tail.
Both metadata blocks use schema 2.

Both metadata blocks have response length at 4, model digest at 8 and processor digest at 40.
Parsed item count is at 72. Selected semantic model role is at 76.
First-call item count includes only accepted statements. Its raw response retains both labels.

The role is a portable 32-bit value. Both model digests and roles must agree.
The first block has profile ID at 80, total source-span count at 84, and profile digest at 88 through 119.
Bytes 120 through 127 are zero. Accepted count cannot exceed total count.
Replay recomputes total count and requires profile ID 1 with the exact profile digest.

The final block records second-call execution at 80: 1 for execution or 0 for an empty-first skip.
Final bytes 84 through 127 are zero.

Old rows retain their first 13,920 bytes and have zero extra fields.
Their processor 2 or 3 target indices still refer to the pre-write selection.
Manual rows retain their zero metadata convention.

Replay checks both processor identities, model role and model digest.
It rebuilds source spans before prompt capacity checks, then parses complete first-output coverage without generation.
It checks the empty skip or rebuilds the second-call target table at the original cut.
The same prompt renderer supplies the exact remaining capacity calculation.
Replay then checks whole-statement coverage, optional span containment, both selections and the complete canonical mutation.
An altered output, stage flag, role, digest, target table or capacity-dependent result refuses replay.

Complete runtime files retain both outputs through their existing journal sections.
Memory-only checkpoints retain canonical state, exact source references and final processor provenance.
They do not archive all historical raw model outputs. Their restore does not repeat interpretation.

`source_recall`, `intake_source`, `intake_stages` and `shared_source` check these boundaries at N=1 and N=64.
Run `source_boot_test.py BUILD SOURCE STORE OUTPUT 1|64` for the separate actual-model check.
Use `--operation-seconds SECONDS` to set its explicit wait limit. The default is 180 seconds.
It saves raw model decisions and tests facts, corrections, unrelated claims, uncertainty,
scoped recall, copied-file continuation and journal replay.

Run `source_matrix_boot_test.py BUILD SOURCE STORE OUTPUT 1|64` to check source sentence labels.
The 23 cases include questions, direct requests, conditional requests, quoted commands and statements.

Use `--case INDEX` to select the first case. N=64 repeats the cases with distinct source text.
The test saves both raw outputs and checks exact labels. Responses have a 32-token limit.
It also accepts `--operation-seconds SECONDS`.

## Qualified automatic memory

Automatic interpretation requires an exact qualified model and processor pair.
The device also checks both processor digests, the source profile and every wrapper field.
Ordinary model usability does not establish this capability.
The qualification table in the device module has no operator override.
An unknown or changed pair returns status 12 before a semantic update can publish.
The device checks qualification before either call and before publication.

Current qualification does not change validated historical decisions during replay.

Direct semantic-retention requests refuse when their pair is unavailable.
Shared conversation input then uses source-text retention and reports memory availability through the API.
It does not silently interpret input with a different model.
No legacy single-call processor is qualified in the supplied table.

`intake_capability` checks exact admission, identity mismatches, atomic refusal and replay at N=1 and N=64.
Structural fixtures use explicit synthetic qualification rows with fixed responses.
Actual-model acceptance retains the supplied product qualification table and runs the complete native workflow.

Run `memory_capability_boot_test.py BUILD SOURCE STORE OUTPUT 1|64` to check the native HTTP boundary.
The default expects unavailable automatic memory and preserved source text.
Use `--qualified` for an accepted pair, or `--wrapper-mismatch` to check a changed wrapper.
The test checks both discovery routes, ordinary replies, scope and copied-file recovery.
It also checks assertion retention and later recall when qualification is expected.
