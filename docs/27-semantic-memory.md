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

The 64-row profile allocates 15,863,552 bytes for source indexes and 1,450,344 bytes for
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
AOTX source interpretation 2; source-substring JSON token mask; exact unique UTF-8 source spans; inferred evidence; same-owner scoped inferred-assertion corrections; greedy resident decoder; no tools or affect
```

The hexadecimal digest is `fd01a61c7c4c34a6bb64796432f5d55bf30e74c6c0f4abcea0ae3571604b61da`.

## Decision and recovery

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
binding modes. The model and its other runtime assets remain external under the current
checkpoint profile. See [memory checkpoints](25-memory-checkpoints.md).
