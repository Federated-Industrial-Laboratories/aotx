# Prepared memory and recorded recall

`aotx_ccir_recall` selects prepared text on the GPU.
The command records each request and its exact selected object versions in a new CCIR file.
It then prints one JSON object per request.
Replay reads that file and reconstructs the selected context without vector search.
The command does not load a language model or change a base conversation.

Build the `aotx_ccir_recall` target through the normal CUDA build.
Use these commands:

```text
aotx_ccir_recall select INPUT REQUESTS OUTPUT
aotx_ccir_recall replay INPUT
```

`INPUT` is a typed CCIR file described in [Typed state](18-typed-state.md).
`REQUESTS` is a binary request file described below.
`OUTPUT` must not exist.
The source file remains open under a shared lease until output completes.
The command folds the source tail into a checkpoint and preserves unknown optional sections.
It prints selected context only after the new file is durable.

The request path must name a regular file.
A pipe is refused without waiting for a writer.

An output error returns a nonzero status; an IO error can follow file publication.
Inspect the output file before a retry after an IO error.

The exit status is 0 for success, 1 for an operation error, or 2 for invalid command arguments.
Errors go to standard error; context rows go to standard output.
Each JSON row has `request`, `selection`, `cut`, `count`, `searches` and `context` fields.
IDs use 32 lowercase hexadecimal digits in byte order.
The context retains exact UTF-8 text; JSON escaping preserves quotes, backslashes, tabs and newlines.
Replay reports zero searches.

Replay needs only the exported file; the original source and request paths are not required.
`searches` is 1 for each successful selection row and 0 for each replay row.
It counts selection operations, not vector candidates.

The context repeats these first two lines for each selected object, then appends the current input:

```text
[memory id=<32 hex digits> version=<version> source=<source> evidence=<evidence> reason=<reason>]
<exact memory text>
[input]
<exact input text>
```

Each memory text ends with one added newline.
The command adds no newline after the input text.
Source and evidence numbers use the typed state values in [Typed state](18-typed-state.md).
Reason 1 means required, 2 means focus, and 3 means semantic recall.

## Prepared payloads

All integers use little-endian byte order.
Text belongs to an EVENT, ASSERTION, CUE, INTENTION, WORKING or IDENTITY object.
Its payload starts with a 32-byte header: `AOTXMEM1`, schema uint32 1 at byte 8, and text length uint32 at byte 12.
Bytes 16 through 31 are zero.

The header is followed by 1 through 2048 exact UTF-8 bytes.
NUL and controls other than tab and newline are refused.
Other opaque payloads do not become recall candidates.

An optional embedding reference names an earlier COMPONENT object.
The component payload has a 128-byte header followed by a vector of little-endian F32 values.
The header starts with `AOTXVEC1`.
Schema uint32 1 is at byte 8; width uint32 is at byte 12.
Dtype uint32 4 is at byte 16; cosine metric uint32 1 is at byte 20.
Model, processor and source digests occupy bytes 24, 56 and 88; each has 32 bytes.

Bytes 120 through 127 are zero.
Width is 1 through 1024; values must be finite and the vector must be nonzero.
The query model and processor digests must match the component exactly.

The vector space filter also requires the same width.
Cosine recall accumulates dot products and squared norms in double precision on the GPU.
Eligible vectors from other spaces are ignored when compatible candidates are available.
If all eligible vectors use incompatible spaces, recall refuses with `AOTX_COG_SOURCE` (7).

These digests describe the prepared vector space; they do not authenticate a publisher.
The command does not calculate embeddings from text.

## Request file

The 64-byte header starts with `AOTXREQ1`.
Count uint32 is at byte 8 and must be 1 through 64.
Schema uint32 1 is at byte 12; lineage occupies bytes 16 through 31.
The exact current sequence uint64 is at byte 32; row bytes uint32 8192 is at byte 40.
Bytes 44 through 63 are zero.
The file contains exactly the header and `count` rows.

Each 8192-byte row has these fields:

| Byte | Field | Size |
| --- | --- | --- |
| 0 | Request event ID | 16 bytes |
| 16 | Principal ID | 16 bytes |
| 32 | Room ID | 16 bytes |
| 48 | Selection ID | 16 bytes |
| 64 | Model digest | 32 bytes |
| 96 | Processor digest | 32 bytes |
| 128 | Vector width | uint32 |
| 132 | Selection limit | uint32 |
| 136 | Memory context budget | uint32 |
| 140 | Required reference count | uint32 |
| 144 | Focus reference count | uint32 |
| 148 | Input text length | uint32 |
| 152 | Scope | uint32 |
| 156 | Reserved zero | 4 bytes |
| 160 | Query vector | 1024 F32 slots |
| 4256 | Required references | 8 rows of 24 bytes |
| 4448 | Focus references | 8 rows of 24 bytes |
| 4640 | Input text | 2048 bytes |
| 6688 | Reserved zero | 1504 bytes |

Each reference contains a 16-byte object ID and uint64 version.
Unused references, vector slots and input bytes are zero.
Request and selection IDs are nonzero and distinct across the batch.
Scope selects one private, room or instance domain under the typed state rules.
The caller supplies authorized principal and room bindings; this offline command is not an authentication service.

The selected domain can also use instance-visible sources where the typed derivation rules permit them.
Private and room sources do not mix within one persisted selection.

Required references precede focus references and cosine recall; duplicate references occur only once in the result.
Selection limit is 1 through 16 objects.
Memory budget is 1 through 4096 bytes, including annotations.
A required or focus object that cannot fit refuses the request.
An optional candidate that cannot fit is skipped.

Equal scores use ascending object ID and then version.
Each context has a separate bounded buffer; current input follows the memory text.
Source and evidence labels remain visible in the context.

## State and limits

All request records and selections enter one device transaction.
A failed request or capacity check leaves the original store and source file unchanged.
The store holds at most 256 immutable object versions and 1 MiB of payload.
Each request adds two objects and retains its full request row.
The sequence advances by twice the request count; the tick advances by one.
No eviction or implicit disk offload occurs.

Each request batch contains at most 64 rows.
The CCIR file also permits at most 64 current recorded request and selection pairs in total.
A batch that exceeds this total is refused before publication.
The command does not remove saved pairs to make room.

Each authored request EVENT has an 8208-byte payload.
Bytes 0 through 7 contain `AOTXQUE1`; the original sequence cut uint64 starts at byte 8.
The exact 8192-byte request row starts at byte 16.
Its inferred SELECTION names that event as its source and stores the exact ordered selected IDs and versions.
The selection payload uses the existing 16-byte header and 32-byte reference rows described in [Typed state](18-typed-state.md).

Before publication, the device checks selected objects and all referenced dependencies at the resulting sequence cut.
A dependency that expires during the sequence advance refuses the complete batch.
An explicit cut can equal `UINT64_MAX`; expiry checks use that value without substituting the prior sequence.
An expiry equal to that cut refuses publication; an expiry of zero still means no expiry.

Replay uses recorded IDs and versions, not a new ranking.
It checks current scope, expiry, supersession and source restrictions before context output.
Withdrawn or stale selected facts refuse replay.
The command does not replace them with newer facts.
Live conversation activation, automatic constraint discovery, hosted mirrors and native modules are separate functions.
