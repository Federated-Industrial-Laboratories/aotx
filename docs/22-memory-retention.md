# Retain accepted input

`memory retain PATH` retains the last accepted input of an idle cognitive binding.
The device copies its exact text and prepared vector. It does not encode the input
again. The operator supplies IDs and policy fields, without a new text or vector payload.
All bound conversations must be idle while the common store changes.

Each accepted input creates three immutable objects: a reported event, an inferred
vector component and a reported working memory. Evidence remains unknown. Importance
does not change evidence. The binding supplies principal and scope; retention cannot
publish private input to a wider scope. Repeating retention of the same event is refused.

## Request layout

The feeder reads a bounded regular file. Its class A type-33 fragments use operation
8, with the existing 32-byte prefix and at most 160 data bytes per record. The file
has magic `AOTXRTN1`, the existing 64-byte header and 1 to 64 rows of 160 bytes.
The header must name the exact current store sequence. All integers are little-endian.

| Row offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | Bound slot |
| 4 | 4 | Focus admission: 0 or 1 |
| 8 | 16 | Conversation ID |
| 24 | 8 | Last accepted ordinal |
| 32 | 16 | Accepted request ID; also the new event ID |
| 48 | 16 | New working-memory ID |
| 64 | 16 | New vector-component ID |
| 80 | 16 | Optional superseded memory ID |
| 96 | 8 | Superseded version |
| 104 | 8 | Nonzero policy revision |
| 112 | 16 | Subject ID |
| 128 | 4 | Importance: 0 through 1,000,000, or unknown |
| 132 | 4 | Retention: ordinary 0, retained 1, pending 2 |
| 136 | 8 | Expiry sequence; zero means no expiry |
| 144 | 16 | Zero |

The maximum request is 10,304 bytes. New event, memory and component IDs must be nonzero,
distinct and new across the batch. The event ID must match the bound accepted request.
Supersession names a current working memory with the same subject and owner.
Events remain unchanged.

A nonzero expiry must exceed the sequence after the full batch. Any invalid row refuses the batch.
Supersession refuses a protected target or a lower retention value.

## Recorded result and vector format

Only the device writes operation 9, with magic `AOTXRCH1`. The header has row size
384, status at 44, 64-bit canonical tail length at 48 and zero bytes 56 through 63.

Each successful row contains the exact 160-byte request and a 32-bit focus count at 160.
Bytes 164 through 191 are zero. Eight 24-byte ID/version references start at 192.
Unused references are zero. The canonical `AOTXLOG1` tail follows all result rows.

A refusal has zero rows and no tail. The result must match one outstanding request
and its transfer ID. The device records the complete result before it publishes the
store and focus together. The existing 256-version and 1 MiB payload caps still apply.
No stored object is removed to make space.

Text payloads use `AOTXMEM1`. A device-produced vector uses `AOTXVEC2`, with a 128-byte
header and schema 2. Width, type, metric, model digest and processor digest keep their
existing offsets.

Offset 88 has the 16-byte source event ID; offset 104 has its 64-bit
version; bytes 112 through 127 are zero. The component's source fields must match this
event reference. Vector values follow the header. `AOTXVEC1` remains readable.

## Working focus and queries

Each binding has at most eight working references. Focus admission adds the new memory.
If the supersession target is already in focus, the new memory replaces it even when
focus admission is 0. This preserves the existing working reference.

At capacity, the oldest ordinary reference without protection leaves focus.
Retained and pending references do not leave focus
automatically. A full protected focus refuses the batch. Focus removal does not delete
an object or authorize paging.

Set the 32-bit live query flag at prefix offset 4 to 1 to use this working set.
Offsets 8 through 15 remain zero. The device appends working references after the
caller's explicit focus references and removes exact duplicates from the added suffix.
More than eight focus references refuses the query. Required references keep priority.
Flag 0 keeps the existing caller-supplied reference path.

External updates can make a focus reference stale or unavailable. Later queries refuse
it through the normal version and dependency checks. The audit permits an appended
focus suffix only when flag 1 is present. The original explicit focus prefix, required
references and input remain exact. Retaining an input does not create a second
conversation input in the audit file.

## Recovery

The journal contains the request, canonical mutation and complete focus result. Replay
checks their bytes and the bounded focus transition. It does not embed or search again.
Missing, changed or repeated results fail. Recovery does not need the original CCIR,
query or retention files. Model weights must remain available for the existing model
and prepared-query identity checks.

Retention is explicit. It does not add automatic extraction, appraisal, compaction,
CCIR mirroring or paging. Base conversations retain their existing behavior.
