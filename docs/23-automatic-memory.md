# Automatic input retention

A cognitive binding can retain each accepted input without a separate `memory retain`
command. Set the 32-bit value at bind row offset 60 to 1. Value 0 keeps explicit
retention. Other values refuse the complete bind batch. The setting lasts for the binding.

Both `memory query PATH` and `memory text PATH` support this setting. A batch can mix
bindings with automatic and explicit retention. Unbound conversations keep the base input path.
Use the [live binding layout](20-live-memory.md#byte-layout) and the same regular-file commands.
Text requests still require both language and embedding model roles.

## Admission and memory

Each automatic input creates three immutable objects: an event, a vector component and
a working memory. The device retains the exact input text and prepared vector.
The binding supplies the principal, room and scope. No input gets a wider scope.

The event ID is the request ID and must be new. The device assigns working and component
IDs with eight bytes `AOTXGEN1` followed by an unsigned little-endian 64-bit number.
Allocation starts after the current store sequence. It skips stored IDs and every request ID
in the batch. Exhaustion refuses the batch.

Evidence and importance remain unknown. Retention is ordinary; expiry and supersession
are absent. The subject is the bound principal. Policy revision 1 names this rule.
Automatic retention does not extract facts or learn importance from the input.

The new working reference enters the binding's [working focus](22-memory-retention.md#working-focus-and-queries).
At eight references, the oldest ordinary reference without protection leaves focus.
A full focus containing only protected, retained or pending references refuses the input.
Focus replacement does not delete stored history.

Set the query prefix flag at offset 4 to 1 to recall this focus. Caller limits still apply.
The selected-object limit must allow all required and focus references after duplicates are removed.
The memory byte budget must also fit those references. Automatic retention does not raise either limit.

If any row has automatic retention, all bound conversations must be idle for admission.
Recall searches the store before the write. The current input cannot recall itself.
The device checks selected dependencies again at the resulting store sequence.
An expired dependency refuses admission with status 11 because it is no longer visible.

Object or byte pressure, an occupied event ID, unavailable focus, invalid input or encoder
failure refuses the complete batch. Refusal queues no input and changes no binding, focus
or stored memory. It does not consume an ordinal. Retry with the next ordinal and current cut.

A successful batch advances the store sequence by three per automatically retained row.
Subsequent inputs must name that resulting cut. No stored object is removed or sent to disk
to make space. Configure larger [memory capacity](06-build.md#memory-capacity) when needed.

The device records the whole decision before publishing memory, focus and queued input together.
Later tokenization or generation failure does not erase an already accepted input.
Input admission and generation completion are separate outcomes.

## Combined decision bytes

Only the device writes class A type 33, operation 10. The transfer ID matches the input.
Its 64-byte header contains:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | Magic `AOTXACH1` |
| 8 | 4 | Input row count |
| 12 | 4 | Schema 1 |
| 16 | 16 | Store lineage |
| 32 | 8 | Store sequence before the write |
| 40 | 4 | Row size: 9,168 |
| 44 | 4 | Status: zero for success |
| 48 | 8 | Canonical tail byte count |
| 56 | 8 | Zero |

Each successful input row contains:

| Row offset | Bytes | Field |
| --- | --- | --- |
| 0 | 64 | Exact original live prefix |
| 64 | 8,192 | Complete prepared query |
| 8,256 | 528 | Exact ordered selection |
| 8,784 | 384 | Retained result row |

The retained result has the layout in [explicit retention](22-memory-retention.md#recorded-result-and-vector-format).
A binding with explicit retention has 384 zero bytes here. The canonical `AOTXLOG1` tail
follows all input rows, with three objects per automatic row in input order.
A refusal has a 64-byte header, status 1 through 11, count zero and no tail.

The result capacity is `image_bytes + 586816`. The device writes at most 64 fragments
per tick. Neither a partial result nor a refusal publishes part of the batch.

## Recovery and audit

Replay validates the original input, prepared query and selected references. It reconstructs
the canonical mutation and compares the complete decision bytes before publication.
It does not encode, search or create another automatic write. Missing, changed, unrelated
or repeated results refuse recovery. The binding setting and exact focus are restored.

The journal permits recovery after the source container and input files are removed.
Required model files must remain available. Each accepted input has one audit input line
and one selection line. The selection includes `retained ID@1` for automatic rows.
Refusal records its status without an accepted input line.

The journal remains the recovery file. Continuous CCIR mirroring, portable active-conversation
export, memory reclamation and disk offload are separate functions.

Binding value 2 adds [semantic intake](27-semantic-memory.md) to the same atomic input path.
