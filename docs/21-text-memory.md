# Text memory requests

`memory text PATH` sends a batch of text requests to the live cognitive store. Load
both model roles with `--roles language,embedding`. The device prepares query vectors
with the loaded embedding model, selects memory and sends the context to the language
model. The disk reader transports bytes and writes audit files.

Each input has a limit of 192 UTF-8 bytes. Longer input is refused without truncation.
This limit belongs to the current embedding service. Prepared `memory query` requests
keep their 2,048-byte input limit. A text request still names its binding, IDs, ordinal,
store cut, scope, memory budget and optional required or focus references.

## Input bytes

Use the query envelope in [live memory](20-live-memory.md), with magic `AOTXTXT1`.
The header is 64 bytes and each row is 8,256 bytes. A batch has 1 to 64 rows.
In the 8,192-byte query part, these fields must be zero:

| Query offset | Bytes | Field supplied by the device |
| --- | --- | --- |
| 64 | 32 | Loaded embedding model digest |
| 96 | 32 | Processor digest |
| 128 | 4 | Vector width |
| 160 | 4,096 | Vector slots |

All other query fields retain their existing meaning. The device supplies the model
digest, fixed processor digest, width and vector. The processor identity describes the
current tokenization, pooling and normalization path. It does not state the quality of
a model's semantic representation.

The processor digest is SHA-256 of this exact ASCII line, without a line ending:

```
AOTX text embedding 1; exact UTF-8 1..192 bytes; model GGUF vocabulary; clean/pretok/merge/gather; all tokens from position zero; final row RMS output norm F32; L2 F32; cosine query F32; no instruction prefix
```

The digest in hexadecimal is:

```
7d12af1d2cd1e5194def983d1fd8073d1c36c444eea39c2dcf9bbe394e75892d
```

The feeder uses class A type 33, operation 6. Its regular-file checks, random transfer
ID, exact byte transport and 160-byte fragments match the prepared-query path. Both
standard input and attached input accept the command. Disk framing checks do not encode
text or interpret the cognitive fields.

## Recorded decision

Only the device writes operation 7, with magic `AOTXTCH1`. Its header uses the same
status and count fields as a prepared choice. Each successful row has 8,784 bytes:

| Row offset | Bytes | Value |
| --- | --- | --- |
| 0 | 64 | Exact live request prefix |
| 64 | 8,192 | Complete prepared query |
| 8,256 | 528 | Exact ordered selection |

The maximum decision is 562,240 bytes. A refusal has a 64-byte header, count zero
and status 1 through 11. It does not advance the ordinal or change the previous bound
context. A text decision must match an outstanding text request and its transfer ID.
A prepared choice cannot complete a text request, or the reverse.

The device records at most 64 fragments per tick. Prompt admission waits for the
complete recorded decision. Embedding uses bounded temporary pages and a deadline of
128 service ticks. Missing or incompatible model state, invalid input and resource failure
refuse the full batch without a partial prompt.
New text requests also refuse while a model load is pending.

An unserved page queue can delay cleanup. Prompt publication waits for queued page releases.

## Recovery and audit

Replay uses the recorded prepared query and selection. It does not tokenize, embed or
search again. The embedding weight identity and processor identity must remain valid.
The four device-supplied field ranges can differ from the original text request.

With live prefix flag 1, the device can also append working references after the exact
explicit focus prefix. See [Retain accepted input](22-memory-retention.md) for that rule.
Wrong request types, changed input, malformed vectors and incomplete decisions refuse
recovery of the affected work.

The audit pairs each original text batch with its complete matching decision. It writes
the original input, IDs, scope, ordinal and selected object versions to the existing
per-agent transcript. Bad framing, changed input or a partial decision cannot create accepted input.
Recorded vectors remain journal bytes; the audit does not construct a CPU memory index.

The interface prepares queries. `memory retain PATH` can retain the last accepted input
and its prepared vector. Other memory changes require explicit typed state operations.
Base conversations keep their existing input path.
