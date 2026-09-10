# Typed state files

The typed state module is separate from the base conversation runtime.
It stores admitted objects in GPU memory and exports an explicit checkpoint to a CCIR file.
It does not load a language model, encode media, rank memories, or start a background process.
The base data-state profile uses schema 1. Optional [memory maintenance](26-memory-maintenance.md) selects schema 2.
Neither profile is a complete runtime package.

Use `aotx_ccir_state INPUT OUTPUT` to restore, replay and export a checkpoint on the GPU.
The output path must not exist. Use `CUDA_VISIBLE_DEVICES` to select the device.

The command retains optional file sections and removes the applied tail from the new directory.
The source file remains intact. `fallback=1` reports that one nonempty root or its generation failed validation.
See [CCIR files](17-ccir.md) for the file commands and transaction rules.

## Configured capacity

The default build holds 8,192 immutable object versions and 16 MiB of payload in each store.
Set `AOTX_MEMORY_OBJECTS` and `AOTX_MEMORY_BYTES` during CMake configuration to change these bounds.
Rebuild the complete runtime and its disk programs together. Running instances do not resize.
See [Build](06-build.md#memory-capacity) for the options and allocation costs.

Use `aotx_ccir_state --limits` to inspect the compiled bounds without a GPU.
The output is one line with decimal byte counts:

```
objects=8192 payload_bytes=16777216 image_bytes=18874496
```

Image capacity is `128 + objects * 256 + payload_bytes`. Schema 1 and the 256-byte
object row do not depend on the configured capacity. A larger build accepts smaller
admitted images unchanged. A smaller build refuses images that exceed either bound.
The API admits a whole batch or preserves the previous state. Pressure does not evict
objects or enable disk offload.

## Device interface

`cuda/cognitive/state.cuh` declares restore, apply, checkpoint and resolve kernels.
The caller owns separate live and staging stores, input/output buffers, and result buffers.
Initialize the live store to zero before the first restore.

Serialize state operations and all readers. Launch state operations as one block of 64 threads.
Wait for completion before reading results, reusing buffers, or publishing a file checkpoint.
Input, output, live, staging and result buffers must not overlap.

Only successful restore and apply operations can supply a store to checkpoint or resolve.

Each caller allocates the complete configured store and its required scratch buffers.
Payload capacity alone is not the full GPU memory cost.

Restore validates the full state in staging memory before publication.
The caller must provide an authenticated principal and authorized room to resolve.
The module checks scope. It does not authenticate a network client or parse a grant.

Resolve takes a batch of object IDs, exact versions, principals and rooms.
It checks the current version, scope, expiry and tombstone for each object.

It also checks current visibility through source, supersession, embedding and selection references.
A recorded selection fails if any selected version is stale or inaccessible.
A supersession record also makes its exact target version stale. It does not erase the target bytes.
Denied and stale results return no object index or version.

Historical records remain stored after a new revision. A failed current lookup does not erase them.

## State image

All integers are unsigned and little endian. No native structure is a file layout.
The 128-byte image header has these fields.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `AOTXOBJ1` for a checkpoint; `AOTXLOG1` for a tail |
| 8 | 4 | Schema, 1 |
| 12 | 4 | Header bytes, 128 |
| 16 | 4 | Object record bytes, 256 |
| 20 | 4 | Object count |
| 24 | 8 | Payload arena bytes |
| 32 | 8 | Checkpoint covered sequence, or tail first sequence |
| 40 | 8 | Recorded source tick |
| 48 | 16 | Nonzero lineage ID |
| 64 | 8 | Object offset, 128 |
| 72 | 8 | Payload offset, 128 + count * 256 |
| 80 | 8 | Exact total image bytes |
| 88 | 4 | Data-state mode, 1 |
| 92 | 36 | Reserved, zero |

A tail contains consecutive update sequences in record order.
New sequences must follow the live cut without a gap. A new tail tick cannot precede the live tick.

Covered records must match the stored record and payload bytes, except for the arena offset.
A matching covered prefix is skipped. A conflicting retry fails the whole batch.

Replay installs recorded bytes. It does not run inference, retrieval, tools or URL requests.

## Object record

`cuda/cognitive/format.h` defines all field offsets and enum values.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 2 | Schema, 1 |
| 2 | 2 | Kind |
| 4 | 4 | Flags: tombstone 1, protected 2 |
| 8 | 16 | Nonzero object ID |
| 24 | 16 | Lineage ID |
| 40 | 8 | Version, from 1 |
| 48 | 8 | Creation sequence |
| 56 | 8 | Update sequence |
| 64 | 16 | Nonzero owner principal |
| 80 | 16 | Room ID; nonzero only for room scope |
| 96 | 16 | Source object ID |
| 112 | 8 | Source version |
| 120 | 16 | Subject entity ID |
| 136 | 16 | Superseded object ID |
| 152 | 8 | Superseded version |
| 160 | 8 | Offset within the payload arena |
| 168 | 8 | Payload bytes |
| 176 | 4 | Scope: private 0, room 1, instance 2 |
| 180 | 4 | Source: authored 1, observed 2, reported 3, inferred 4 |
| 184 | 4 | Evidence: unknown 0, supported 1, disputed 2, withdrawn 3 |
| 188 | 4 | Retention: ordinary 0, retained 1, pending 2 |
| 192 | 4 | Importance, scaled value or unknown |
| 196 | 4 | Reserved, zero |
| 200 | 8 | Expiry sequence; zero disables expiry |
| 208 | 16 | Embedding component object ID |
| 224 | 8 | Embedding component version |
| 232 | 8 | Nonzero policy revision |
| 240 | 16 | Reserved, zero |

Kinds are event, assertion, appraisal, relationship, cue, intention, working state, media,
component, selection, policy and identity, numbered 1 through 12 in that order.
The common rules apply to each kind. Specialized payload rules apply to appraisal, media and selection.
Other payloads are opaque nonempty bytes in this profile. They do not activate a policy or native module.

References pair a nonzero ID with an exact version. An absent reference uses zero for both fields.
Each reference must resolve to a prior update. Cyclic sources cannot pass admission.
Supersession retains the target kind and subject.

Inferred records require a source. A child of an inferred source must also be inferred.
Revisions preserve kind, owner, subject, creation sequence and original source fields.

Schema 1 versions are consecutive. Schema 2 new versions equal their update sequence.
Event, media and component content cannot change through a revision.
These kinds permit a tombstone revision. A tombstone has zero payload bytes and offset.

Protected records cannot lose protection or receive a tombstone. Tombstones cannot be revived.

A child cannot widen source visibility. Revisions can narrow scope, but cannot widen it.

Private derivations retain the source owner. Room derivations retain the source room and scope.
Publishing private learned state requires a separate authorized operation beyond this profile.

Payload extents cannot overlap. Their combined size equals the payload arena size.
Without maintenance, the store retains all revision history. Schema 2 permits explicit retained roots and version reclamation.

## Appraisal payload

An appraisal uses eight uint32 fields in 32 bytes:
schema, benefit, harm, arousal, consequence category, confidence, units revision, and reserved zero.

Schema and units revision are 1. Consequence is unknown 0 or an ordered category from 1 through 4.
Benefit, harm, arousal, confidence and object importance use the range 0 through 1000000.
UINT32_MAX means unknown. Unknown is distinct from zero.

Benefit and harm remain independent. Confidence does not imply a calibrated probability.
An appraisal requires a nonzero subject equal to its source subject.

The profile stores these values; it does not extract them from language.
Optional [contextual recall](24-contextual-memory.md) uses exact source-linked appraisals to rank relevant memories.

## Media payload

The payload starts with a 192-byte descriptor, then data bytes, then uint64 positions.
The descriptor binds caller-supplied source, model and processor SHA-256 identities.
Those identity fields are provenance declarations. Admission checks their presence, not their authenticity.
The container digest separately checks the stored bytes for corruption.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | Schema, 1 |
| 4 | 4 | Modality: image 1, audio 2, features 3 |
| 8 | 4 | Representation: source bytes 1, exact features 2 |
| 12 | 4 | Dtype: U8 1, I16 2, F16 3, F32 4 |
| 16 | 32 | Four uint64 dimensions; unused dimensions are zero |
| 48 | 8 | Data bytes |
| 56 | 8 | Position count |
| 64 | 4 | Layout: linear 1, spatial 2, temporal 3 |
| 68 | 4 | Rank |
| 72 | 32 | Nonzero source identity |
| 104 | 32 | Model identity |
| 136 | 32 | Processor identity |
| 168 | 4 | Audio samples per second; zero for other representations |
| 172 | 20 | Reserved, zero |

Images use canonical row-major sRGB U8 RGB data, dimensions height/width/3 and spatial layout.
Audio uses little-endian interleaved I16 or F32 data, dimensions frames/channels and temporal layout.

The audio sample rate is from 1 through 384000. Source bytes have no positions or component identities.
Features use F16 or F32, dimensions rows/width, and nonzero model and processor identities.
Floating-point data must be finite. The shape and dtype must give the exact data byte count.

Linear positions are row indices. Temporal positions are nanoseconds from the source origin.
Spatial positions are frame/row/column triples. Their count is three per feature row.

Other feature layouts have one position per row. Position bytes are preserved without recomputation.
No URL is a required replay dependency. Encoded image and audio file decoding is outside this profile.

## Selection payload

A selection has a 16-byte header: uint32 schema 1, uint32 count, and eight reserved zero bytes.
At most 64 ordered rows follow. Each row has an object ID, uint64 version, uint32 representation and reserved uint32 zero.

Representation is 2 for media and 1 for all other objects.
The selected objects must exist at earlier update sequences and permit the selection's scope.
Stored selections retain historical exact versions. Current resolution rechecks their selected objects.

## Limits

Checks cover bounded GPU admission and local file recovery. They do not prove application-level memory quality.
The module provides no authentication, grant administration or disk paging.
Live maintenance and mirror persistence use the separate runtime consumers.
Language-model activation, media encoding, native kernel admission and full runtime restoration remain separate consumers.
