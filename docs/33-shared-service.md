# Shared service

The native shared service keeps participants, spaces, conversations, and operation receipts on the GPU.
The complete runtime file saves this state.
The network gateway holds transport state and credentials.
A conversation keeps its identity when its execution slot changes.
Ordinary Chat Completions keeps its separate, temporary request contract.

The service requires a complete runtime profile with shared state enabled.
Configure `shared_read`, `shared_write`, and `shared_manage` in each principal grant.
Current grants intersect with stored membership on each admission and read.
Credentials do not enter the complete runtime file.
Restore requires fresh deployment grants.

## Enable shared operation

Add `--shared` when [packaging a complete runtime](28-runtime-files.md).
To enable an existing complete file, stop its runtime and run:

```sh
build/aotx_ccir_pack --runtime identity.aotxccir --shared
```

The command saves the required shared profile and reuses the existing component extents.
Repeating the command leaves an already compatible profile unchanged.
An older reader that lacks shared state support refuses this required profile.

Add `shared_read`, `shared_write`, and `shared_manage` to the selected principal's gateway actions.
Use the [gateway configuration](31-http-gateway.md) to write the deployment grant file.
Then start the complete runtime and gateway:

```sh
build/aotx_boot --ccir identity.aotxccir --journal /srv/aotx/journal --service-grants /srv/aotx/grants
.venv-gateway/bin/python -m gateway serve --config /srv/aotx/gateway.json
```

The configuration socket must name `/srv/aotx/journal/service.sock` for this example.
The prepared runtime must include its text embedding model for shared memory input.
Each client reads capabilities and its participant resource before registering or submitting a mutation.

## Scope

A private space is the default.
Its members share that space's conversations and permitted memory.
Two private spaces have different memory owners, including when one participant owns both spaces.
A room uses its exact space ID for room scope.
An instance space permits all registered participants with current shared grants.

Space scope cannot change.
A new room or instance space does not publish an existing private space.
The space owner has read, write, and manage rights.
Explicit member rows can add or remove these rights.
A zero-rights row also blocks implicit instance access.

## Resources

The base path is `/aotx/v1/shared`.
All routes require authentication.
All responses disable caching.

| Method | Path | Result |
| --- | --- | --- |
| GET | `/capabilities` | Current lineage and configured limits |
| GET, POST | `/participant` | Current participant, or registration |
| GET, POST | `/spaces` | Permitted spaces, or a new space |
| GET | `/spaces/{space}` | Current space rights |
| GET, POST | `/spaces/{space}/members` | Explicit members, or one membership change |
| GET, POST | `/spaces/{space}/conversations` | Conversations, or a new conversation |
| GET | `/conversations/{conversation}` | Current conversation state |
| POST | `/conversations/{conversation}/inputs` | New text and media input |
| GET | `/conversations/{conversation}/events` | Ordered input receipts |
| GET | `/operations/{operation}` | Exact permitted result and save state |
| GET | `/operations/{operation}/events` | Authenticated server-sent result events |
| POST | `/operations/{operation}/cancel` | Explicit cancellation |
| GET | `/spaces/{space}/memory` | Permitted current memory rows |
| GET | `/spaces/{space}/memory/{object}` | Current metadata and exact payload bytes |
| POST | `/spaces/{space}/publish/{object}` | Explicit memory publication |
| GET, POST | `/save` | Save status, or a recorded save request |
| POST | `/retire` | Retire saved receipts below an explicit floor |

Space IDs use `spc-{lineage}-{id}`.
Conversation IDs use `con-{lineage}-{id}`.
Operation IDs use `op-{lineage}-{id}`.
Each component after the prefix has 32 lowercase hexadecimal digits.
Operation IDs are assigned by the device and saved with their admission.
They differ from caller-supplied operation keys.

## Mutations and retries

Every POST body has these fields:

```json
{
  "schema": "aotx.shared.mutation.v1",
  "lineage": "0123456789abcdef0123456789abcdef",
  "operation_key": "123456789abcdef0123456789abcdef0",
  "sequence": "1"
}
```

Read `/participant` to obtain the actual lineage and next sequence.
Counters use decimal strings.
The operation key is a nonzero opaque 16-byte value.
The gateway rejects unknown fields and invalid UTF-8.
The device compares the complete canonical operation bytes for each known sequence.

An exact retry returns its existing receipt.
Changed input or a future sequence returns 409.
A sequence below the saved retry floor returns 410.
A concurrent client can read the participant resource after a sequence conflict.
A key can be reused after retirement, but a retired operation handle cannot select the new receipt.

A space POST accepts `id` and `scope`.
A conversation POST accepts `id`.
An omitted resource ID uses the operation key.
A membership POST requires a participant ID and a `permissions` array containing `read`, `write`, or `manage`.
An empty array removes all explicit rights.

An input POST requires `text` and `model`.
It accepts `max_output_tokens`, `pages`, `temperature`, `top_p`, and `media`.
Defaults are 256 output tokens, the current page grant, zero temperature, and top_p of one.
Text can be empty when media is present.
The text and device-added media links must fit 2048 UTF-8 bytes.
Each media link requires 73 bytes, including its separator.

A media row contains `type` with `image` or `audio`, and `sha256` with 64 lowercase hexadecimal digits.
At most eight media references are accepted.
The device checks source ownership, model compatibility, generation leases, and current resource limits.
The accepted model digest and effective page limit are recorded separately from caller bytes.

Only media references in the current shared input become model feature inputs.
Source links in recalled memory remain text and do not attach an earlier source again.
Reserved media control tokens in submitted text are refused before memory retention.

Cancellation requires `target_sequence`, which identifies the exact operation sequence.
Disconnect does not cancel execution.
Cancellation cannot undo committed memory.
A retirement POST requires `retry_floor`.
Every receipt below that floor must have a saved terminal result.
Retirement advances the persisted floor before a receipt slot is reused.

## Results and memory

A receipt reports transport acceptance, device commit, saved admission, and saved terminal result separately.
Its save fields identify the acknowledged source, file incarnation, generation, boot, and commit digest.
Pending bytes and disk errors are explicit.
Execution waits for a saved admission.
A restored unfinished admission becomes interrupted and does not repeat its memory input.

The device admits execution groups within a conservative KV page budget.
The budget accounts for current page owners, pending page requests, and each input's page limit.
Inputs that do not fit remain queued until a later group.

The execution deadline starts when the recorded lease is applied.
It includes memory processing, model generation, and result publication.
Later requests from other interfaces can use the same KV pool.
A request that exceeds its execution deadline returns status 504.

Operation reads accept an `offset` in bytes.
Replies contain exact base64 bytes, their byte length, and `next_offset`.
The text field is null when that byte span is not complete UTF-8.
The gateway does not replace invalid or partial byte spans.
Results include final input and output token counts and finish status.

List reads accept `cursor` and `limit`.
Event cursors count conversation input order.
A retired event window reports a gap and its current event floor.
Permitted readers can inspect another member's input result.
Foreign participant sequence, next-sequence, and retry-floor counters are null.

Add `stream=true` to conversation event reads for authenticated server-sent events.
Operation event reads always stream.
The event ID contains the path and exact byte or order cursor.
`Last-Event-ID` resumes that cursor when no explicit cursor is supplied.
Each new read rechecks membership and grants.

Memory lists contain 128-byte metadata rows translated to JSON.
Each row identifies object, version, kind, scope, owner, room, byte length, source, and actor.
Detail reads return one metadata row and exact payload bytes at the requested offset.
The device checks transitive visibility and current object versions.

Publication requires `source_version` and current manage rights in the destination space.
It copies a supported retained text memory, its source, and its embedding into distinct destination objects.
The original private objects keep their scope.
Unsupported source kinds return 409.
The admission record defines the complete publication and its source provenance.

## Capacity and records

CMake settings independently bound participants, spaces, conversations, members, receipts, command bytes, and result bytes.
Defaults are 1024 participants, 1024 spaces, 4096 conversations, 4096 member rows, and 1024 receipts.
Each receipt permits 8192 canonical command bytes and 65536 result bytes by default.
A full receipt table refuses new work until an eligible recorded retirement frees receipt capacity.
Participants, spaces, and conversations have fixed capacities for this runtime profile.

Shared journal records use type 37 and class A.
Each body contains at most 160 data bytes after its 32-byte part header.
Admission, lease, output, and completion transfers publish only after every part is present.
The emitter writes at most 16 records per tick.
Partial or inconsistent recovery transfers fail closed.

Execution lease revisions 2 and 3 use an 8-byte header and 40-byte rows.
Revision 3 selects compact context rendering and the historical-data system rule.
New leases use revision 4, which also shares repeated source labels within the same context byte limit.
Revision 2 retains its previous query and prompt bytes.

The header contains row count and revision as unsigned 32-bit words.
Each row contains request index, slot, sequence, actor, retention mode and a zero reserved word.
Their offsets are 0, 4, 8, 16, 32 and 36. Sequence uses 8 bytes; actor uses 16 bytes.

Mode 1 retains source text. Mode 2 also performs qualified automatic interpretation.
The device records mode 2 only when the requested model has the automatic-memory capability.
Both memory calls use that same requested model role.
Ordinary conversations remain available with mode 1 when interpretation is unavailable.

Recovery uses the recorded mode, independent of the current qualification table.
Revisions 0 and 1 retain their original 32-byte rows and mode 2 interpretation records.
The device checks every lease row before it changes any slot or receipt.

The complete file retains its logical audit prefix.
The active receipt window is bounded separately from that recovery representation.
File-cap or disk pressure pauses persistent admission.

During a journal hold, clients can read existing receipts, memory, and save status.
Current grants still control those reads.
New mutations return 429 without changing persistent state.
This service does not promise unlimited history in a fixed file size.
