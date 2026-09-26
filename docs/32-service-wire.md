<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Native API and broker protocol

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Native routes use the same authentication, grants, model aliases and input controls as standard completions.
They expose device-owned request and source resources.
They contain no frontend account, scene or renderer fields.

<details>
<summary>On this page</summary>

- [Client sequence](#client-sequence)
- [Request identity and state](#request-identity-and-state)
- [Event stream](#event-stream)
- [Capabilities and telemetry](#capabilities-and-telemetry)
- [Broker packet format](#broker-packet-format)
- [Broker submission and reply](#broker-submission-and-reply)
- [Broker media lists](#broker-media-lists)
- [Policy operations](#policy-operations)

</details>

## Client sequence

1. Read `/aotx/v1/capabilities` and choose an available model alias.
2. Submit the complete input to `/aotx/v1/requests`.
3. Retain the returned handle, runtime epoch and media handles.
4. Read output windows or subscribe to the request event stream.
5. Continue reading until a terminal state and all output bytes have arrived.
6. Remove unused media after its request leases end.

The [gateway guide](31-http-gateway.md) covers credentials, input fields and deployment.
[Shared conversations](33-shared-service.md) use a separate saved-operation contract.

## Request identity and state

A request ID has this form:

```text
req-<16 lowercase hexadecimal epoch digits>-<32 lowercase hexadecimal identity digits>
```

The epoch names one runtime activation.
The remaining bytes identify one admitted request independently of its execution slot.
A handle is not an access credential.
Each read and cancellation checks the caller's current device grant and principal.

POST `/aotx/v1/requests` accepts the standard message envelope with `stream` absent or false.
It also accepts typed reusable media parts.
An HTTP 202 response has this shape:

```json
{
  "schema": "aotx.admission.v1",
  "id": "req-0000000000000001-00000000000000000000000000000001",
  "runtime_epoch": "1",
  "state": "accepted",
  "persistence": "ephemeral",
  "media": []
}
```

Acceptance means the device holds the request.
It does not mean inference completed or disk storage is durable.
The HTTP connection can close after admission without stopping the request.
There is no persistent idempotency key in this profile.

GET `/aotx/v1/requests/{id}` returns `aotx.request.v1`.
Its fields are:

| Field | Meaning |
| --- | --- |
| `id`, `runtime_epoch` | Request identity and decimal runtime epoch |
| `model_role` | Loaded device role number |
| `state` | `queued`, `preparing`, `running`, `completed`, `failed`, or `cancelled` |
| `cancel_requested` | The exact request has received a stop request |
| `status` | Zero or the device failure status |
| `persistence` | `ephemeral` |
| `usage` | Actual prompt, completion and total token counts |
| `finish_reason` | `stop`, `length`, or null |
| `output` | Base64 bytes and decimal cursor positions |

`cancel_requested` can be true while the state remains running.
Only a terminal state confirms that device execution has ended.
A completed request remains completed if cancellation arrives after its stop.
Cancellation cannot stop a later user of its former slot.

An output window has `encoding`, `bytes`, `cursor`, `next_cursor` and `total_bytes` fields.
The encoding is `base64`.
Supply `?cursor=DECIMAL` for a byte offset.
If `next_cursor` is less than `total_bytes`, read the next window before treating a terminal result as fully received.
An offset greater than the current result returns 409.

A foreign, reclaimed or unknown handle returns 404.
An ended runtime epoch returns 410.
These errors do not return an empty successful result.

## Event stream

GET `/aotx/v1/requests/{id}/events` returns SSE.
It supports `?cursor=DECIMAL` or a `Last-Event-ID` header, but not both.
An emission event contains:

```text
event: emission
id: req-0000000000000001-00000000000000000000000000000001:3
data: {"schema":"aotx.emission.v1","request_id":"req-0000000000000001-00000000000000000000000000000001","runtime_epoch":"1","source_kind":"model","model_role":2,"offset":"0","next_cursor":"3","encoding":"base64","bytes":"YWJj"}

```

The event ID combines the exact request ID and the next byte cursor.
Reconnect with that ID to resume from the following byte.
Offsets count UTF-8 source bytes, not characters or rendered cells.
A base64 part can end inside a UTF-8 sequence.
Keep an incremental decoder when converting parts to text.
Standard completion SSE performs this decoding in the gateway.

`status` events contain the request state fields without an output window.
`error` events contain the standard error object.
The native stream sends one terminal status after all final bytes, then closes.
It does not use the standard completion `[DONE]` marker.
Reconnection validates epoch, ownership and cursor again on the device.
No display frame rate or animation rule controls event ordering.

## Capabilities and telemetry

Capabilities use `aotx.capabilities.v1` and report effective device and transport limits.
Model entries contain alias, role, model SHA-256, input modalities and publication time.
Each entry also contains the boolean `automatic_memory` capability.
The model list at `/v1/models` exposes the same field.

The device checks the exact loaded model, wrapper, processors and source profile.
An unavailable memory capability does not remove ordinary text input support.
These fields have the same meaning for all HTTP clients.
The optional lineage names the active cognitive runtime file lineage.
It does not make ordinary request results persistent.
False feature values mean that no supported route provides that function.

Telemetry uses `aotx.telemetry.v1` and requires the telemetry action.
The sample contains service allocation bytes, device tick and device clock nanoseconds as decimal strings.
These counters describe the current runtime epoch.
The service allocation value is not total GPU use.
Unavailable affect and expression fields are null.
Clients can share and sample this resource independently of rendering frequency.

## Broker packet format

The gateway connects to `service.sock` with Linux Unix `SOCK_SEQPACKET`.
The broker accepts only its own user ID and never sends a mapped file descriptor.
Each packet is at most 65536 bytes with a 128-byte header.
All integer fields are little endian.
Unused input fields and reserved bytes are zero.
The binary format is an internal versioned transport; HTTP clients do not need it.

The information payload retains its 192-byte header and 40-byte model rows.
The unsigned word at header offset 156 contains the qualified automatic-memory role bits.
Only visible model roles can have a bit set. Other reserved header bytes remain zero.

| Offset | Width | Input |
| --- | --- | --- |
| 0 | 8 | ASCII `AOTXAPI1` |
| 8 | 4 | Operation |
| 12 | 4 | Zero |
| 16 | 16 | Principal |
| 32 | 8 | Grant revision |
| 40 | 8 | Runtime epoch for request operations |
| 48 | 16 | Request or media ID |
| 64 | 8 | Read cursor |
| 72 | 4 | Submission model role |
| 76 | 4 | Submission output token limit |
| 80 | 4 | Submission temperature, IEEE 754 binary32 |
| 84 | 4 | Submission top_p, IEEE 754 binary32 |
| 88 | 4 | Payload bytes |
| 92 | 36 | Zero |

| Operation | Code |
| --- | ---: |
| Install grants | 1 |
| Read information | 2 |
| Submit request | 3 |
| Read request | 4 |
| Cancel request | 5 |
| Transfer media | 6 |
| Read media | 7 |
| Read telemetry | 8 |
| List media | 9 |
| Shared mutation | 10 |
| Shared read | 11 |
| Policy read or control | 12 |
Network client connections cannot submit grants.
The private control mailbox installs an operator file with a strictly increasing revision.
An empty table revokes all principals.

Grant rows are 64 bytes and contain principal, revision, action/model masks and resource limits.
Result reads and cancellation require the admission revision and a current model grant.
A replacement grant does not restore access to results from an earlier revision.
They contain no bearer token or credential hash.
The header declares the row count at offset 76.

## Broker submission and reply

Submission payloads start with a 32-bit message count.
Each message has a 32-bit role and part count.
Roles are system=0, user=1 and assistant=2.
Each part has a 32-bit kind, byte count and exact bytes.
Kinds are UTF-8 text=0, image digest=1 and audio digest=2.

Media digests contain exactly 32 bytes and require a private ready source owned by the principal.
CUDA applies the loaded model's trained message format to the whole array.

Replies place HTTP-like status and request phase at offsets 8 and 12.
The runtime epoch is at 40; output windows retain their requested cursor at 64.
The role, total output bytes, prompt tokens and sampled tokens are at 72, 76, 80 and 84.
The payload length is at 88.
Finish, failure status and cancellation request are at 92, 96 and 100.

Finish values are zero, stop=1 and length=2.
Request phases follow the order free, queued, preparing, running, completed, failed and cancelled.

Mapped mailbox ownership changes through release/acquire operations.
State zero belongs to the broker; one publishes a request; two publishes the device reply.
A disconnected in-flight mailbox remains unavailable until its reply arrives.
Its old reply cannot reach a new connection.
The host holds transport state only; the authoritative grants, request results and source leases remain on the device.


## Broker media lists

Media-list replies contain zero or more 80-byte rows.
Each row starts with a 16-byte source ID, then the 64-byte media-read metadata.
The reply cursor at offset 64 is the next source-table position, or zero at the end.
The scan includes only sources owned by the current principal and excludes refused entries.
A source change during pagination can change a later page.

## Policy operations

Operation 12 reads or controls the resident policy. Shared operations retain codes 10 and 11.
An empty payload reads aggregate status. A control payload is 16 bytes: schema 1, action and expected 64-bit control revision.
Actions are pause=1, resume=2, stop=3, review-on=4 and review-off=5.

The frame epoch must match for mutations. Request ID and cursor are zero.
Mutation requires action grant bit 128. Reads require bit 8 or bit 128.

The 160-byte response starts with eight 32-bit words: schema, ABI, mode, state, review-enabled, pending, active rows and result status.
State values are off=0, quiet=1, active=2, paused=3, stopped=4, error=5 and recording=6.
Eleven 64-bit fields start at byte 32.
The fields contain control revision, source frontier, completed, interrupted, refused, decision and saved generation.
The remaining fields contain maximum nanoseconds, last nanoseconds, written bytes and result bytes.

Byte 120 contains a 32-bit reason: quiet=0, foreground=1, paused=2, capacity=3, active=4 or disabled=5.
Bytes 124 through 159 are zero. No private evidence identifiers or text enter this response.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
