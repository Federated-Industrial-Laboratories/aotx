# HTTP gateway

The gateway serves ordinary inference through a separate host process.
CUDA owns request admission, model input, media processing, results, usage and cancellation.
The gateway owns HTTP, TLS, credentials, bounded uploads and HTTPS transfers.
It has no renderer or application account dependency.

Ordinary requests use the submitted messages in order.
They do not add stored conversations, create learned memories or execute server tools.
Existing local conversations use their existing path.
Continuing CCIR conversations use the separate [shared resources](33-shared-service.md).
The capability resource reports whether the active runtime supports them.

## Installation

Build the core and broker with the normal CUDA build.
The broker target is `aotx_service`.
Place it beside `aotx_boot`, `aotx_feed` and `aotx_drain`.

The supplied dependency lock supports CPython 3.12 on Linux x86-64.
Run these commands from the repository root:

```sh
python3.12 -m venv .venv-gateway
.venv-gateway/bin/python -m pip install --require-hashes --only-binary=:all: -r gateway/requirements-linux-cp312.txt
```

The gateway uses aiohttp 3.14.3.
It does not require an inference SDK or CUDA Python package.
The core does not require the gateway Python environment.

## Operator configuration

The gateway and runtime use one trusted operating-system account.
Account access has operator authority; device grants constrain the HTTP and service protocols.

Create a JSON configuration file owned by the service user.
Remove group and public write access.
Use mode 0600 for configuration and grant files.
The socket path must fit the Linux Unix socket path limit of 107 bytes.

```json
{
  "socket": "/srv/aotx/journal/service.sock",
  "host": "127.0.0.1",
  "port": 8081,
  "revision": "1",
  "models": {
    "vision": {"role": "language", "published_at": 0},
    "audio": {"role": "language-audio", "published_at": 0}
  },
  "principals": [{
    "id": "00000000000000000000000000000001",
    "token_sha256": ["REPLACE_WITH_64_LOWERCASE_HEX_DIGITS"],
    "models": ["vision", "audio"],
    "actions": ["infer", "upload", "fetch", "telemetry"],
    "pages": 0,
    "tokens": 256,
    "requests": 4,
    "media": 16,
    "media_bytes": 33554432
  }],
  "origins": [],
  "urls": {"public": true, "private": []}
}
```

Replace the credential hash with the SHA-256 of a strong random bearer token.
The token must contain 32 through 256 characters.
Keep the token outside model prompts and CCIR files.
`published_at` is the operator's Unix publication time for that model alias.
Only permitted loaded aliases appear in discovery.
Supported roles are `language`, `language-q4` and `language-audio`.

`pages: 0` selects the device profile page ceiling.
A positive page quota can reduce it.
The effective quota appears in capabilities.
Other principal defaults are 256 output tokens, two active requests, 16 media objects and 32 MiB of media.
Each principal can have up to four credential hashes for key rotation.

Write the binary grant table before boot:

```sh
.venv-gateway/bin/python -m gateway grants --config /srv/aotx/gateway.json --output /srv/aotx/grants
build/aotx_boot --models /srv/aotx/models --roles language,language-audio --journal /srv/aotx/journal --service-grants /srv/aotx/grants
.venv-gateway/bin/python -m gateway serve --config /srv/aotx/gateway.json
```

The device installs the complete grant table before the broker accepts clients.
To change grants, increase the decimal revision, write the file, and send SIGHUP to the owned broker process.
Restart the gateway with the matching configuration.
The device refuses old revisions and cancels their active requests.
A new grant revision cannot read or cancel results admitted under an earlier revision.
Do not reuse a principal ID for a different person who must not read its stored media.

## Routes

Every data route requires `Authorization: Bearer TOKEN`.
Credentials in query strings or message bodies have no authority.
Responses disable shared caching.

| Method and path | Result |
| --- | --- |
| GET `/v1/models` | Permitted ready model aliases |
| POST `/v1/chat/completions` | Standard JSON or SSE text completion |
| GET `/aotx/v1/capabilities` | Current limits, codecs, models and available features |
| POST `/aotx/v1/requests` | Native inference admission, HTTP 202 |
| GET `/aotx/v1/requests/{id}` | Owned state and output byte window |
| POST `/aotx/v1/requests/{id}/cancel` | Exact request cancellation, with an empty JSON object |
| GET `/aotx/v1/requests/{id}/events` | Owned byte events and state changes |
| POST `/aotx/v1/media` | Immutable JPEG or WAV body |
| GET `/aotx/v1/media` | Current owned sources, including interrupted HTTP uploads |
| POST `/aotx/v1/media/import` | HTTPS import with `{"url":"https://..."}` |
| GET `/aotx/v1/media/{id}` | Owned source state and digest |
| DELETE `/aotx/v1/media/{id}` | Source removal when no request holds it |
| GET `/aotx/v1/telemetry` | Separately granted current device counters |
| GET `/aotx/v1/policy` | Aggregate policy and task review state |
| POST `/aotx/v1/policy` | Revision-bound operator control; requires `policy_manage` |

Unavailable routes return a structured error.
The service does not provide Responses, embeddings, transcription, tools, expression or presence resources.
The [native schema](32-service-wire.md) defines request identities, state and byte cursors.

## Standard requests

Set a standard client's base URL to `http://127.0.0.1:8081/v1`.
Choose an alias returned by `/v1/models`.
Supply a `messages` array with system, user and assistant roles.
Text content can be a string or an array of `{"type":"text","text":"..."}` parts.
User parts can also contain:

```json
{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...","detail":"auto"}}
```

```json
{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
```

An image URL can instead name an authorized HTTPS source.
Media must match the selected model's input capability.
Plain text with the reserved `[image:` or `[audio:` prefix is refused.
Native media markers cannot bypass typed source ownership.

Supported controls are `temperature`, `top_p`, and either `max_tokens` or `max_completion_tokens`.
Temperature is 0 through 2; top_p is greater than 0 and at most 1.
Only `n: 1`, `modalities: ["text"]`, and `store: false` are accepted.
Use `stream: true` for SSE.
Use `stream_options: {"include_usage": true}` for a final usage chunk.

Unknown fields, tool roles, seed, penalties and custom stop strings are refused.
Duplicate keys, non-finite numbers and invalid UTF-8 are refused.
The service never silently trims submitted history.

Usage records actual prompt and sampled token counts.
A model stop returns `stop`; an output token limit returns `length`.
An error after stream headers produces an error event and closes without `[DONE]`.

A disconnected HTTP client does not cancel admitted device work.
Use the exact handle in `X-Request-ID` to inspect or cancel it.
Disable automatic POST retries when admission could have succeeded before a transport failure.
This profile does not provide persistent idempotency.

## Media lifetime

Uploads return an opaque `media-` handle, SHA-256 and device preparation state.
CUDA validates and prepares the actual JPEG or WAV bytes.
Uploads, including inline message media, count against the principal's source quotas until removal.

A full device source or feature pool returns HTTP 429.
Retry after the reported delay, and remove unused sources to release their storage.
Known feature pressure refuses BEGIN before source bytes enter the journal.
Preparation checks feature capacity again because other admitted sources can use that capacity.
A source that exceeds the complete configured pool returns HTTP 413.
Media status 12 identifies temporary pool pressure.

The GPU keeps an asynchronous upload refusal until cleanup or receipt expiry.
Source descriptor reuse cannot remove this pending result.
The receipt table has the configured source count; a full table returns HTTP 429.
Pending refusal receipts count against the principal's source quota.
The gateway removes a failed upload after it reads the refusal.
Unclaimed refusal receipts expire after `AOTX_SERVICE_UPLOAD_SECONDS`.

A removed source cannot supply a new input and is absent from the current source list.
Its owned handle can still report canceled state until the descriptor slot is reused.
Inline uploads return their handles in `X-AOTX-Media-Ids` when the complete header fits 4096 bytes.
Native admission also returns a `media` array.
Retain those handles to reuse or remove the sources.

GET `/aotx/v1/media` lists owned sources, including sources prepared before an interrupted response.
This also supplies handles after a failed request or an omitted large media header.
Follow its `next_cursor` with `?cursor=DECIMAL` until the cursor is null.
Each page reports current device state; concurrent source changes do not form a fixed multi-page snapshot.

Native requests accept this user part:

```json
{"type":"media","media_id":"media-0123456789abcdef0123456789abcdef","modality":"image"}
```

Use `audio` for a WAV source.
Queued and running requests hold source generation leases.
Removal returns 409 while a lease is active.
A refused or incomplete transfer does not become a usable source.
The device expires incomplete transfers after its configured upload deadline.

Canonical source bytes and ownership use the existing journal and runtime mirror.
HTTP acceptance is not a disk durability receipt.
Ordinary request results are ephemeral GPU state and are not learned conversation history.
Restarting the gateway can read retained handles from the same runtime epoch.
Restarting the runtime expires them and requires fresh deployment grants.

## Deployment and bounds

Loopback is the default bind address.
Use a standard TLS reverse proxy, VPN or tunnel for remote access.
A direct non-loopback bind requires `certificate` and `key` paths.
TLS handshake and public connection pressure also require bounds at the network edge.

The gateway does not trust forwarded identity headers.
It accepts no cookie login.
Browser origins require exact entries in `origins`; preflight permits only the supported methods and headers.

HTTPS imports verify the original hostname and pin all checked address results to that connection.
They reject redirects, proxy environment settings, cookies, compressed bodies and unsafe public destinations.
Public imports use port 443.
Explicit private exceptions use an exact origin and address networks:

```json
{"origin":"https://feed.example:8443","networks":["10.20.30.40/32"]}
```

Place these entries in `urls.private`.
An optional `urls.ca_file` adds a trusted certificate authority for approved private feeds.
The private exception does not disable TLS verification.

Transport limits are positive configurable integers in `limits`.
Defaults are 128 connections, eight body readers, 64 active operations, and 64 local packet connections.
The body byte credit is 64 MiB; JSON and source limits are 8 MiB and 32 MiB.
There are four concurrent HTTPS imports, with one per principal.
JSON parser storage and bounded output buffers add to the byte credit.

Connection headers, request bodies and writes have separate deadlines.
Capabilities report configured transport bounds and current device limits.

Device capacities use `AOTX_SERVICE_CHANNELS`, `AOTX_SERVICE_PRINCIPALS`, `AOTX_SERVICE_REQUESTS` and `AOTX_SERVICE_OUTPUT_BYTES`.
The defaults are 128 mailboxes, 256 principals, 128 retained requests and 65536 output bytes per request.
One mailbox is reserved for operator grants.
The control frame permits at most 1022 principal rows.

`AOTX_SERVICE_REQUEST_SECONDS` and `AOTX_SERVICE_UPLOAD_SECONDS` default to 300 and 120 seconds.
The reference profile supports 64 agent slots; the conductor retains slot zero.
Queued requests execute in groups on free unbound slots.
Occupied operator slots are not replaced.

Temporary pressure returns 429.
Use `Retry-After` when retrying status reads after transport pressure.

During a journal hold, scoped reads and deployment grant updates remain available.
New mutations return 429 until journal capacity is available.
Upload expiry waits until recorded work can resume.

Permanent byte or context limits return 413; invalid shapes return 400.
Model capability or transport loss returns 503.
Requests can be reclaimed after their terminal result when the retained table needs space.
No minimum result retention duration is promised.

## Checks

Run transport regression tests with the gateway environment:

```sh
.venv-gateway/bin/python tests/gateway_protocol_test.py
```

These tests inject device replies to check HTTP boundaries.
`tests/gateway_runtime_test.py` checks real GPU inference at N=1 and N=64 with the Python SDK.
It takes a built runtime, source tree, existing model store, new short output path and batch count.
An optional fixture file supplies actual media paths and questions.
The seam gate restricts gateway dependencies and refuses dynamic process or code execution.
Device and disk code retain their separate seam checks.


A submit error after an uncertain device exchange includes `X-Request-ID`.
Use its native status resource to check admission before any manual retry.
A disconnect before response headers can prevent delivery of this handle.
This profile does not provide persistent idempotency.

Policy control uses a separate explicit operator grant. See [task reviews](38-task-reviews.md) for actions and revision checks.
