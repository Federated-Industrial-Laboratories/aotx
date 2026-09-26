# Fitted control bindings

`AOTX_AFFECT` is ON at build time. The settings `affect.on` and `quality.on` default to 0.
A build with `AOTX_AFFECT=OFF` excludes affect state and the quality stream.

A fitted control belongs to one exact model file and resolved turn format.
Equal vector widths do not establish compatibility.
The loader refuses a missing binding, changed model, changed format, or changed asset.
This rule applies to model directories and complete runtime files.

The derivation tool writes a `.binding` file beside each vector, probe, and calibration file.
Retain this file when copying an asset.
An old asset without a binding must be derived against an identified model.
A binding identifies bytes. It does not establish a behavioral effect or an acceptable capability cost.

## Binding format

The format is little-endian and contains exactly 608 bytes.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `AOTXCTL1` or `AOTXCTL2` |
| 8 | 4 | Version, 1 or 2 |
| 12 | 4 | Hook, 1 |
| 16 | 4 | Kind: vector 1, probe 2, calibration 3 |
| 20 | 4 | Position mode: all tokens 0, response tokens 1 |
| 24 | 32 | SHA-256 of the complete model file |
| 56 | 512 | Canonical resolved turn format |
| 568 | 32 | SHA-256 of the complete asset |
| 600 | 8 | Zero |

Version 1 requires position mode 0 and retains all-token behavior.
Version 2 requires position mode 1 and a vector asset.
A legacy consumer refuses version 2. Probe and calibration bindings retain version 1.

Response controls start at the first complete tokenizer piece inside the final generation prefix.
The prefix includes the declared thought markers.
A piece that crosses the prefix boundary stays unchanged.
A missing or invalid prefix disables the response control.
The GPU retains the logical boundary through prompt chunks and generated tokens.
System text, history, and user text before this boundary receive no response control.

Hook 1 reads or adds after the complete layer residual update.
The model digest also identifies its tokenizer and quantization.
The turn format stores resolved strings, token IDs, and format kind.
Its derived `usable` field is zero. Both derived thought token IDs are `UINT32_MAX`.

Asset bytes identify the exact layers, width, vector values, and calibration revision.
Calibration also records both composite asset digests.

A live model replacement disables controls whose model or turn format no longer matches.
Probe accuracy below 0.8 or neutral agreement below 0.9 permits monitoring only.
A guard probe never drives state.
An absent or excluded probe is unavailable. A zero state value does not establish neutral emotion.

## Qualification component

A binding alone cannot enable a runtime control.
An absent `.qualification` file leaves the control unavailable.
An invalid qualification file refuses loading.
The same checks apply to directories and complete runtime files.

The qualification file is a JSON object with schema 1.
Unknown keys, repeated keys, unknown versions, and unknown control kinds are refused.
The object contains these fields:

| Field | Content |
| --- | --- |
| `schema` | 1 |
| `kind` | Vector 1, probe 2, or affect composite calibration 3 |
| `status` | `accepted` or `unavailable` |
| `checks` | Completed acceptance requirements, as a bit mask |
| `doses` | Accepted settings in units of 0.0001 |
| `binding` | Reference to this asset's `.binding` file |
| `source` | Editable identity description and explicit preferences |
| `commitments` | Stable commitments, requested tools, and relationship rules |
| `examples` | Contrast examples and their provenance |
| `calibration` | Separate calibration inputs and results |
| `acceptance` | Acceptance inputs, commands, results, and limits |
| `consumer` | Authored identity consumer and operational defaults |

Each reference contains `file` and `sha256`.
The file name is relative and its complete contents must match the digest.
Retain every referenced file when packing or copying the component.
The binding connects qualification to the exact model, tokenizer, format, hook, positions, layers, and control bytes.

Identity source and commitments remain text. A vector does not store these facts.
Requested tool rights cannot increase the operator's grant.

The check bits are effect 1, capability cost 2, controls 4, calibration 8, conversation 16,
architecture observations 32, scope 64, and recovery 128.
An accepted component requires all eight bits and the corresponding evidence files.
These files record measured acceptance. Their hashes do not establish that the measurements are correct.
The operator must check the evidence before marking a component accepted.

A vector lists up to 16 distinct, nonzero doses between -40000 and 40000.
The GPU accepts only those exact settings. It does not interpolate between them.
A probe has no dose entries. Its accuracy and guard-axis restrictions still apply.

A composite lists three positive bounds: absolute valence dose, absolute arousal dose, and divergence budget.
The GPU limits both axes and the budget to those bounds.
An unavailable component has no accepted settings.

Separate accepted vectors do not establish an accepted combination.
This schema admits one vector at a time, without affect or voice coupling.
The sampler refuses other combinations; the residual hook checks again before application.
Measurement programs use a separate registration call for candidates.
A qualification file cannot grant measurement access.

Model and control names do not select admission rules.
New entries use the same reader, asset references, device checks, and recovery path.
New control kinds or setting contracts need an explicit versioned adapter.
An older reader refuses them without reinterpreting an accepted component.

`GET /aotx/v1/capabilities` includes control entries for each granted model.
Each entry reports availability, exact accepted doses, positions, layers, and the qualification digest.
Unavailable entries have an empty dose list.
The route is independent of the graphical client.

Device information revision 2 retains the 192-byte header and 40-byte model rows.
Header offsets 160 and 164 hold the control count and the 160-byte control row size.
Control rows follow the model rows.

| Control row offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | Model role |
| 4 | 4 | Control kind |
| 8 | 4 | Available flag |
| 12 | 4 | Position mode |
| 16 | 4 | Hook |
| 20 | 4 | Accepted dose count |
| 24 | 8 | Layer mask |
| 32 | 32 | Zero-terminated name |
| 64 | 32 | Qualification digest, or zero when absent |
| 96 | 64 | Sixteen signed doses, with unused entries zero |

Revision 2 reports named residual vectors here.
The shared affect read reports probe availability for the recorded turn.

## Authored identity consumer

Keep the identity description and commitments in the referenced text files.
Put the active description in a role module's `overlay.txt`.
The role manifest names this file with `body: overlay.txt` and records the identity revision in `version`.
Record the module name, overlay digest, requested tools, and defaults in the referenced consumer file.
The local conversation and shared service read the selected role overlay on the GPU.
The native stateless request uses the system message supplied by its caller.

A source file reference does not install a role or grant tools.
The operator selects the role, checks its overlay against the source, and sets tool policy.
Requested module tools remain subject to the operator's effective tool mask.
Leave a control off when its component is unavailable.

Keep assets, bindings, qualification files, and their referenced files in the model directory.
Use regular files for ancillary assets.
Pass the identity role directory through `aotx_ccir_pack --modules` when creating the runtime.
The complete file retains the model assets, authored modules, settings, and saved scope state.
A new model/control pair uses new measured data with these same readers and routes.

## Request selection

The native request and shared conversation input routes accept an optional `control` object.
It contains `schema`, `kind`, `qualification_sha256`, and `dose`.
Use schema `aotx.control.selection.v1`, kind `residual_vector`, and a nonzero accepted integer dose.
The digest selects the exact qualification file reported by capabilities.
No control field keeps the existing request behavior.
The OpenAI-compatible route does not accept this extension.

Capability feature `control_selection` reports support for this request contract.
Device information revision 2 stores selection version 1 at header offset 168.
An older device reports no selection support or refuses the nonzero reserved bytes.

A selector has 48 bytes: U32 version, U32 kind, I32 dose, four zero bytes, and a 32-byte qualification digest.
All-zero bytes mean no selection.
A service submission stores its selector size at header offset 92 and appends the selector after its message payload.
The payload size includes the selector. Only sizes zero and 48 are supported.

A shared input stores the same selector at command offsets 144 through 191.
Other shared commands require zero in this range.

Admission and execution check the current model, qualification, and exact dose on the GPU.
An unavailable, changed, or ambiguous qualification returns status 503.
An invalid selector shape returns status 400.
The shared journal retains the selector with the exact input command.
Recovery restores recorded outcomes without repeating inference or checking current control availability.
New work after recovery uses the current qualification checks.

An active affect scope cannot also select a residual vector.
After affect is disabled, one unselected turn completes the recorded state reset before vector selection.
Accepted controls cannot increase tool rights or alter request scope.

## Shared affect state

The shared service uses the existing sharing preset.
Private conversations own separate affect state.
Room and instance conversations use their space's affect state.
A shared state admits one affect-managed turn at a time.
Separate private states can execute in one batch.

An execution slot receives a working copy when its lease begins.
Completion records an absolute successor and advances the scope revision once.
Slot release clears the working state, sums, and law.
Recovery applies the recorded successor without computing a new update.

Cancellation before inference leaves the scope state unchanged.
Cancellation after inference starts records the stop event.
An off turn clears previously enabled state when that turn completes.
Default-off turns retain the earlier lease and completion record formats.

`GET /aotx/v1/shared/conversations/{id}/affect` reads authorized scope state.
It uses the same membership checks as the conversation read.
The response includes the revision, quantized parts, event mask, and actuator cost.
It names the model and each probe's availability at the last recorded turn.

These fields describe runtime controls. They do not measure a person's emotional state.
A build without affect returns status 501 for an authorized read.

## Shared records

Lease revision 5 retains the 40-byte row of revision 4.
Row offset 36 contains an affect management flag, either zero or one.
The flag uses the recorded value during recovery.
The memory query remains `AOTXCTX4`.

A managed completion has 128 bytes. Offset 28 contains version 1.
Its first 64 bytes otherwise retain the earlier completion fields.

| Offset | Bytes | Field |
| --- | --- | --- |
| 64 | 8 | Previous scope revision |
| 72 | 8 | Successor scope revision |
| 80 | 8 | Four fast signed Q1.15 parts |
| 88 | 8 | Four slow signed Q1.15 parts |
| 96 | 2 | Budget scale |
| 98 | 2 | Axis count, 2 |
| 100 | 4 | Actuator flags |
| 104 | 4 | Finite nonnegative F32 budget cost |
| 108 | 4 | Event mask |
| 112 | 4 | Available probe mask: valence 1, arousal 2 |
| 116 | 4 | Enabled flag |
| 120 | 8 | Zero |

A malformed successor or incorrect previous revision refuses before state mutation.
A duplicate completion cannot advance the state again.
An interrupted lease without a completion retains the previous scope revision.

## Consumer checks

Use `tests/control_runtime_test.py` to check selection, scope, refusal, and copied-file recovery.
The console control command is `agent 0 decode.steer0 name:dose`.
Use `agent 0 decode.steer0 absent` to disable that control.
The test checks the console acknowledgement before it starts inference.
Identical greedy replies do not prove that a numerical control is absent.
Qualification measures its effect on output scores as well as consumer behavior.
