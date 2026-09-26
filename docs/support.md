<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Support and qualification

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Support depends on the exact feature, model file, wrapper and build configuration.
A model that can generate text does not automatically qualify for semantic memory or a numerical control.
The current source contains unreleased changes after version 0.3.0.

## Hardware and profiles

| Profile | Agent slots | Build status | Runtime boundary |
| --- | ---: | --- | --- |
| `8g` | 32 | Compiles | Earlier reference-card measurements; workload fit still requires admission. |
| `12g` | 64 | Compiles | Current reference profile on a 12 GiB RTX 3060. |
| `24g` | 128 | Compiles | Full-capacity hardware execution and throughput remain unqualified. |
| `48g` | 256 | Compiles | Full-capacity hardware execution and throughput remain unqualified. |

The accepted integration builds use compute capability 8.6.
Compute capability 8.0 is the minimum supported architecture, subject to the selected toolkit and driver.
A runtime uses one GPU. Multiple cards do not form one weights or cache pool.

The affect and quality JSON file writers currently accept slots 0 through 63.
Larger profile builds do not extend that derived-stream limit.

N=64 checks exercise 64 distinct rows where the fixture requires them.
They do not establish full-capacity behavior at 128 or 256 slots.
The combined image/audio integration uses one active conversation with a 64-slot build.
It does not establish 64 simultaneous media encoders or conversations.

## Models and optional features

| Feature | Current boundary |
| --- | --- |
| Ordinary text | Supported tensor, tokenizer, layer and wrapper paths; inspect each exact file. |
| Automatic semantic memory | One exact model, wrapper, source profile and two-stage processor entry. |
| Automatic appraisal | Accepted evidence applies to the same exact 9B model configuration; other models need separate semantic qualification. |
| Numerical control | One qualified Qwen3-4B Q4_0 curiosity vector, response positions, layer 24, dose 0.5. |
| Other fitted controls | Unavailable unless their exact component passes all qualification requirements. |
| Image input | Native supported Qwen3.5 visual path and compatible image assets. |
| Audio input | Native supported Qwen2-Audio path and compatible audio assets. |
| Task reviews | Exact supported task outcomes and source evidence; no generated general advice. |

Automatic memory checks the immutable table in `cuda/cognitive/intake_capability.cu`.
The accepted model SHA-256 is:

```text
2ca636d9e81d3d23ca9b60c234fe185d30ec082eeba69ce770fdb0c76559a4f5
```

The complete wrapper and both processor identities must also match.
Unqualified shared conversations retain source text and expose unavailable automatic memory through discovery.
They do not silently use another interpretation model.

Appraisal remains an optional operator control. Its response grammar checks structure and source evidence, not general semantic accuracy.
An accepted appraisal result does not establish a universal model qualification.
Unsupported dimensions remain unknown.

The qualified control's model SHA-256 is:

```text
ae782b4a90b57dc4faf880855ea4b285a96b4a335544144526dab73df91d4d61
```

Its exact binding, qualification component and referenced evidence must also be present.
The repository does not bundle model or fitted control assets.
The loader checks component identity; the operator must verify its declared acceptance evidence.
See [control bindings](37-control-bindings.md) for extension and recovery contracts.

## Recovery and compatibility

A complete CCIR runtime carries required assets, scope state and recorded decisions.
Recovery validates those records without repeating completed inference or tool calls.
A reader refuses unknown required features instead of silently discarding them.

Creator-policy updates support explicit preservation for compatible ABI, state schema and size.
The operator must affirm that existing state bytes retain their meaning.
An arbitrary state-schema conversion is not implemented.

Host or device failure can lose work after the last complete durable tick.
HTTP admission is not a durability receipt. Ordinary service results expire with their runtime epoch.
Persistent shared operations use the separate recorded service contract.

## Known limits

The numerical full-row reference comparison remains unresolved.
Its recorded failures and fixed bounds remain in [architecture accuracy](17-accuracy.md).
No new numerical accuracy claim follows from integration or exact replay.

Source selection can miss a requested topic even when the response follows the current instruction.
The instruction-conflict control retains this retrieval limitation.
Accepted substantive recall cases still require their exact source and evidence.

Media integration checks lifecycle, source ownership and recovery.
It does not establish general image recognition, transcription or media reasoning quality.
[Historical measurements](08-measured.md) retain their original versions and are not current performance promises.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
