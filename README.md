<p align="center">
  <img src=".github/assets/mark.png" width="720" alt="AOTX-1, Ahead Of Time eXecutive">
</p>

<p align="center">A local inference operating system designed in CUDA.</p>

<p align="center">
  <a href="LICENSE"><img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Latest release 0.3.0" src="https://img.shields.io/badge/release-0.3.0-2ea44f">
  <img alt="CUDA 13.2" src="https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white">
  <img alt="Compute capability 8.0 and above" src="https://img.shields.io/badge/compute%20capability-8.0%2B-76B900">
</p>

<p align="center">
  <img alt="Languages C, C++, CUDA and PTX" src="https://img.shields.io/badge/languages-C%20%7C%20C%2B%2B%20%7C%20CUDA%20%7C%20PTX-555555">
  <img alt="Profiles 8g, 12g, 24g and 48g" src="https://img.shields.io/badge/profiles-8g%20%7C%2012g%20%7C%2024g%20%7C%2048g-555555">
  <img alt="Platform Linux" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
</p>

<p align="center">
  <a href="docs/06-build.md">Build</a> |
  <a href="docs/07-operation.md">Run</a> |
  <a href="docs/31-http-gateway.md">HTTP API</a> |
  <a href="docs/README.md">Documentation</a>
</p>

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

AOTX-1 runs agents, model inference, memory selection and a message bus on an NVIDIA GPU.
CUDA owns the live system state. An asynchronous disk journal preserves completed changes for recovery.
Local window and terminal clients expose the same system. A separate HTTP gateway provides text and media access for application clients.

This source includes changes after release 0.3.0. The [changelog](CHANGELOG.md) separates unreleased changes from tagged releases.

Model files are separate downloads with their own licenses and verified digests.

## Capabilities

| Area | Current behavior |
| --- | --- |
| Inference | Batched text generation, paged attention, routed experts and supported hybrid layers. |
| Agents and tools | Conductor, worker and verifier roles; explicit tool selection and operator grants. |
| Persistent memory | Typed GPU objects, scoped recall, source evidence, corrections and cold storage. |
| Model controls | Identity modules, voice profiles and controls bound to exact qualified model packages. |
| Task reviews | Saved task outcomes and exact evidence for later matching tasks. |
| Media | CUDA image and audio preparation with compatible native model paths. |
| Applications | Standard chat completions plus native requests, media and persistent shared resources. |
| Recovery | Ordered journal replay and complete CCIR runtime files with embedded assets. |

A runtime uses one GPU. Profiles configure 32, 64, 128 or 256 agent slots; they do not combine memory across cards.
Available VRAM, model shape and configured capacities determine which workloads fit.

**Qualification**

Automatic memory and numerical controls require exact qualified components.
Appraisal acceptance applies to the documented model configuration.
Ordinary model support does not establish semantic qualification.

See [support and qualification](docs/support.md) for the current boundaries.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Build from source

The reference environment uses Linux, CUDA Toolkit 13.2 and a GPU with compute capability 8.6.
The build requires compute capability 8.0 or above, a compatible driver, CMake 3.28 or later, Ninja and host development libraries.
Read [build requirements](docs/06-build.md#requirements) before configuration.

From the repository root:

```sh
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash tools/profile-detect.sh
```

Use the reported profile and architecture. This example targets the 12 GiB reference card:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_PROFILE=12g -DAOTX_ARCH=86
cmake --build build
build/aotx_boot --version
```

A larger profile is not a promise that every model combination fits that card.
The 24g and 48g profiles compile; full-capacity execution on those cards remains unqualified.

[Build options](docs/06-build.md#build-options) cover context, memory, media and optional components.
[Testing](docs/testing.md) explains focused checks, required model files and display tests.

## Start a local conversation

Fetch and activate a language model from the supplied catalog:

```sh
build/aotx_models --dir models list
build/aotx_models --dir models fetch language
build/aotx_models --dir models activate language language
```

Create `aotx.settings`:

```ini
journal.dir = build/run
models.dir = models
models.roles = language
tui.on = 1
```

Start the runtime:

```sh
build/aotx_boot --settings aotx.settings
```

Enter `say what is a tick` in the console. Enter `quit` to stop and flush the run.
Use a new journal directory for a new instance.
[Operation](docs/07-operation.md) covers settings, model selection, tools, shutdown and restore.

Attach another terminal to this run:

```sh
build/aotx_tui --attach build/run --settings aotx.settings
```

The [graphical control client](docs/13-control.md), `build/aotx_ctrl`, can create, attach to and stop local instances.
The GPU window starts with `--window`.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Connect an application

The [HTTP gateway](docs/31-http-gateway.md) runs separately from the CUDA runtime.
It provides standard chat completions, streaming output, owned media and exact request cancellation.
The [native protocol](docs/32-service-wire.md) defines request state and output cursors.

Applications that need persistent participants, spaces and conversations use the
[shared service](docs/33-shared-service.md). It provides durable operation identities, scoped memory and saved responses.
Clients discover available features through `/aotx/v1/capabilities`.
No particular frontend is required.

## Preserve and restore state

The journal records authoritative inputs and device decisions in order.
Recovery applies recorded tokens and selections without repeating completed tool actions.
Only complete durable ticks survive a failed run.

A [complete runtime file](docs/28-runtime-files.md) packages models, modules, settings and recovery state in one CCIR container.

[Checkpoints](docs/25-memory-checkpoints.md) and [cold memory](docs/36-cold-memory.md) preserve typed state beyond the live payload store.
These files can contain private conversations and executable modules; keep them under the intended account's control.

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

## Documentation

Start at the [documentation index](docs/README.md) for procedures, reference manuals and shared terms.
Read [architecture](docs/01-architecture.md) for state ownership, [security](SECURITY.md) for trust boundaries,
and [support](docs/support.md) for measured limits.

| Task | Guide |
| --- | --- |
| Select or inspect a model | [Model files](docs/16-model-files.md) |
| Add roles, skills or tools | [Modules](docs/09-modules.md) and [tool SDK](docs/10-tool-sdk.md) |
| Enable scoped memory | [Live memory](docs/20-live-memory.md) and [semantic memory](docs/27-semantic-memory.md) |
| Configure identity and controls | [Control bindings](docs/37-control-bindings.md) |
| Use images or audio | [Image input](docs/29-image-input.md) and [audio input](docs/30-audio-input.md) |
| Inspect saved state | [Journal format](docs/04-journal-format.md) and [CCIR format](docs/17-ccir.md) |
| Understand earlier performance figures | [Measurements](docs/08-measured.md) |

<details>
<summary>Source layout</summary>

```text
cuda/     device modules and host glue; host glue names end in _host.cu
ptx/      embedded CUDA driver modules
disk/    file, journal, model-store and transport programs
ctrl/     graphical control client and its vendored ImGui source
gateway/  HTTP transport, credentials and deployment configuration
modules/  supplied role and identity modules
sdk/      tool contracts and examples
share/    model catalog and terminal artwork
tests/    structural, numerical and runtime checks
docs/     guides and format references
tools/    source gates and authoring tools
```

</details>

<p align="center"><img src=".github/assets/divider.png" width="720" alt=""></p>

<p align="center">Apache License, Version 2.0. See <a href="LICENSE">LICENSE</a>.</p>
