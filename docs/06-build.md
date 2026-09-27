<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Build and configuration

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Build AOTX on Linux with the CUDA Toolkit and host development libraries.
A card profile sets device table capacities. Separate options set context, cache, memory and service limits.
Use a separate build directory for each configuration that must remain reproducible.

<details>
<summary>On this page</summary>

- [Requirements](#requirements)
- [Configure and build](#configure-and-build)
- [The profile and the architecture](#the-profile-and-the-architecture)
- [Build options](#build-options)
- [Context and cache capacity](#context-and-cache-capacity)
- [Memory capacity](#memory-capacity)
- [Check the build](#check-the-build)
- [Prepare model files](#prepare-model-files)

</details>

## Requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA compute capability 8.0 or above; 8.6 is the reference architecture. |
| CUDA | Toolkit 13.2 or later and a driver that supports the selected toolkit. |
| Build tools | CMake 3.28 or later, Ninja, C11 and C++17 compilers. |
| Source checks | Python 3; numerical reference checks also use NumPy. |
| Window and raster | pkg-config, GLFW 3, GLEW, OpenGL, EGL and X11 development packages. |
| Image reference tests | JPEG development headers and library. |
| Model downloads | libcurl development package when `AOTX_FETCH` is enabled. |
| Operating system | Linux, pthreads and memfd support. |
| Optional HTTP gateway | Its separate locked CPython 3.12 environment. |

The CUDA compiler must be compatible with the selected host compiler.
The disk programs contain no CUDA calls, although the complete build requires CUDA.
Disabling the control client does not remove the core window and raster dependencies.

On Ubuntu, install the host packages:

```sh
sudo apt install build-essential cmake ninja-build python3 python3-numpy pkg-config \
  libglfw3-dev libglew-dev libegl1-mesa-dev libx11-dev libxtst-dev \
  libcurl4-openssl-dev libjpeg-dev
```

Install the CUDA Toolkit separately. Add its `bin` directory to `PATH` before configuration.
See [HTTP gateway installation](31-http-gateway.md#installation) for the optional Python environment.

## Configure and build

Run from the repository root:

```sh
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash tools/profile-detect.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_PROFILE=12g -DAOTX_ARCH=86
cmake --build build
build/aotx_boot --version
```

Replace the example profile and architecture with the intended card values.
`AOTX_ARCH` controls CUDA compilation; setting only `CMAKE_CUDA_ARCHITECTURES` does not override it.
The build includes native code and PTX for the selected architecture.

Executables appear directly in the build directory.
Keep `aotx_boot`, `aotx_drain`, `aotx_feed`, `aotx_restore` and required companion programs together.
The boot program finds its child programs beside itself.

The clock and Q8 matrix PTX are embedded during configuration.
Their runtime loading does not need the source-tree PTX files.
Ordinary directory-based boot still needs its module directory. Use `--modules` when moving the binaries, or pack a complete runtime file.

## The profile and the architecture

| Profile | Slots | Virtual weights range | Physical cache pool | Default sequence tokens | Pages per slot |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8g` | 32 | 5 GiB | 1 GiB | 8192 | 148 |
| `12g` | 64 | 8 GiB | 2 GiB | 8192 | 640 |
| `24g` | 128 | 16 GiB | 8 GiB | 16384 | 320 |
| `48g` | 256 | 40 GiB | 24 GiB | 32768 | 640 |

These are configuration bounds, not simultaneous physical-memory promises.
Weights map as tensors load. Cache pages, device tables, media workspaces and model buffers also require physical memory.
Startup checks available memory and reports allocation failures.

The 24g and 48g profiles compile and link. Their full hardware capacity and throughput remain unqualified.
A 64-row test does not establish behavior at every larger profile width.
See [support](support.md#hardware-and-profiles) for the distinction between compile and runtime coverage.

Profiles reside in `cuda/profile/`. `profile.cuh` applies explicit context and cache overrides.
On Linux x86-64, the NVIDIA 48g build uses the medium host address model for its larger static tables.

## Build options

| Option | Default | Effect |
| --- | --- | --- |
| `AOTX_PROFILE` | `12g` | Select `8g`, `12g`, `24g` or `48g`. |
| `AOTX_ARCH` | `86` | Select the CUDA compute architecture. |
| `AOTX_CTRL` | `ON` | Build the graphical control client. |
| `AOTX_AFFECT` | `ON` | Include affect state and the quality instrument. |
| `AOTX_FETCH` | libcurl availability | Build model downloads; explicit ON requires libcurl. |
| `AOTX_SEQUENCE_TOKENS` | `0` | Set prompt-plus-reply token capacity; zero uses the profile. |
| `AOTX_KV_SLOT_PAGES` | `0` | Set each slot's page-table capacity; zero uses the profile. |
| `AOTX_KV_POOL_PAGES` | `0` | Set the shared physical page count; zero uses the profile. |
| `AOTX_MEMORY_OBJECTS` | `8192` | Set object-version slots per typed store. |
| `AOTX_MEMORY_BYTES` | `16777216` | Set payload bytes per typed store. |
| `AOTX_MEMORY_SNAPSHOTS` | `2` | Set pending memory checkpoint slots. |
| `AOTX_CCIR_FILE_BYTES` | `17179869184` | Set the CCIR file byte cap; zero uses the system offset limit. |
| `AOTX_MODELS_DIR` | `models` | Select the model directory used by checks. |
| `AOTX_DISPLAY_TESTS` | `OFF` | Register tests that open a local window. |
| `AOTX_FAULT_TESTS` | `OFF` | Register deliberate CUDA context-fault tests. |
| `AOTX_SANITIZER_TESTS` | `OFF` | Register CUDA memory and race checks. |
| `AOTX_BUS_LINT` | empty | Select an external validator for derived bus lines. |

A build with `AOTX_AFFECT=OFF` omits affect state, quality streams and their corresponding client controls.
An ON build still starts with `affect.on=0` and `quality.on=0`.
Control assets need their own exact [qualification components](37-control-bindings.md).

Other capacities have separate configuration families:

| Family | Reference |
| --- | --- |
| `AOTX_MEDIA_*` | [Image device capacity](29-image-input.md#device-capacity) |
| `AOTX_AUDIO_*` | [Audio capacity](30-audio-input.md#device-capacity-and-recovery) |
| `AOTX_SERVICE_*` | [Gateway bounds](31-http-gateway.md#capacity-and-failures) |
| `AOTX_SHARED_*` | [Shared capacity](33-shared-service.md#capacity) |
| `AOTX_POLICY_*` | [Policy capacity](34-creator-policy.md#operator-controls) |

## Context and cache capacity

The sequence capacity includes system text, selected memory, submitted history, media tokens and the maximum reply.
All profiles permit at least 8192 tokens by default.
The `24g` and `48g` defaults are 16384 and 32768 tokens.
An explicit setting overrides the profile value and can be lower or higher.
Set it independently of the card profile:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_PROFILE=12g -DAOTX_ARCH=86 -DAOTX_SEQUENCE_TOKENS=8192
cmake --build build
```

Larger token tables do not add physical cache pages or change the model's usable context.
The prompt byte limit, per-slot page limit and shared page pool remain separate bounds.
The runtime refuses excess input rather than removing selected memory to make it fit.
Recovery needs enough capacity for the recorded prompts and replies.

Each physical cache page contains 2 MiB. Its token capacity depends on model shape and paged-layer count.
Recurrent layer state has separate allocations.
A larger per-slot page table does not reserve that many pages for every slot.

Reducing the shared pool can leave room for combined model workspaces, but reduces cache capacity.
The [combined audio configuration](30-audio-input.md#combined-12-gib-configuration) documents this tradeoff.
Do not apply its measurements to an unrelated model combination.

## Memory capacity

Set object and payload bounds together, then rebuild the runtime and its disk programs:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTX_MEMORY_OBJECTS=16384 -DAOTX_MEMORY_BYTES=67108864
cmake --build build
build/aotx_ccir_state --limits
```

The limits report needs no GPU. Its image size includes a 128-byte header, 256-byte object rows and the payload allocation.
CMake refuses a capacity that overflows the two-image transfer's 32-bit byte count.
A running store cannot resize.

The runtime also allocates input, result, scratch, binding and checkpoint buffers.
The image size is not the runtime's total memory requirement.
Models, context, shared state, media workspaces and the CUDA context add separate costs.

Typed schema 1 remains stable across these capacities.
A larger build can restore a smaller valid file.
A smaller build refuses excess objects or payload before partial publication.
Capacity pressure does not implicitly enable offload, eviction or reclamation.

## Check the build

Inspect the registered checks before execution:

```sh
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```

Model-dependent checks can report `Skipped` when their assets are absent.
A skipped check is not a passed check.
Use [testing](testing.md) to select the affected source, device, display and runtime checks.

Run source gates with an explicit scope:

```sh
tools/gate.sh README.md docs cuda disk gateway tests
```

Without arguments, `tools/gate.sh` checks staged content only.
Its exit status is 0 for clean checks, 1 for findings and 2 for an environment or usage error.

## Prepare model files

Model files are not part of the repository.
The model store records file identity, role, source, revision, license, byte count and SHA-256.
The supplied catalog is `share/models/catalog.jsonl`.

```sh
build/aotx_models --dir models list
build/aotx_models --dir models fetch language
build/aotx_models --dir models activate language language
build/aotx_models --dir models check
```

Fetching changes the store; activation selects a manifest role.
Neither operation replaces a resident model in an already running instance.
[Model files](16-model-files.md) covers local files, custom catalogs, turn wraps and runtime loading.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
