<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Testing

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Select checks according to the behavior under change.
Source gates check repository form. Device checks exercise batched contracts. Runtime checks exercise admission, visible results, cancellation and recovery.
Each result applies to its exact source, build options, model files and inputs.

<details>
<summary>On this page</summary>

- [Source gates](#source-gates)
- [Registered checks](#registered-checks)
- [Device and model checks](#device-and-model-checks)
- [Runtime acceptance](#runtime-acceptance)
- [Display, fault and sanitizer checks](#display-fault-and-sanitizer-checks)
- [Environment and result interpretation](#environment-and-result-interpretation)

</details>

## Source gates

Run these commands from the repository root:

```sh
tools/gate.sh README.md docs cuda disk gateway tests
python3 tools/parity-gate.py
```

The combined gate runs STE, size and seam checks.
With no arguments it examines staged files only, so an empty staged set proves nothing about unstaged changes.

| Gate | Scope |
| --- | --- |
| `ste-lint.py` | Sentence and paragraph bounds, punctuation and refused wording. |
| `size-gate.py` | File line limits, including the host-glue limit. |
| `seam-gate.py` | Device, disk and gateway dependency boundaries. |
| `parity-gate.py` | Terminal actions, command dispatch and help consistency. |
| `spill-gate.py` | Listed hot kernels and their local-memory or stack allowances. |
| `sanitizer-gate.sh` | CUDA memory errors and data hazards. |

Each gate returns 0 for clean results, 1 for findings and 2 for usage or environment errors.
The writing gate approximates STE; manual review checks the remaining rules.

## Registered checks

Configure and build before running CTest:

```sh
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```

Use `-R` for an exact set of affected test names. Inspect `-N` output to confirm that the selection is nonempty.
For example, these structural checks need no model download:

```sh
ctest --test-dir build -R '^(parity|size_gate_boundary|boot_children)$' --output-on-failure
```

Registration depends on build options. `tests/CMakeLists.txt` and included test configuration files define the current list.
A missing model manifest can produce `Skipped`; retain that status separately from passes.
Some live checks are standalone scripts and do not run through CTest.

## Device and model checks

Batch coverage must name its actual row count and use distinct content for each row.
The principal reference batches are N=1 and N=64.
Profile-width checks can use another count, and a larger profile build does not prove full hardware execution.

| Area | Representative checks or guide |
| --- | --- |
| Rings, scheduling and records | `seam`, `bus`, `sched`, journal and replay checks. |
| Matrix and layer behavior | `matrix_blocks`, `expert`, `bias`, `delta` and architecture fixtures. |
| Typed state and recall | `memory_config`, `memory_capacity`, `context_live`, `live_context`. |
| Source interpretation | `source_recall`, `intake_source`, `intake_stages`, `intake_capability`. |
| Appraisal | [Appraisal runtime checks](35-automatic-appraisal.md#verify-appraisal). |
| Shared state and cold data | `shared_state`, `shared_publish`, `shared_cold`. |
| Model controls | `control_device`, `qualification_device` and [consumer checks](37-control-bindings.md#verify-selection-and-recovery). |
| Child failures and embedded modules | `boot_children`, `embedded_modules` and the live boot-failure script. |

A real tensor comparison can require an exact file:

```sh
build/aotx_matrix_device_test --real-file FILE TYPE TENSOR
```

`TYPE` is Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K or Q6_K.
Omit `TENSOR` to select the first two-dimensional tensor of that type.
Missing required input fails instead of skipping.
The test compares every value with a double reference and checks both matrix products at batches of one and 64.

[Architecture accuracy](17-accuracy.md) defines the separate full-logit comparison and its unresolved results.
Structural passes do not replace numerical or semantic qualification.

## Runtime acceptance

Use existing model stores and a new output directory for each owned test run.
Keep the exact configuration, model digests, command, output counts and exit status with its results.
A failed prerequisite must remain visible in the result set.

Automatic-retention examples:

```sh
python3 tests/auto_boot_test.py build . MODEL_STORE NEW_OUTPUT_1 vector-1
python3 tests/auto_boot_test.py build . MODEL_STORE NEW_OUTPUT_64 vector-64
python3 tests/auto_boot_test.py build . MODEL_STORE NEW_OUTPUT_TEXT text-1
```

Vector cases need a language model. The text case also needs the embedding role.
See the relevant memory guide for semantic, scope and recovery fixtures.

Run HTTP transport checks in the locked gateway environment:

```sh
.venv-gateway/bin/python tests/gateway_protocol_test.py
```

Transport fixtures supply device replies; they do not establish real inference acceptance.
`tests/gateway_runtime_test.py` checks actual GPU requests with a selected model store.
Shared, appraisal and control scripts add their respective persistent-state and recovery requirements.

A runtime check must prove that its intended agents or requests were admitted before testing their results.
An empty batch, a mocked result or a copied earlier tally does not establish acceptance for a new subject.

## Display, fault and sanitizer checks

These checks need deliberate build options:

```sh
cmake -S . -B build-display -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_DISPLAY_TESTS=ON
cmake --build build-display
ctest --test-dir build-display -L display --output-on-failure
```

Display checks require a local display session and open actual windows.
Run them separately from headless GPU work.
The window test fails when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

`AOTX_FAULT_TESTS=ON` registers deliberate invalid-memory accesses that terminate a CUDA context.
Run them only on the intended test device.
`AOTX_SANITIZER_TESTS=ON` registers memory and race checks; select them with `-L sanitizer`.

```sh
python3 tools/spill-gate.py --cuobjdump /usr/local/cuda-13.2/bin/cuobjdump build
tools/sanitizer-gate.sh memcheck build models tests/fixtures/tokenizer
```

Sanitizer fixtures reduce selected workloads and omit rate measurements.
Do not present their durations as performance results.

## Environment and result interpretation

| Variable | Use |
| --- | --- |
| `AOTX_SANITIZER` | Select reduced sanitizer fixture behavior. |
| `AOTX_SANITIZER_SKIP` | Space-separated programs excluded from a sanitizer run. |
| `AOTX_SANITIZER_BIN` | Explicit compute-sanitizer executable. |
| `AOTX_FAULT_TESTS` | Enable deliberate guard-gap cases. |
| `DISPLAY`, `WAYLAND_DISPLAY` | Select the local graphical session. |
| `TMPDIR` | Parent directory for temporary disk fixtures; default `/tmp`. |

Changes limited to documentation need wording, links, source references and rendered-page checks.
They do not require new model qualification when the implementation and accepted contracts are unchanged.
Retain historical measurements under their original identities; rerun only checks affected by a changed behavior or unresolved finding.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
