<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Numerical accuracy comparison

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

This tool records an empirical comparison; it is not a completed accuracy qualification for version 0.3.0.
The measured float32-reference comparison fails 136 of 40,928 rows: hybrid 0, attention bias 76, and experts 60.
The cause of those differences remains unresolved. The frozen bounds have not changed.
These results do not change the exact same-runtime replay requirement.

## Reference and inputs

The full-row check uses an external reference package and the original runtime model files.
The package records separate original and decoded-weight model hashes.
Each decoded model stores every original decoded weight value in float32.
Model metadata remains identical, except for the declared weight storage type.
The CPU reference uses float16 key and value caches and the recorded CPU libraries.

Teacher tokens stay fixed. Argmax tokens are separate observations.

The reference index names four input groups: `original`, `short`, `long` and `fresh`.
Each group keeps its own 64 distinct sequence slots and prefill shape.
Only original calibration members can set bounds. All other retained inputs are regression inputs.
Fresh inputs supply independent validation coverage in both serial and batch modes.

## Metrics and acceptance

The three metrics are maximum absolute logit error, relative L2 error and probability total variation.
Relative L2 uses the CPU row norm. Total variation uses the full vocabulary after separate stable softmax operations.
Each bound is 1.5 times its original calibration maximum across both modes.
This empirical factor does not give an error guarantee for other inputs.
The probability bound must be less than one.

Every checked row must satisfy all three bounds.
If the CPU top-two gap exceeds twice the absolute error bound, the argmax tokens must match.
Each model and mode must have a clear winner in the fresh group.
Clear historical rows cannot supply that coverage. A failed validation cannot raise a bound.

## Capture and compare

Set `AOTX_ACCURACY_ASSETS` to the root of the recorded external assets.
Set `AOTX_ACCURACY_REFERENCE` to the reference package directory.
Set `AOTX_ACCURACY_RESULTS` to an empty results directory that already exists.
Set `AOTX_ACCURACY_STORE` to the model store for the selected original model.
Set `AOTX_ACCURACY_MODEL` to its model ID in the reference index.
Run this capture for each model in the original group:

```sh
python3 tests/arch_accuracy.py capture \
  --corpus "$AOTX_ACCURACY_REFERENCE/reference-original.manifest.json" \
  --asset-root "$AOTX_ACCURACY_ASSETS" --model "$AOTX_ACCURACY_MODEL" \
  --store "$AOTX_ACCURACY_STORE" --executable build/aotx_arch_logits \
  --output "$AOTX_ACCURACY_RESULTS/original-$AOTX_ACCURACY_MODEL"
```

Each capture runs all serial inputs and the complete batch.
Freeze bounds before comparisons with fresh runtime rows:

```sh
OPENBLAS_NUM_THREADS=1 python3 tests/arch_accuracy.py calibrate \
  --reference-index "$AOTX_ACCURACY_REFERENCE/reference-index.json" \
  --asset-root "$AOTX_ACCURACY_ASSETS" \
  --captures "$AOTX_ACCURACY_RESULTS"/original-*/capture.json \
  --output "$AOTX_ACCURACY_RESULTS/bounds.json"
```

Capture every model in the remaining groups with the corresponding manifest and output prefix.
Check all groups against the frozen bounds:

```sh
OPENBLAS_NUM_THREADS=1 python3 tests/arch_accuracy.py check \
  --reference-index "$AOTX_ACCURACY_REFERENCE/reference-index.json" \
  --asset-root "$AOTX_ACCURACY_ASSETS" \
  --captures "$AOTX_ACCURACY_RESULTS"/*/capture.json \
  --bounds "$AOTX_ACCURACY_RESULTS/bounds.json" \
  --output "$AOTX_ACCURACY_RESULTS/validation.json"
```

The check refuses missing groups, changed inputs and mixed capture executables.
It writes complete counts, per-row results and separate fresh clear-winner counts.
Missing clear coverage fails the check even when all numerical bounds pass.
Keep each build configuration's captures in a separate directory.

## Reproduce the CPU reference

For CPU reproduction, set `AOTX_REFERENCE_SOURCE` to the recorded CPU source revision.
Set `AOTX_REFERENCE_LIB` to its pinned CPU library directory.
Set `AOTX_REFERENCE_OUTPUT` to a new output directory with space for all reference rows.

```sh
c++ -std=c++17 -O2 -x c++ tests/arch_reference.cu \
  -I"$AOTX_REFERENCE_SOURCE/include" -I"$AOTX_REFERENCE_SOURCE/ggml/include" \
  -L"$AOTX_REFERENCE_LIB" -Wl,-rpath,"$AOTX_REFERENCE_LIB" \
  -lllama -lggml -lggml-base -o build/arch_reference_cpu
OPENBLAS_NUM_THREADS=1 python3 tests/arch_reference.py \
  --reference-index "$AOTX_ACCURACY_REFERENCE/reference-index.json" \
  --asset-root "$AOTX_ACCURACY_ASSETS" --executable build/arch_reference_cpu \
  --output "$AOTX_REFERENCE_OUTPUT"
```

Reproduction checks every row and token metadata hash before it writes success metadata.
It retains the original reference records and records the actual reproduction executable separately.
The finite input set does not certify conversational quality or accuracy for other inputs.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
