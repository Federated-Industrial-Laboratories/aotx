# Conduct controls and instruments

The conduct path applies steer vectors and voice bias profiles to one agent. The model store
holds the files. The journal holds each selection as an `agent <id> decode.*` input line, so a
restore applies the same selection.

## Derive a steer vector

`aotx_steer_derive` loads the language model through the normal model loader. Its pair file has
one positive prompt, one tab, and one negative prompt on each line. It accepts at most 32 pairs.
The layer list uses comma-separated zero-based layer numbers in ascending order, each named
once. The rows of the vector file stand in that order, and the loader applies them in that
order.

```text
aotx_steer_derive --models models --trait directness \
  --pairs tests/fixtures/steer/directness.tsv --layers 8,16,24
```

The program captures the final residual stream of each named layer. It writes the mean positive
minus negative residual as `directness.aotxvec`. It then uses the pair prompts as a probe set and
measures the mean per-token KL divergence between the unsteered and steered next-token
distributions. The unit is nats. The file enters `steer.jsonl` only with this potency figure. A
file with no matching figure is refused with the reason on the error output.

The vector file starts with `AOTXSTV1`, the hidden width, the layer count, the potency, and a
reserved value. Layer numbers and layer-major float values follow. Keep derived vector files in
the model store. They are model data and are not source files.

## Select conduct items

One sampler row holds two steer selections. Each selection has a strength from -4 through 4.
The value uses a name, a colon, and the strength.

```text
agent 0 decode.steer0 directness:0.75
agent 0 decode.steer1 absent
agent 0 decode.voice concise
```

The forward pass makes one fused add after the final feed-forward residual add of each named
layer. No selected vector means that the kernel writes no residual value. This state preserves
the prior forward result bit for bit.

The store can hold 16 loaded vectors and 16 loaded voice profiles. A voice profile can hold 128
entries. The first line of a `.profile` file is its name. Each later line holds a bias, a tab,
and one vocabulary string. Load tokenizes each string once. The sampler adds its bias before the
softmax.

The token statistics stream records each selected token. A reader can therefore count
the profile strings in one reply and state their frequency shift.

## Page map

Attention adds normalized attention mass to the key and value page that supplied each key. Every
64 ticks, the tick commit writes one class B page record for each resident page or page with mass.
The record holds the agent, page, residency, cadence, and mass. The drain writes `pages.jsonl`
with the tick, agent, page, residency, and mass.

The instrument makes a second key traversal only during an instrumented language pass. It adds
one atomic value per head, key block, and page contribution. The 64-tick flush scans the fixed
page table and reserves the worst-case record count. No eviction or model policy reads this
table.

## Model parameters

A store scan writes `parameters.jsonl`. One line names a model and its declared sampling
controls. A control enters the line only when the model metadata gives its `default`, `min`, and
`max` values. The scanner recognizes temperature, top-p, minimum-p, top-k, repeat penalty,
repeat window, presence penalty, and frequency penalty. It does not supply a missing value.
