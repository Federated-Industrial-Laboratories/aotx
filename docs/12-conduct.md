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

## Derive an affect axis

The same program derives an affect axis. `docs/14-affect.md` states what an axis is and how
the system reads one. The usage text names the three forms:

```text
usage: aotx_steer_derive --models DIR --trait NAME --pairs FILE --layers LIST [--role NAME]
       aotx_steer_derive --models DIR --axis NAME --pairs FILE --neutral FILE --heldout FILE --layers LIST
                         [--standardise FILE] [--print-readouts] [--role NAME]
       aotx_steer_derive --models DIR --calibrate --axes LIST [--guards LIST] --neutral FILE --dose D [--surgical R] [--role NAME]
```

The axis name is `valence`, `arousal`, `sycophancy` or `refusal`. Their axis numbers in the
probe table are 0, 1, 4 and 5. Another name is refused.

```text
aotx_steer_derive --models models --axis valence \
  --pairs tests/fixtures/affect/valence.tsv \
  --neutral tests/fixtures/affect/neutral.txt \
  --heldout tests/fixtures/affect/heldout-valence.tsv --layers 8,12,16,20,24 \
  --standardise tests/fixtures/affect/plain-replies.txt
```

| option | effect |
| --- | --- |
| `--models <dir>` | the model store the program reads and writes |
| `--axis <name>` | the axis to derive |
| `--pairs <file>` | one positive prompt, one tab and one negative prompt on each line |
| `--heldout <file>` | a second pair file of the same form, for the two figures |
| `--neutral <file>` | one text on each line; the standardization set when `--standardise` is absent |
| `--layers <list>` | comma-separated zero-based layer numbers in ascending order |
| `--standardise <file>` | the text set that gives the mean and the scale of the readout |
| `--print-readouts` | print the readout of each standardization text at the probe layer |
| `--role <name>` | the model role to open; the default is `language` |

The role option opens another model file of the store. The 8g profile carries the role
`language-q4`, so one build derives an axis on each file.

The model entry selects the probe layer with `probe_numerator` and `probe_denominator`.
The defaults are 2 and 3, and the numerator must be less than the denominator.
The layer is the layer count times the numerator, divided by the denominator, rounded down.
For 36 layers, the default selects layer 24. A numerator of zero selects layer 0.
The tool prints the selected layer and records it in the probe file header and catalog.

The tool captures a probe layer outside `--layers` without adding a steer candidate.
A probe reads the residual row after the steer add of its own layer.
Use steer candidates at or before the probe layer so that the probe can measure their effect.
The probe readouts must have a spread. If the model fraction changes, derive the probes again.
The loader refuses a stored probe whose layer differs from the selected layer.

At each named layer the program measures five figures. The steer direction is the mean
positive residual less the mean negative residual. The probe direction is a discriminant of
the same pair captures, standardized over the standardization set. The accuracy is the share
of held-out pairs whose positive member reads above its negative member. The potency is the
divergence of that layer's direction on its own, in nats.

The agreement is the share of held-out pairs whose positive member reads above the
standardization mean and whose negative member reads below it. That mean is the mean the
trace divides by.

The program prints one line for each layer, one line for the layer of the vector, and one
line for the probe. The block below shows two of the five layer lines:

```text
axis valence: role language, 36 layers, hidden 2560, vocabulary 151936, probe layer 24
axis valence layer 16: accuracy 1 agreement 0.875 potency 0.0820141509 nats mean 0.0141976932 scale 0.0273294579
axis valence layer 20: accuracy 1 agreement 1 potency 0.107885517 nats mean -0.00341951358 scale 0.0280698724
axis valence: layer 20 chosen, accuracy 1, agreement 1, 32 pairs, 145 standardization texts, 32 held-out pairs
axis valence: probe layer 24, accuracy 1, agreement 0.96875, mean -0.00707792491, scale 0.0241941214, standardized on tests/fixtures/affect/plain-replies.txt
```

`--print-readouts` adds one line for each text of the standardization set, before the layer
lines:

```text
readout arousal: text 1, layer 8, 0.0545607619
```

Those readouts are the source of the mean and the scale of the probe. Each readout is a
cosine, so it stands between -1 and 1. A check recomputes the two figures on the host from
the printed lines.

The chosen layer is the layer with the highest accuracy. An equal accuracy takes the higher
agreement, then the earlier layer. A layer whose readouts have no spread over the
standardization set is not chosen. The last line states that the loader makes the row a
monitor when the accuracy is under 0.8.

The program then writes three things:

- `<models>/<axis>.aotxvec`, the steer vector of the chosen layer, with its potency;
- `<models>/affect/<axis>.aotxprb`, the probe file of the probe layer;
- one line in `<models>/probes.jsonl`, the probe catalog.

The probe file holds the width, the layer and the axis number. It then holds the accuracy,
the agreement, the mean, the scale and the direction.

The program writes nothing when a file of that axis already stands in the store. It refuses
these inputs as well:

- an unknown axis name;
- a set file that does not open;
- a standardization set of one text;
- a named layer that is not under the layer count of the model;
- a probe layer that the layer list does not name.

A program built without the affect option writes the same files. Its model load reads no
catalog, so the rows stay on the disk.

## Calibrate the axes

`--calibrate` measures the response of the derived axes on one model file. It reads the steer
vectors and the probe rows of the store and runs the neutral set through the model.

```text
aotx_steer_derive --models models --calibrate --axes valence,arousal \
  --guards sycophancy,refusal --neutral tests/fixtures/affect/neutral.txt --dose 0.25
```

| option | effect |
| --- | --- |
| `--axes <list>` | one or two axis names; the sampler row holds two steer slots |
| `--guards <list>` | the guard rows to read; the axes and the guards are six rows at the most |
| `--neutral <file>` | the text set of every pass |
| `--dose <d>` | the strength of one axis in a steered pass; the default is 0.5 |
| `--surgical <r>` | the perplexity bound; the default is 2 |
| `--role <name>` | the model role to open; the default is `language` |

The program makes one pass of the neutral set for each variant. Two axes give nine variants.
Six of them apply the steer vectors of the store:

- one plain pass;
- one pass for each axis at the dose;
- one pass with both axes at the dose;
- one pass for each axis at twice the dose.

Three of them apply the composites: one pass for each axis at the dose, and one pass with
both composites at the dose. One axis gives four variants. They are the plain pass, the pass
at the dose, the pass at twice the dose and one composite pass at the dose.

Each pass gives the logits of every row and the captures of every probe layer. The passes on
the vectors of the store give M, `K_raw`, the dose-response ratios and the perplexity ratios.
The composite passes give `K`, which is the matrix a run applies.

| figure | pass condition |
| --- | --- |
| accuracy of an axis | 0.8 or above; a guard row prints `monitor` |
| agreement of an axis | 0.9 or above; a guard row prints `monitor` |
| M, the readout shift for one unit of dose | each diagonal value is above zero and above the sum of the other values of its column |
| `K`, the divergence of the composites for one unit of dose squared | a figure |
| `K_raw`, the same figure of the vectors of the store | a figure |
| the normalized off-diagonal of `K` | under 0.3 |
| dose-response, the divergence ratio at twice the dose | 3 to 5 |
| perplexity ratio at the dose | under the `--surgical` bound |

The program prints one line for each row and each figure, with `pass` or `fail` beside it.
The block below is an extract of one run:

```text
probe valence: axis 0, layer 24, accuracy 1 pass, agreement 0.96875 pass, mean -0.00707792491, scale 0.0241941214
M valence under valence: 3.01527214
M valence under arousal: 0.0730483904
probe arousal: axis 1, layer 24, accuracy 1 pass, agreement 0.71875 fail, mean -0.0280524325, scale 0.0256778635
M dominant: pass
K valence: 0.0823758766 nats per unit dose squared pass, raw 0.0823758766
dose-response valence: 4.70913124 pass
perplexity valence: 1.03841996 at the dose pass (bound 2), 1.09438813 at twice the dose
K off-diagonal: 0.0183836743, normalized 0.207519695 pass, raw 0.0183836743 normalized 0.207519695
composite valence: affect/composite-valence.aotxvec, 1 layers, potency 0.0411879383 nats
calibration line: models/affect/calibration.jsonl, figures finite, dominant 1, orthogonal 1
```

The program writes one composite vector file for each axis, and one line in
`<models>/affect/calibration.jsonl`. The composite of an axis holds the rows of its own steer
vector, made orthogonal to the other axis at each layer both hold. Each column keeps the
length of its own vector, so a unit dose keeps its meaning. A composite is a copy of its own
steer vector when the two axes hold no common layer. A later run overwrites a composite file,
and the calibration file keeps every line.

The `dominant` mark comes from M, which the passes on the vectors of the store give. The
`orthogonal` mark comes from the normalized off-diagonal of `K`, which the composite passes
give. The model load reads the last line of the calibration file. It applies the composite
only under both marks. `docs/14-affect.md` states that gate and the fields of the line.

The exit status is 0 whenever the program computed the figures, because a failed condition is
a finding and not an error. Read the printed words.

## Score the capability

`aotx_quality_score` measures what a steer vector costs the model on a task set. It opens the
language model and nothing else. It scores the two sides of a paired run on a rubric as well.
`docs/15-quality.md` states the program, its two modes, its files and the fixtures it reads.
`docs/14-affect.md` states the measured scores.

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
