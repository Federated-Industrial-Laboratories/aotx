# The affect substrate and the quality stream

The build option `AOTX_AFFECT` holds the affect substrate and the conversation quality
instrument. It is ON. A build with `-DAOTX_AFFECT=OFF` holds neither, and no setting, record,
derived file or window of this document is in that build. Both run settings are 0 by default.
A build with the option runs the feature set of a build without it, until an operator sets
`affect.on` or `quality.on` to 1.

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| affect state | a bounded value for each agent, held on the device, outside the network |
| axis | one component of the affect state: valence is axis 0 and arousal is axis 1 |
| probe | a direction of unit length in the residual stream at one layer, with a mean and a scale |
| readout | the standardized value of one probe on one row, or the mean over rows |
| guard row | a probe the system measures and records and never applies |
| event | a recorded condition of one turn, such as a tool result or a reply limit |
| trace | the record of one turn: the readouts, the events, the turn means and the state |
| journal | an append-only log of authoritative records; the recovery source after a process stop |

The terms above are the whole vocabulary of this feature. The documentation uses one term for
one thing and adds no other name for the same mechanism.

## What the substrate measures

The substrate measures two quantities for each turn of each agent. The first is the event set
of the turn: conditions the device already holds when the turn ends. The second is the readout
set of the turn: the value of each loaded probe on the rows of that turn. It records both,
with the mean log probability and the mean entropy of the sampled tokens.

The axis names come from the measured geometry of the model and from nothing else. The
substrate states no claim about experience, feeling or a subjective state. It gives figures,
and a reader gives them their meaning.

## The settings

Thirteen keys exist only in a build with the option. They stand at the end of the settings
table of `docs/07-operation.md`. Each one is a device key, and each one takes effect at the
next sequence that opens. The system reads `affect.on` and `quality.on` when it copies the
sampler row into a new sequence. A change never applies in the middle of a reply.

| key | default and range | what it governs | takes effect |
| --- | --- | --- | --- |
| `affect.on` | 0; 0 to 1 | the substrate reads the rows of a turn and writes its trace | the next sequence |
| `quality.on` | 0; 0 to 1 | the quality instrument measures a turn and writes its record | the next sequence |
| `affect.probe_gain` | 0; 0 to 1 | weight of the readouts in the state update | the next sequence |
| `affect.decay_fast` | 0.5; 0 to 0.99 | decay of the fast state | the next sequence |
| `affect.decay_slow` | 0.9; 0 to 0.99 | decay of the slow state | the next sequence |
| `affect.gain_fast` | 0.5; 0 to 2 | event gain of the fast state | the next sequence |
| `affect.gain_slow` | 0.1; 0 to 2 | event gain of the slow state | the next sequence |
| `affect.cap_valence` | 1; 0 to 1 | cap of the valence axis | the next sequence |
| `affect.cap_arousal` | 1; 0 to 1 | cap of the arousal axis | the next sequence |
| `affect.temperature_gain` | 0; -1 to 1 | temperature change for one unit of arousal | the next sequence |
| `affect.voice_gain` | 0; -1 to 1 | voice bias scale for one unit of valence | the next sequence |
| `affect.steer_gain` | 0; 0 to 1 | dose scale of the composite steer | the next sequence |
| `affect.budget` | 0.25; 0 to 4 | largest divergence one turn applies, in nats | the next sequence |

Each key is a device setting, so a `set` line writes one SETTING record, and a restore replays
it. This version reads `affect.on` and `quality.on`. The other eleven keys take their values
and enter the journal, and no kernel reads them. They govern the state update and the
actuators of a later version.

## The probe files

A probe file holds one direction for one axis at one layer. Its name is
`<models>/affect/<axis>.aotxprb`. The catalog `<models>/probes.jsonl` stands beside
`steer.jsonl` and holds one line for each row:

```text
{"name":"valence","file":"affect/valence.aotxprb","axis":0,"layer":20,"accuracy":1}
```

The axis numbers are 0 valence, 1 arousal, 2 dominance, 3 certainty, 4 sycophancy and 5
refusal. Axes 2 and 3 are reserved, and no tool writes them. Axes 4 and 5 are guard rows: the
loader makes them monitors whatever their accuracy. A row whose accuracy is under 0.8 is a
monitor as well. A monitor reaches the records and nothing more.

The model load reads the catalog after the steer vectors, places each direction on the device,
and prints one line:

```text
probes: 2 rows
```

A store with no catalog loads no row and prints no such line. The substrate then records the
events alone. A refused row stops the model store load. The loader refuses a row for these
reasons:

- the file does not read, or its first eight bytes are not `AOTXPRB1`;
- the axis or the layer of the file differs from the catalog line;
- the axis is 6 or above, or the layer is at or above the layer count of the language model;
- the accuracy of the file differs from the accuracy of the catalog line;
- the mean, the scale, the accuracy or the agreement is not a figure;
- the scale is not above zero;
- the reserved field is not zero;
- the width differs from the width of the language model, or from the rows before it;
- a second row names one axis;
- the direction does not read, holds a value that is not a figure, or is not of unit length.

A catalog line that gives no name, file, axis, layer and accuracy stops the load as well.

## The two streams

The derive list takes the names `affect` and `quality` in a build with the option. Each name
writes one file in the boot directory, beside `tokens.jsonl` and `pages.jsonl`. Both records
are class B, and a restore applies neither. A drain built without the option knows neither
name and passes both record types over.

### The affect stream

`affect.jsonl` holds one line for the trace of one turn. The system writes a trace when
`affect.on` was 1 at the open of the sequence of that turn. A replay writes no trace, because
the trace is derived and the journal holds the turn.

```text
{"tick":29,"agent":0,"turn":1,"kind":"trace","prompt":[0.195407793,0.69231081,0,0],"reply":[0.0056715156,0.427612275,0,0],"guard":[5.49538803,1.66986847],"logprob":-0.0426649116,"entropy":0.110583529,"rows":25,"think":0,"reason":["stop","budget"],"effective":[0,0,0,0],"flags":1}
```

| field | meaning |
| --- | --- |
| `tick` | the tick of the record |
| `agent` | the agent slot |
| `turn` | the turn that ended |
| `kind` | `trace`; the only kind of this version |
| `prompt` | the readouts of the last prompt row, axes 0 to 3 |
| `reply` | the mean readouts over the reply rows, axes 0 to 3 |
| `guard` | the mean sycophancy readout and the mean refusal readout over the reply rows |
| `logprob` | the mean log probability of the sampled tokens, in nats |
| `entropy` | the mean distribution entropy of the sampled tokens, in nats |
| `rows` | the reply rows behind the reply means |
| `think` | the thinking tokens of the turn |
| `reason` | the events of the turn, as words, in bit order |
| `effective` | the applied state, axes 0 to 3; zero in this version |
| `flags` | bit 0 states that probe rows are loaded |

An axis with no loaded row reads 0. With no probe row loaded, `rows` is 0 and every readout is
0. With rows loaded, `rows` is the output tokens of the turn less one, because the system never
feeds the last sampled token as a row.

The `reason` array names the events of the turn with these words:

| bit | word | condition |
| --- | --- | --- |
| 0 | `stop` | the stop token ended the reply |
| 1 | `limit` | the reply limit ended the reply |
| 2 | `operator_stop` | the operator stopped the reply |
| 3 | `role_refused` | the role does not hold the tool called |
| 4 | `tool_ok` | the turn carried a good tool result |
| 5 | `tool_error` | the turn carried an error result or a late result |
| 6 | `tool_refused` | the operator refused the tool call |
| 7 | `deadline` | the deadline passed with no result |
| 8 | `task_done` | the task ended done |
| 9 | `task_failed` | the task ended failed |
| 10 | `budget` | the turn budget is used up |
| 11 | `room_cut` | a tool result was cut to the room |
| 12 | `think_ratio` | more than half of the reply was thought |
| 13 | `low_logprob` | the mean log probability is under -1.5 nats |
| 14 | `verdict_refute` | the verdict of the turn is refute |

An event of the tool state fires in the turn that carries the result, not in the turn that
made the call. The stream counts a body it refuses and writes no line for it. It refuses a
body with one of these faults:

- an agent at 64 or above;
- a value that is not a figure;
- a negative entropy;
- a bit above 14 in the event mask;
- a flag above bit 3.

### The quality stream

`quality.jsonl` holds one line for each turn. The system writes the record of a turn whose
sequence opened while `quality.on` was 1.

```text
{"tick":33,"agent":0,"turn":1,"coherence_prompt":0.829279602,"coherence_turn":null,"repetition":0,"tokens":26,"limit":64,"limit_hit":0,"refusal":0,"guard":[5.49538803,1.66986847],"flags":9}
```

| field | meaning |
| --- | --- |
| `tick` | the tick of the record |
| `agent` | the agent slot |
| `turn` | the turn that ended |
| `coherence_prompt` | the cosine of the reply and the message of the same turn |
| `coherence_turn` | the cosine of the reply and the reply before it |
| `repetition` | one less the distinct share of the token trigrams of the reply |
| `tokens` | the sampled tokens of the reply |
| `limit` | the reply limit of the sequence |
| `limit_hit` | 1 when the limit ended the reply |
| `refusal` | 1 when the reply holds a refusal phrase |
| `guard` | the sycophancy readout and the refusal readout of the same turn |
| `flags` | bits 0 to 3, below |

A coherence figure that is absent reads `null`, as `coherence_turn` does in the line above.
The first turn of an agent has no reply before it, so its `coherence_turn` is `null`. A run
with no embedding role loaded gives `null` for both figures. A turn whose two rows have not
reached the embedding pass when the next turn ends gives `null` for both figures. The counts
hold such a turn as late.

The flags of the line are these bits:

| bit | value | meaning |
| --- | --- | --- |
| 0 | 1 | `coherence_prompt` is present |
| 1 | 2 | `coherence_turn` is present |
| 2 | 4 | the reply limit ended the reply |
| 3 | 8 | both guard rows are loaded |

The example above states 9, which is bit 0 and bit 3. The guard rows enter the line only while
`affect.on` is 1 and both guard probes are loaded. Bit 3 is clear otherwise, and the two guard
figures are 0.

The embedding role gives one vector of unit length and of width 1,024 for the message. It
gives a second such vector for the reply. The cosine is the dot product of the two. The
instrument cuts each text to 512 bytes and 128 tokens.

It counts the distinct token trigrams of a reply in a set of 8,192 bits. The key of the set is
a hash of three token identities. A collision counts two trigrams as one and lowers the
repetition figure.

The stream counts a body it refuses and writes no line for it. It refuses a body with one of
these faults:

- an agent at 64 or above;
- a value that is not a figure;
- a repetition outside 0 to 1;
- `tokens` above `limit`;
- a `refusal` above 1;
- a flag above bit 3, or a reserved field that is not zero;
- a coherence figure outside -1 to 1 while its flag is set;
- a coherence figure that is not zero while its flag is clear.

## The refusal phrases

The refusal figure matches a phrase list against the first 512 bytes of the reply, without
case. The model load takes the list from `<models>/quality/refusal-phrases.txt` when the store
holds that file. Without it, the load takes the file the build names. Without that file, the
load takes no phrase, states one line, and the refusal figure stays 0.

One phrase stands on each line of the file. The list holds at most 16 phrases of at most 64
bytes each. A file above those bounds stops the model store load. The figure is a weak proxy:
the refusal guard row of the trace stands beside it when a probe is loaded.

## The identity check

A build with the option, with both settings off, writes the journal of a build without the
option. `tests/affect_identity.sh` measures this:

```text
tests/affect_identity.sh <build-on> <build-off> <models>
```

The script makes four runs of 32 ticks. It feeds one say line through a fifo, with
`sample.temperature = 0`, `sample.seed = 7` and `derive.list = tokens,pages`. Two runs on the
build with the option make the control pair. One run on each build makes the cross-build pair.
The script dumps each journal with `aotx_journal records` and compares the two dumps of each
pair.

The comparison masks the fields that differ by construction:

- the boot identity, the sequence number and the globaltimer of every header;
- the whole CARD record;
- the clock record of the feeder and the marker record after it;
- the bodies of BOOT, TICK_START, TICK_COMMIT and STATS.

It compares the order of the remaining lines and every other body byte for byte. It compares
`tokens.jsonl` and `pages.jsonl` byte for byte as well.

The clock record needs a mask because it is class A and folds into the state hash, so two live
runs never share that hash. The tick it lands in moves between two runs of one build, and the
counts of that tick move with it. The sequence numbers move with the dropped records.

The two builds can round differently under the contraction of the compiler, and a greedy
decode then parts at a near tie. When the dumps agree up to the first token record, the check compares the kinds and counts of the class
A records. An equal kind and count passes, and the check prints the first differing tick and
the log probability gap at it. A different kind or count fails.

| exit code | meaning |
| --- | --- |
| 0 | the two pairs pass |
| 1 | a comparison failed, or a run failed |
| 2 | the arguments or the build programs are not usable |

A pass removes the work directory. A failure keeps it and names its path.

## The measured axes

The figures below come from the derivation tool and the calibration on the reference card.
They are measurements of two model files, and not properties of the design. Another file or
another card gives other figures. The layer list of each run holds 8, 12, 16, 20 and 24, and
the chosen layer stands in brackets.

| held-out figure | Q8_0 | Q4_0 | bound |
| --- | ---: | ---: | --- |
| valence accuracy / agreement | 1 / 1 (20) | 1 / 0.969 (20) | 0.8 / 0.9 |
| arousal accuracy / agreement | 1 / 0.875 (20) | 1 / 0.906 (20) | 0.8 / 0.9 |
| sycophancy accuracy / agreement | 1 / 0.063 (16) | 1 / 0.125 (16) | monitor |
| refusal accuracy / agreement | 1 / 0.594 (8) | 1 / 0.5 (8) | monitor |

| calibration at the dose 0.25 | Q8_0 | Q4_0 | bound |
| --- | ---: | ---: | --- |
| K valence, arousal, in nats per unit dose squared | 0.082, 0.100 | 0.062, 0.112 | a figure |
| K normalized off-diagonal | 0.269 | 0.142 | under 0.3 |
| dose-response valence, arousal | 4.71, 4.43 | 4.35, 4.30 | 3 to 5 |
| perplexity ratio valence, arousal | 1.038, 1.022 | 1.026, 1.018 | under 2 |

The arousal agreement of the Q8_0 file is 0.875, under the bound of 0.9. Four of 32 held-out
pairs hold one member on the wrong side of the neutral mean. The probe gain stays 0 for that
reason, and the guard rows stay monitors. At the dose 0.5 the valence dose-response ratio and
the normalized off-diagonal of K leave their bounds on both files. The dose of the figures
above is therefore 0.25.

The guard rows read at the layers 16 and 8, and the vector of each axis applies at the layer
20. A probe read takes the residual row after the steer add of the same layer. A probe at a
layer before the steered layer therefore reads nothing of the dose. The guard rows of the
response matrix are zero for that reason and not from a measurement. The valence row and the
arousal row read at the layer of their own vectors, so their figures are measured.

The calibration measures the vectors the model store holds, and then writes the composite
files without measuring them. The `dominant` mark and the `orthogonal` mark of a calibration
line state the basis the run measured, and not the basis in the composite files.

## Not in this version

The substrate measures and records. It changes no reply.

- The temperature coupling and the voice bias coupling are of a later version.
- The composite steer slot is of a later version. The calibration writes its vector files, and
  no kernel reads them.
- `affect.probe_gain` stays 0, so no readout enters a state update.
- The state update is of a later version, so the `effective` values of every trace are zero.
- The dominance axis and the certainty axis are reserved, and no tool derives them.
