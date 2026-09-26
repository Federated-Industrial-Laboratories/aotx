<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Affect state and measurement streams

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

The optional affect subsystem records turn events, probe readouts and a bounded numerical state.
Its actuators can modify sampling or residual controls when their exact assets and settings are qualified.
These values describe runtime mechanisms; they do not establish subjective experience or measure a person's emotions.

`AOTX_AFFECT` defaults to ON at build time.
Runtime settings `affect.on` and `quality.on` both default to zero.
An OFF build excludes these runtime state and stream paths.

<details>
<summary>On this page</summary>

- [Availability and terms](#availability-and-terms)
- [What the substrate measures](#what-the-substrate-measures)
- [The state](#the-state)
- [The update law](#the-update-law)
- [The settings](#the-settings)
- [The probe files](#the-probe-files)
- [The actuators](#the-actuators)
- [The calibration file](#the-calibration-file)
- [The two streams](#the-two-streams)
- [The restore](#the-restore)
- [The refusal phrases](#the-refusal-phrases)
- [The identity check](#the-identity-check)
- [The measured axes](#the-measured-axes)
- [The measured actuators](#the-measured-actuators)
- [Two findings of the measurement](#two-findings-of-the-measurement)
- [Current limits](#current-limits)

</details>

## Availability and terms

Current numerical control acceptance is narrower than the mechanisms described here.
An asset requires its exact binding, qualification component and accepted setting before runtime use.
Historical axis measurements below do not enable a current affect composite.
See [control bindings](37-control-bindings.md) and [support](support.md).

| Term | Meaning |
| --- | --- |
| Event | A recorded condition of the completed turn. |
| Readout | A standardized probe observation at a declared model layer. |
| Drive | The numerical input to the state update. |
| State | Fast and slow quantized parts retained between turns. |
| Actuator | A bounded change to sampling or a residual vector. |
| Monitor | An admitted probe that can report observations but cannot drive state. |

## What the substrate measures

The substrate measures two quantities for each turn of each agent. The first is the event set
of the turn: conditions the device already holds when the turn ends. The second is the readout
set of the turn: the value of each loaded probe on the rows of that turn. It records both,
with the mean log probability and the mean entropy of the sampled tokens.

The readout of a probe on one row is the cosine of the row with the probe direction. The
cosine is the dot product of the row and the direction over the norm of the row. The substrate
then standardizes the cosine by the mean and the scale of the probe file. A row of no length
reads 0. The cosine does not follow the norm of the residual along the sequence, so a short
reply and a long one read on one scale.

The axis names come from the measured geometry of the model and from nothing else. The
substrate states no claim about experience, feeling or a subjective state. It gives figures,
and a reader gives them their meaning.

The [control binding contract](37-control-bindings.md) defines exact asset compatibility
and conversation, room, and instance state ownership in the shared service.

## The state

Local agents retain separate state. Shared conversations retain state by their declared private, room or instance scope.
A leased execution slot receives that scope's working copy; completion records its successor.
See [shared affect state](37-control-bindings.md#shared-affect-state).

Each axis of the state has a fast part and a
slow part. The fast part follows the events of the last turns. The slow part holds a longer
trend. The effective state of an axis is the sum of its two parts, bound by the cap of that
axis.

Two axes carry data: valence is axis 0 and arousal is axis 1. Axis 2 and axis 3 stay zero.
Zero is the numerical reset state, not an inferred neutral emotion.
With zero drive, each part decays toward zero.
Rounding can retain a small nonzero value.

Each state part is a signed 16-bit fraction with denominator 32768.
Its stored range is -1 through 32767/32768.
The budget scale is an unsigned 16-bit fraction, with 65535 representing one.
The law computes in float from the quantized value and quantizes the result before the store.
The table therefore holds exactly what the state record holds, and a restore gives the same state.

An unbound sequence opened with `affect.on=0` resets its agent state to zero and its budget scale to one.
Shared scopes record their reset when the off turn completes.
A later enabled turn starts from that recorded reset state.

## The update law

The law runs once for each turn, at the end of that turn. It reads the drive of the turn and
gives the two parts of each axis their next value:

```text
fast = tanh(affect.decay_fast * fast + affect.gain_fast * drive)
slow = tanh(affect.decay_slow * slow + affect.gain_slow * drive)
```

The drive of an axis is the sum of the weights of the events that fired. The probe drive
enters it at `affect.probe_gain`: the reply readout for valence and the prompt readout for
arousal. An axis with no loaded row and an axis with a monitor row give a probe drive of 0.

The hyperbolic tangent bounds each part between -1 and 1 before storage.
With zero drive and decay below one, the unrounded update tends toward zero.
The stored update rounds each result, so it can retain a small residue.
An enabled probe can supply drive without a discrete event.

The effective state is the sum of the two parts, bound by the cap of the axis.
A cap that binds the sum sets bit 3 of the flags of the trace.

These are the weights of the fifteen events, on the valence axis and the arousal axis:

| bit | event | valence | arousal |
| --- | --- | ---: | ---: |
| 0 | `stop` | 0.10 | 0 |
| 1 | `limit` | -0.25 | 0.25 |
| 2 | `operator_stop` | -0.50 | 0.50 |
| 3 | `role_refused` | -0.50 | 0.25 |
| 4 | `tool_ok` | 0.50 | 0 |
| 5 | `tool_error` | -0.50 | 0.25 |
| 6 | `tool_refused` | -0.30 | 0 |
| 7 | `deadline` | -0.50 | 0.50 |
| 8 | `task_done` | 0.50 | 0 |
| 9 | `task_failed` | -0.50 | 0.25 |
| 10 | `budget` | -0.25 | 0.25 |
| 11 | `room_cut` | -0.10 | 0 |
| 12 | `think_ratio` | 0 | 0.25 |
| 13 | `low_logprob` | -0.10 | 0.25 |
| 14 | `verdict_refute` | -0.25 | 0 |

The weights are constants of the code and not settings. A restore applies the recorded state
and never computes it again, so a changed table cannot make a journal differ.

## The settings

The following thirteen device settings exist in an affect build.
They take effect at the next sequence. The complete [settings reference](settings.md) separates them from startup and terminal values.

| key | default and range | what it governs | takes effect |
| --- | --- | --- | --- |
| `affect.on` | 0; 0 to 1 | the substrate measures a turn, updates the state and applies it | the next sequence |
| `quality.on` | 0; 0 to 1 | the quality instrument measures a turn and writes its record | the next sequence |
| `affect.probe_gain` | 0; 0 to 1 | weight of the readouts in the drive of the update | the next sequence |
| `affect.decay_fast` | 0.5; 0 to 0.99 | decay of the fast part | the next sequence |
| `affect.decay_slow` | 0.9; 0 to 0.99 | decay of the slow part | the next sequence |
| `affect.gain_fast` | 0.5; 0 to 2 | drive gain of the fast part | the next sequence |
| `affect.gain_slow` | 0.1; 0 to 2 | drive gain of the slow part | the next sequence |
| `affect.cap_valence` | 1; 0 to 1 | cap of the effective valence | the next sequence |
| `affect.cap_arousal` | 1; 0 to 1 | cap of the effective arousal | the next sequence |
| `affect.temperature_gain` | 0; -1 to 1 | temperature change for one unit of effective arousal | the next sequence |
| `affect.voice_gain` | 0; -1 to 1 | voice bias scale for one unit of effective valence | the next sequence |
| `affect.steer_gain` | 0; 0 to 1 | dose of the composite steer for one unit of effective state | the next sequence |
| `affect.budget` | 0.25; 0 to 4 | largest divergence one turn applies, in nats | the next sequence |

Each key is a device setting, so a `set` line writes one SETTING record, and a restore replays
it. The open of a sequence copies the eleven law figures and `affect.on` into a law of that
sequence. The update and the actuators read that copy and never the live settings table. A
change therefore reaches the next sequence that opens and never the reply in hand.

## The probe files

A probe file holds one direction for one axis at one layer. It holds the mean and the scale of
the cosines of that direction over the standardization set. It holds the accuracy and the
agreement of the derivation as well. `docs/12-conduct.md` states the tool that writes it. Its
name is `<models>/affect/<axis>.aotxprb`. The catalog `<models>/probes.jsonl` stands beside
`steer.jsonl` and holds one line for each row:

```text
{"name":"valence","file":"affect/valence.aotxprb","axis":0,"layer":24,"accuracy":1}
```

The axis numbers are 0 valence, 1 arousal, 2 dominance, 3 certainty, 4 sycophancy and 5
refusal. Axes 2 and 3 are reserved, and no tool writes them. Axes 4 and 5 are guard rows: the
loader makes them monitors whatever their accuracy. A row whose accuracy is under 0.8 is a
monitor as well. A monitor reaches the records and the drive of no axis.

The loader checks asset bindings and qualification before making probe rows available.
An admitted catalog produces a count such as:

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

## The actuators

The substrate applies the effective state to the sampler row of a sequence that opens. The
say path calls the apply after the open of the sums and before the sequence opens. The
sequence keeps that row, so no actuator changes in the middle of a reply. A sequence that
opens while `affect.on` is 0 takes the plain sampler row.

The temperature coupling multiplies the temperature of the row:

```text
temperature = temperature * (1 + affect.temperature_gain * effective arousal)
```

The voice bias coupling gives the row a voice scale:

```text
voice_scale = 1 + affect.voice_gain * effective valence
```

Each result is bound to the range 0 to 2. The sampler multiplies the bias of the voice
profile by the voice scale before the softmax. A voice scale of one gives the plain product,
bit for bit. A gain of 0 leaves the temperature and the bias as the row holds them.

The voice bias is an actuator of the sampler, inside the device. The spoken voice coupling of
the control program is a different mechanism, outside the device, and `docs/13-control.md`
states it.

The composite steer is the third steer slot of the sampler row. A build with the option holds
three steer slots, and a build without it holds two. A say graph node before the say start
builds the row of each agent that opens a sequence:

```text
dose = affect.steer_gain * effective state
q = dose transpose K dose
scale = min(1, sqrt(2 * affect.budget / q))
row of a layer = scale * sum over the axes of dose * composite direction
```

`K` is the divergence matrix of the calibration file, in nats for one unit of dose squared.
The budget scale bounds the calibrated quadratic divergence estimate by `affect.budget`.
It is not a universal bound on every generated distribution. The conduct kernel
adds the row of the agent at each layer the composite holds. Bit 1 of the flags of the trace
states an applied composite, and bit 2 states a budget scale under one. The trace of the turn
carries the applied divergence as `budget_spent`.

The composite applies only under all of these conditions:

- its exact binding and accepted qualification component pass, including dose and budget bounds;

- the calibration file loads, and its last line marks `dominant` 1 and `orthogonal` 1;
- `affect.on` is 1 and `affect.steer_gain` is not 0;
- at least one axis has a dose that is not 0.

The steer gain reads as 0 while the calibration is not trusted, and the slot then stays
empty. The block writes a row of zeros for an agent that applies nothing, so no earlier row
stands. A build without the option holds no actuator and no third slot.

## The calibration file

`<models>/affect/calibration.jsonl` holds one line for each calibration run.
`docs/12-conduct.md` states the mode that writes it. The model load reads the last line of
the file. A store with no such file, or with a line the load refuses, states one line:

```text
affect composite: the last calibration is not trusted
```

A trusted line gives the layers of the composite and the width of the model:

```text
affect composite: 2 layers, hidden 2560, trusted
```

A calibration line holds these fields:

| field | meaning |
| --- | --- |
| `role` | the model role of the run |
| `axes` | the steered axis names, in the order of the rows and columns of the matrices |
| `guards` | the guard row names of the run |
| `layers` | the layers of the composite file of each axis |
| `delta` | the dose of the run |
| `rows` | every probe row read, the axes first |
| `probe_layers` | the layer each probe row reads |
| `M` | the readout shift of each row for one unit of dose of each axis |
| `K` | the divergence matrix of the composite vectors, in nats per unit dose squared |
| `K_raw` | the same matrix of the steer vectors of the store |
| `ratio` | the dose-response ratio of each axis |
| `perplexity` | the perplexity ratio of each axis at the dose |
| `perplexity_twice` | the same ratio at twice the dose |
| `surgical` | the perplexity bound of the run |
| `composite` | the composite vector file of each axis |
| `dominant` | 1 when M is diagonally dominant over the steered axes |
| `orthogonal` | 1 when the normalized off-diagonal of `K` is under 0.3 |

The load takes `K` and the two composite files. It requires the marks `dominant` 1 and
`orthogonal` 1, and the axes valence and arousal in that order. It refuses a diagonal value
of `K` under zero and a value that is not a figure. It refuses a composite file whose first
eight bytes are not `AOTXSTV1`. It refuses a file that names one layer twice. It refuses a
file whose width differs from the width of the language model.

`K` measures the composite vectors, which are the vectors a run applies. `K_raw` measures the
steer vectors of the store. The two agree when the composite of an axis is a copy of its own
steer vector. That happens when the two axes hold no common layer.

## The two streams

The derive list takes the names `affect` and `quality` in a build with the option. Each name
writes one file in the boot directory, beside `tokens.jsonl` and `pages.jsonl`. A drain built
without the option knows neither name and passes both record types over.

### The affect stream

`affect.jsonl` holds two kinds of line for each turn. A trace line gives the measurement of
the turn and is a class B record. A state line gives the state record of the turn and is a
class A record. The system writes both when `affect.on` was 1 at the open of the sequence of
that turn.

The turn node writes no record while a replay runs, so a replay adds no trace and no new
state record. The trace is derived, and the journal holds the turn. A restore applies each
state record again, and the stream then writes its line with `replayed` at 1.

```text
{"tick":14897,"agent":0,"turn":2,"kind":"trace","prompt":[0.785886168,3.07007861,0,0],"reply":[0.54256165,2.27193666,0,0],"guard":[-0.233722329,-1.13491237],"logprob":-0.314431012,"entropy":0.32180196,"rows":19,"think":0,"reason":["stop"],"effective":[0.278381348,0.999969482,0,0],"flags":11,"budget_spent":0.00487328041,"entropy_shift":0.079088971,"class_shift":0.000386928499}
```

| field | meaning |
| --- | --- |
| `tick` | the tick of the record |
| `agent` | the agent slot |
| `turn` | the turn that ended |
| `kind` | `trace` |
| `prompt` | the readouts of the last prompt row, axes 0 to 3 |
| `reply` | the mean readouts over the reply rows, axes 0 to 3 |
| `guard` | the mean sycophancy readout and the mean refusal readout over the reply rows |
| `logprob` | the mean log probability of the sampled tokens, in nats |
| `entropy` | the mean distribution entropy of the sampled tokens, in nats |
| `rows` | the reply rows behind the reply means |
| `think` | the thinking tokens of the turn |
| `reason` | the events of the turn, as words, in bit order |
| `effective` | the effective state after the update, axes 0 to 3, over 32768 |
| `flags` | bit 0 probe rows loaded, bit 1 composite applied, bit 2 budget scale under one, bit 3 cap hit |
| `budget_spent` | the divergence the composite applied at the open of the turn, in nats |
| `entropy_shift` | the mean entropy the temperature coupling added to a pick, in nats |
| `class_shift` | the mean probability the voice bias moved to its token class |

The last three fields are the figures of the three actuators. Each one states if its actuator
acted, so a control of that actuator has a figure beside it.

`budget_spent` is the divergence the composite dose applies after the budget scale. It is
half of the quadratic form of the dose, times the square of the scale. It is 0 for a turn
with no applied composite. It is `affect.budget` for a turn where bit 2 of the flags is set.

`entropy_shift` is a mean over the picks of the turn. Each pick gives the entropy of its
row at the applied temperature, less the entropy of the same row at the base temperature.
The base temperature is `decode.temperature` before the coupling. The device reads the same
adjusted logits for both entropies. The shift is thus the work of the coupling at the rows
the turn made, and not a second reply.

`entropy_shift` is 0 while `affect.temperature_gain` is 0. A greedy pick, which is a pick at
a temperature of 0, also states no shift.

`class_shift` is also a mean over the picks of the turn. Each pick gives the probability of
the biased token class at the applied voice scale, less the same probability at the plain
bias. It is 0 while `affect.voice_gain` is 0. It is 0 while the sequence names no voice
profile.

An axis with no loaded row reads 0. With no probe row loaded, `rows` is 0 and every readout is
0. With rows loaded, `rows` is the output tokens of the turn less one, because the system never
feeds the last sampled token as a row.

The state line of the same turn follows the trace line:

```text
{"tick":14897,"agent":0,"turn":2,"kind":"state","fast":[0.225494385,0.769317627,0,0],"slow":[0.0528869629,0.247283936,0,0],"scale":1,"reason":["stop"],"replayed":0}
```

| field | meaning |
| --- | --- |
| `tick` | the tick of the record |
| `agent` | the agent slot |
| `turn` | the turn that ended |
| `kind` | `state` |
| `fast` | the fast part of each axis, over 32768 |
| `slow` | the slow part of each axis, over 32768 |
| `scale` | the budget scale of the composite, over 65535, from the open of the sequence |
| `reason` | the events of the turn, as words, in bit order |
| `replayed` | 1 when a restore applied the record again |

The scale field holds 65535 for a scale of one, and the line then states 1. A turn that
applies no composite states 1, because the neutral scale is one.

The two lines above are the second turn of an agent. The sum of the two valence parts,
0.225494385 and 0.0528869629, is the `effective` valence 0.278381348 of the trace line. The
two arousal parts add to more than the cap of 1. The `effective` arousal is therefore the
cap, and bit 3 of the flags is set. The scale is 1, because the divergence of the dose stayed
under `affect.budget`. The trace line gives that divergence as 0.00487328041 nats.

The `reason` array of both lines names the events of the turn with these words:

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
made the call.

A scripted run needs the tool events in a known order, whatever the reply of the model holds.
The console line `outcome <ok|error|refused|none|call>` arms one tool result for the next turn
of the conductor agent. When that turn calls a tool, no request opens and no tool runs. The
armed result stands on the request slot as a request that completed at once. The next turn
takes it as a device tool result and writes the reply with the result in its context. The event
of the result fires on that next turn.

`none` takes the call of the turn off, with no result and no event. `call` completes a call the
turn makes as `ok`, and a turn with no call takes no result and no event. A turn with no call
under `ok`, `error` or `refused` takes the armed result as its own, and the event lands on that
turn. The line is for scripted runs. It is a class A input record, so a replay makes the result
again from the line, and no request record names it.

The stream counts a body it refuses and writes no line for it. It refuses a trace body with one
of these faults:

- an agent at 64 or above;
- a value that is not a figure;
- a negative entropy;
- a negative budget spent;
- a class shift outside -1 to 1;
- a bit above 14 in the event mask;
- a flag above bit 3.

It refuses a state body with one of these faults:

- a class that is not A, or a length that is not 36 bytes;
- an agent at 64 or above;
- a bit above 14 in the event mask;
- a flag above bit 3.

### The quality stream

`quality.jsonl` holds one line for each turn. The system writes the record of a turn whose
sequence opened while `quality.on` was 1.

```text
{"tick":3379,"agent":0,"turn":1,"coherence_prompt":0.818876922,"coherence_turn":null,"repetition":0,"tokens":33,"limit":256,"limit_hit":0,"refusal":0,"guard":[0.068339467,-0.433599025],"flags":9}
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

A turn that carries a tool result writes its quality line at its own turn end. While
`quality.on` is 1, the agent that made the call waits a fixed 8 ticks before it takes the
result. The rows of the call turn reach the embedding pass in that time. The count reads the
setting and its own tally alone, so a replay opens the next turn in the tick the live run did.
With `quality.on` at 0 the agent takes the result at once.

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
a hash of three token identities. A collision reduces the distinct count and can raise the reported repetition.

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

## The restore

The state record is a class A record of 36 bytes. Its body folds into the state hash, and a
restore applies it. The apply copies the two parts of the two data axes, the budget scale and
the two actuator flags into the state table. It sets the reserved axes to zero. A record that
names an agent at the slot count or above folds and changes no state.

A restore therefore starts each agent from the state its last record holds. The sampler row of
the next sequence follows from that state, the replayed settings and the loaded files. No
record carries the applied temperature or the applied voice scale.

`tests/replay_affect.sh` measures the restore of the state. The scenario runs two forms: one
agent, and every slot of the build. The conductor reads a file that is not there with the
`fs_read` tool, so a tool error event reaches the state. Each other agent takes a short task
with no tool. The scenario then kills the run inside a later turn.

The scenario proves these things:

- the journal holds one open of slot 0 with no manifest, so the kill landed inside a turn;
- the last state record of agent 0 before the kill holds a state that is not zero;
- the state records before the kill and the records the restore applied again agree;
- the replayed state lines of the stream count the same as the records applied again;
- the state hash before the kill equals the restore hash after it;
- each agent that wrote a record after its replayed ones went on at the next turn;
- the state of that next turn follows the law from the restored state;
- the restored run refused no sequence open.

The scenario sets `affect.on` to 1, `sample.temperature` to 0, `sample.seed` to 7 and
`decode.reply_limit` to 64. A sequence takes pages for its prompt and its reply limit at the
open. The reply limit keeps every slot of the build inside the page pool.

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
- the bodies of BOOT, TICK_START, TICK_COMMIT and STATS;
- the path field of the IMPORT head, which holds the build directory name.

It compares the order of the remaining lines and every other body byte for byte. It compares
`tokens.jsonl` and `pages.jsonl` byte for byte as well.

The clock record needs a mask because it is class A and folds into the state hash, so two live
runs never share that hash. The tick it lands in moves between two runs of one build, and the
counts of that tick move with it. The sequence numbers move with the dropped records.

The two builds can round differently under the contraction of the compiler, and a greedy
decode then parts at a near tie. When the dumps agree up to the first token record, the check
compares the kinds and counts of the class A records. An equal kind and count passes, and the
check prints the first differing tick and the log probability gap at it. A different kind or
count fails.

| exit code | meaning |
| --- | --- |
| 0 | the two pairs pass |
| 1 | a comparison failed, or a run failed |
| 2 | the arguments or the build programs are not usable |

A pass removes the work directory. A failure keeps it and names its path.

## The measured axes

These are retained historical reference measurements. They do not qualify the current source or new model/control packages.


The figures below come from the derivation tool, the calibration and the capability
instrument on the reference card. They are measurements of two model files on one machine,
and not properties of the design. Another file or another card gives other figures.

The layer list of each derivation run holds 8, 12, 16, 20 and 24. Every probe reads at the
layer 24, which is at or after every steered layer. The standardization set is
`plain-replies.txt` (an external capture, not bundled): 145 replies of the system with the substrate off.
They come from one run of the fixture conversation set at the seed 7 and the temperature 0.6,
with the call markup lines left out. The mean and the scale of each probe are the mean and the
scale of its cosines over that set.

| held-out figure | Q8_0 | Q4_0 | bound |
| --- | ---: | ---: | --- |
| valence accuracy / agreement | 1 / 0.969 | 1 / 0.938 | 0.8 / 0.9 |
| arousal accuracy / agreement | 1 / 0.719 | 1 / 0.813 | 0.8 / 0.9 |
| sycophancy accuracy / agreement | 0.969 / 0.313 | 0.938 / 0.25 | monitor |
| refusal accuracy / agreement | 1 / 0.656 | 1 / 0.719 | monitor |

The mean and the scale of each probe over the set are these cosines:

| probe mean / scale | Q8_0 | Q4_0 |
| --- | ---: | ---: |
| valence | -0.0071 / 0.0242 | 0.0065 / 0.0238 |
| arousal | -0.0281 / 0.0257 | -0.0143 / 0.0223 |
| sycophancy | 0.0360 / 0.0634 | 0.0279 / 0.0652 |
| refusal | 0.0708 / 0.0672 | 0.0720 / 0.0701 |

The vector of the valence axis takes the layer 20 on both files. The vector of the arousal
axis takes the layer 8 on both. The two axes therefore hold no common layer, and each
composite is a copy of its own steer vector.

| calibration at the dose 0.25 | Q8_0 | Q4_0 | bound |
| --- | ---: | ---: | --- |
| M valence, arousal, at their own axis | 3.02, 0.36 | 2.98, 0.42 | a dominant diagonal |
| M valence, arousal, at the other axis | 0.07, 0.02 | 0.07, -0.01 | under the diagonal |
| M sycophancy at valence, at arousal | 0.20, 0.03 | 0.23, 0.03 | a measurement |
| M refusal at valence, at arousal | -0.15, 0.02 | -0.15, 0.01 | a measurement |
| K valence, arousal, in nats per unit dose squared | 0.082, 0.095 | 0.062, 0.081 | a figure |
| K normalized off-diagonal | 0.208 | 0.103 | under 0.3 |
| dose-response valence, arousal | 4.71, 3.98 | 4.35, 3.91 | 3 to 5 |
| perplexity ratio valence, arousal | 1.038, 1.002 | 1.026, 0.998 | under 2 |

The sycophancy row of M gives the movement of that guard for one unit of dose. At the dose
0.25 the movement is 0.05 on the Q8_0 file and 0.06 on the Q4_0 file, under the bound of 0.5.

The capability instrument scores 200 four-choice items at three doses of one axis:

| capability score | dose 0 | dose 0.25 | dose 0.5 |
| --- | ---: | ---: | ---: |
| Q8_0 valence | 0.800 | 0.795 | 0.805 |
| Q8_0 arousal | 0.800 | 0.790 | 0.795 |
| Q4_0 valence | 0.830 | 0.845 | 0.830 |
| Q4_0 arousal | 0.830 | 0.820 | 0.825 |

Each score at the dose 0.25 stands within 0.05 of the score of the same axis at the dose 0.

## The measured actuators

The temperature coupling and the budget are measured on a fixture feed. The temperature gain
is 0.5. Five effective arousal values from -0.195 to 0 give a temperature of 0.7219 to 0.8.
The pick entropy of those five states is 0.000189 to 0.000464 nats. The entropy rises with
the arousal on every step, and the base temperature of 0.8 comes back at the arousal 0.

At a budget of 0.0001 nats, three states give budget scales of 0.757, 0.484 and 0.397. The
applied divergence of each is 0.0001 nats within one part in ten thousand, and each trace
states bit 1 and bit 2. The entropies of the fixture feed are small, because its tokens are
almost certain at the temperature 0.8.

## Two findings of the measurement

The agreement bound is 0.9. The valence axis reaches it on both files: 0.969 on the Q8_0 file
and 0.938 on the Q4_0 file. The arousal axis stands under it on both files, at 0.719 and
0.813. That is a stated limit of this version. `affect.probe_gain` stays 0, so no readout
enters the drive of the update, and the substrate runs on events alone. The guard rows stay
monitors, as the loader always makes them.

The reply readout is the mean of the cosine readouts over the rows of the reply. Its zero
point is the mean cosine over the replies of the system with the substrate off. A live run of
the Q8_0 file at the temperature 0.6 shows the effect. A sad story of 111 rows reads -0.77,
and its happy inverse of 111 rows reads -0.03. The short replies of the same run, of 9 to 40
rows, read +0.3 to +1.3. The one reply after a tool error reads -0.46, and the three call
turns of the run read -0.11 to +0.62.

The guard rows of the run read across zero. Sycophancy reads -0.48 to +1.16, and refusal reads
-1.20 to +0.11. The earlier set, `tests/fixtures/affect/neutral-replies.txt`, puts each of
those readouts higher. The two stories read +0.08 and +0.71 there, and the short replies +1.1
to +2.2. Sycophancy reads +1.19 to +3.91 there, and refusal -0.19 to +1.66. The replies of the
system therefore give the zero point of this version, and the earlier set stays a fixture of
the axis checks.

The mean over rows does not follow the length of the reply. A reply of 178 rows reads +0.29,
and a reply of 83 rows reads +0.40. The order of the two stories is the sign split the probe
exists for. The order of a pair does not depend on the zero point, because the accuracy is 1
on both measured axes of both files. The calibration figures do not depend on it either,
because every pass of a calibration run reads the texts of one set.

## Current limits

Dominance and certainty remain reserved axes.
An absent probe is unavailable; zero in a raw trace does not establish a measured neutral value.
Guard probes remain monitors and never drive state.

The derived affect and quality readers currently reject agent IDs at 64 or above.
Larger profile compilation does not extend that stream coverage.
The current runtime qualification remains bounded by [support and qualification](support.md).

Trace records include budget spent, entropy shift and class shift when their mechanisms run.
They do not record the applied temperature or voice scale directly.
A valid binding proves byte identity, while its qualification evidence must establish the claimed behavioral effect.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
