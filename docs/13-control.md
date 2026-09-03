# The control program

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| instance | one system with its own journal directory, started or attached by the control program |
| replica | the disk files of a journal directory, as the control program reads them |

`aotx_ctrl` is the graphical control program. It starts, attaches to and stops systems, and it
shows their state in windows. It reads the replica of each journal directory, and it sends the
console commands over the attach socket. It uses GLFW, OpenGL and the vendored ImGui sources.

## Start it

Run the program from the build directory:

```
build/aotx_ctrl [--journal <dir>] [--settings <file>] [--sim] [--frames <n>] [--help]
```

| option | effect |
| --- | --- |
| `--journal <dir>` | the journal directory of the system to control |
| `--settings <file>` | the settings file; its `journal.dir` key names the journal directory |
| `--sim` | run against simulated data, with no system |
| `--frames <n>` | stop after this count of frames; the smoke check uses it |
| `--help` | print the usage text and stop |

With no journal option, the settings file names the journal directory. Without one, the program
opens the last journal it bound, which it keeps in `$XDG_CONFIG_HOME/aotx/journal` or
`~/.config/aotx/journal`. With no such journal, it makes a new one in `$XDG_DATA_HOME/aotx` or
`~/.local/share/aotx`. The window layout is kept in `ctrl-layout.ini` beside the journal name.

## The windows

The Windows menu opens and closes each window. The View menu rebuilds the layout.

| window | content |
| --- | --- |
| Instances | the known systems, their phase words and the Start, Attach, Stop and Remove controls |
| Control | the tick figures, the requests of the agents and the Start, Stop, Grant and Refuse controls |
| Models | the catalog with Fetch and Use, the role assignments, the model controls, the presets and the conduct |
| Modules | the imported skills, roles and tools, and a directory import |
| Sync | the module and voice profile files whose disk copies changed, and a Sync act for each |
| Settings | the settings file, with Save for a file key and Apply for a device key of a running system |
| Monitor | the tick, ring, memory, page map and agent figures of the selected system |
| Trace | the affect readouts and the quality figures of the last turns of each agent |
| Dials | the thirteen affect and quality settings as controls, with the figure that instruments each |
| Transcripts | the stored runs and the transcript of each conversation |
| Voice | the speech engine controls and the voice of each agent |
| First run | the six pages Detect, Build, Model, Activate, Start and First say |
| conversation | one chat with one agent; New conversation in the Windows menu opens one |

A conversation sends `say` lines to the console. Its Continue control resumes a reply that
ended at its reply limit. The conversation of agent 0 sends `continue`. A worker conversation
sends `agent <id> continue`. The New control of a conversation starts a worker conversation.

## The Trace window

The Trace window shows the two streams of `docs/14-affect.md`. It reads `affect.jsonl` and
`quality.jsonl` of the boot directory of the selected system, as a client of the replica. It
holds the last 32 turns of each agent from each stream. A build without the `AOTX_AFFECT`
option has no such window and no item for it in the Windows menu.

An agent selector at the top names the agents the two streams hold. The table below it joins
the two streams by agent and turn, and gives one row for each turn:

| column | content |
| --- | --- |
| Turn | the turn number |
| Prompt valence, Prompt arousal | the readouts of the last prompt row |
| Reply valence, Reply arousal | the mean readouts over the reply rows |
| Sycophancy, Refusal | the two guard readouts |
| Events | the events of the turn, as words |
| Logprob, Entropy | the turn means |
| Effective | the four applied state values |
| Coherence | the prompt figure and the prior-turn figure |
| Repetition | the repeated token-trigram share |

A cell with no figure behind it shows a dash. Beside the table stands one explaining sentence
for each group of columns. Below the table stand eight line figures over the held turns: the
four readouts and the four effective state values, one series for each. They are figures
alone, with no face and no color that carries a meaning.

The window takes the on state and the off state from the lines the streams hold, because it
reads the replica and not the settings. A system whose first turn has not ended shows an off
sentence until its first line lands. The window shows one of these sentences, dim:

- The affect substrate and the quality stream are off. Set affect.on or quality.on to 1 to start one.
- The affect substrate is off. Set affect.on to 1 to start it.
- The quality stream is off. Set quality.on to 1 to start it.

The first sentence stands alone when both streams hold no line, and the table and the figures
are absent. The second and the third stand above the table, and the columns of the stream that
is off are dim.

The Monitor window states the mean prompt coherence and the mean repetition of each agent,
over the quality lines it holds. A dash stands for a mean with no figure behind it. With no
quality line, the Monitor window shows the third sentence above.

## The Dials window

The Dials window holds the thirteen settings of `docs/14-affect.md` as controls. `affect.on`
and `quality.on` are check boxes. The eleven other settings are sliders with the bounds of the
settings table. One explaining sentence stands beside each control, and the figure that
instruments the setting stands under the sentence. A build without the `AOTX_AFFECT` option has
no such window and no item for it in the Windows menu.

| control | figure beside it | source |
| --- | --- | --- |
| `affect.on`, `quality.on` | the value in the settings file | the settings file |
| `affect.probe_gain` | the accuracy of each probe | `<models>/probes.jsonl` |
| `affect.decay_fast`, `affect.decay_slow`, `affect.gain_fast`, `affect.gain_slow` | the effective valence and arousal of the last turn | the last trace line of `affect.jsonl` |
| `affect.cap_valence`, `affect.cap_arousal` | the last effective value of that axis | the last trace line |
| `affect.steer_gain` | the two diagonal values of K and the two dose-response ratios | the last line of `<models>/affect/calibration.jsonl` |
| `affect.temperature_gain` | the entropy shift of the last turn, in nats | the last trace line |
| `affect.voice_gain` | the class shift of the last turn | the last trace line |
| `affect.budget` | the budget spent of the last turn, in nats | the last trace line |

The window reads the model store that `models.dir` of the settings file names. The trace line
is the last line of the selected agent, and the window names that agent above the table.

A control whose figure is absent is off, and the sentence "No calibration figure is loaded for
this control." stands under it. A trace line of an earlier version carries no actuator
figure. The three actuator controls stay off until a line of this version reaches the
stream.

Apply sends the value of every control through the one path of a setting. That path is a
`set` line to a running system, and the settings file of a stopped one. A device key so set
takes effect at the next sequence that opens. Reset sets every control to its default and
changes nothing else.

The window states one of five sentences at its top, from its two check boxes and the loaded
figures:

- The affect substrate and the quality stream are off. Set affect.on or quality.on to 1 to start one.
- No calibration or trace figures are loaded, so the related controls are gray.
- Calibration figures are loaded, but trace figures are absent, so trace controls are gray.
- Trace figures are loaded, but calibration figures are absent, so calibration controls are gray.
- Calibration and trace figures are loaded beside their controls.

The first sentence stands while both check boxes of the window are off.

## The instance phases

The phase word of an instance comes from the `phase` file of its journal. A phase word `placing`
or `replaying` is live for 30 seconds after its file was written. A phase word `running` is live
while the attach socket answers or while the program owns the child. A `running` word with no
socket and no child is stale: the instance shows as stopped, and a Start is permitted.

## The attach

The feeder owns `<journal>/aotx.sock`. The program connects there, receives a read-only mirror
descriptor and reads the newest complete frame. A lost connection is stated once, and a retry
starts every 2 seconds. A refusal of the attach, for example when the system runs for another
user, is stated once as an error. The same refusal from a later retry is not stated again.

## The speech engine

The Voice window speaks the replies and the lifecycle lines. The engine needs the programs
`piper` and `pw-play` on the search path, and at least one `.onnx` voice file in
`~/.local/share/piper-voices`. Without one of them the window states the refusal and the
controls are off. A close of the program drops the queued lines and ends the line in synthesis.

The Voice window holds the `Couple` control in a build with the `AOTX_AFFECT` option. It is off
by default. With it on, the spoken voice coupling sets three controls of the speech engine for
each reply of an agent. The three are the length scale, the noise scale and the noise width.
They come from the effective state of the last trace line of that agent, with fixed gains and
caps:

```text
length scale = clamp(1 - 0.25 * arousal, 0.75, 1.25)
noise scale = clamp(0.667 + 0.167 * arousal, 0.5, 0.834)
noise width = clamp(0.8 + 0.1 * valence, 0.7, 0.9)
```

The valence and the arousal are the effective values, each held to the range -1 to 1. A
coupled line takes its length scale from the coupling and not from the Speech rate control.
The engine writes one line on the error output for each coupled utterance, with the values
applied. The line has this form:

```text
Spoken voice coupling: agent 0; length 0.9872; noise 0.6756; width 0.8286; valence 0.2857; arousal 0.0513.
```

The coupling changes no speaker: the voice file of each agent stays as the window assigns it.
The Test control speaks with the state of agent 0 while the coupling is on.

The spoken voice coupling is a mechanism of the control program, outside the device. The voice
bias of `docs/14-affect.md` is an actuator of the sampler, which scales the bias of a voice
profile. The two are different things, and this document keeps the two names apart.

## The build

The option `AOTX_CTRL` builds the program; it is ON. The option `AOTX_AFFECT` is ON. It adds the
Trace window, the Dials window, the quality means of the Monitor window and the `Couple`
control of the Voice window. The build needs pkg-config and
GLFW 3, as `docs/06-build.md` lists. The check `ctrl_fix` runs the fix cases, and the check
`ctrl_smoke` runs the program with `--sim --frames 300`. The directory `ctrl/vendor/` holds
the ImGui sources, and the gates do not read it.
