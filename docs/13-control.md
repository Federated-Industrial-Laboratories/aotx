<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Graphical control client

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

`aotx_ctrl` creates, attaches to and stops local instances through a graphical interface.
It reads journal replicas and the display mirror, then sends explicit console commands through the attach socket.
CUDA retains runtime authority. Application clients can instead use the separate [HTTP gateway](31-http-gateway.md).

<details>
<summary>On this page</summary>

- [Start it](#start-it)
- [The windows](#the-windows)
- [The Trace window](#the-trace-window)
- [The Dials window](#the-dials-window)
- [The instance phases](#the-instance-phases)
- [The attach](#the-attach)
- [The speech engine](#the-speech-engine)
- [The build](#the-build)
- [Conversation tool controls](#conversation-tool-controls)

</details>

## Start it

Run from the repository root after building:

```sh
build/aotx_ctrl --journal build/run --settings aotx.settings
```

Create the journal directory first when supplying `--journal`.
The complete command form is:

```text
aotx_ctrl [--journal DIR] [--settings FILE] [--sim] [--frames N] [--help]
```

| option | effect |
| --- | --- |
| `--journal <dir>` | the journal directory of the system to control |
| `--settings <file>` | the settings file; its `journal.dir` key names the journal directory |
| `--sim` | run against simulated data, with no system |
| `--frames <n>` | stop after this count of frames; the smoke check uses it |
| `--help` | print the usage text and stop |

Without an explicit journal, the settings file supplies its directory.
Otherwise, the client uses its saved journal under `$XDG_CONFIG_HOME/aotx/journal`, or `~/.config/aotx/journal`.
If no saved journal exists, it creates one under `$XDG_DATA_HOME/aotx`, or `~/.local/share/aotx`.
The saved layout is `ctrl-layout.ini` beside that journal-name file.

## The windows

Use the Windows menu to show or hide panels. Use View to rebuild the layout.

| window | content |
| --- | --- |
| Instances | the known systems, their phase words and the Start, Restore, Attach, Stop and Remove controls |
| Control | the tick figures, the requests of the agents and the Start, Restore, Stop, Grant and Refuse controls |
| Models | catalog and local files, Inspect, Fetch, Activate, Load, role assignments, model controls, presets, and conduct |
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

A conductor conversation sends `say` and `continue` commands.
Worker conversations use the corresponding agent commands.
New creates a worker conversation; it does not replace the conductor.
Model inspection reports header compatibility only, not reply quality or successful runtime execution.

Stop an instance before starting a replacement boot or restoring its journal.
Wait for its closed state before Restore.
Recorded device settings take precedence during recovery; editing the settings file does not rewrite those records.
Keep the required exact model and executable module assets available.

## The Trace window

Trace reads `affect.jsonl` and `quality.jsonl` from the selected boot.
It joins up to 32 retained turns per agent from each stream.
These are recorded measurements, not a read of live private GPU memory.

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

A dash marks an absent measurement. The plots show available readouts and effective state over retained turns.
The stream's recorded enable state determines the display.
An empty stream cannot establish that a configured measurement ran.
The Monitor panel uses available quality rows for coherence and repetition means.

The client presents runtime control values; it does not infer a person's emotional state.
A build without `AOTX_AFFECT` omits Trace and its menu entry.

## The Dials window

Dials exposes the thirteen affect and quality settings with their allowed ranges.
Enable flags use checkboxes; numeric settings use bounded controls.
Measurements beside those controls come from the selected model store and recorded trace.

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

Missing measurements disable their related controls and produce an explanatory message.
Apply sends `set` commands to a running instance or writes the stopped instance's settings file.
Device changes affect the next sequence, not a reply already in progress.
Reset restores defaults without changing unrelated configuration.

A visible calibration value does not establish a qualified numerical effect.
The loader and GPU still enforce exact [control bindings](37-control-bindings.md) and accepted settings.
An OFF build omits Dials.

## The instance phases

The journal's `phase` file supplies startup progress.
Placement and replay entries remain live for 30 seconds after their file update.
Running state requires an answering attach socket or an owned child process.
A stale running entry without either is shown as stopped.

Essential child supervision belongs to the runtime.
A writer, feeder or broker failure must be read as a failed run, not a successful saved state.
[Operation](07-operation.md#how-a-run-stops) describes current shutdown and recovery behavior.

## The attach

The client connects to `<journal>/aotx.sock` and receives a read-only mirror descriptor.
It retries a lost connection every two seconds.
Repeated identical refusals produce one retained error instead of repeated notices.
The feeder restricts attachment to the same operating-system user.

## The speech engine

Voice uses `piper`, `pw-play` and at least one ONNX voice under `~/.local/share/piper-voices`.
Missing requirements disable speech controls and show the reason.
Closing the client discards queued lines and ends current synthesis.

An affect build adds optional spoken voice coupling, initially off.
It changes synthesis parameters from the last recorded effective valence and arousal:

```text
length scale = clamp(1 - 0.25 * arousal, 0.75, 1.25)
noise scale = clamp(0.667 + 0.167 * arousal, 0.5, 0.834)
noise width = clamp(0.8 + 0.1 * valence, 0.7, 0.9)
```

Both input values are bounded from -1 through 1.
Coupling retains the selected speaker file and replaces the manual length scale for that utterance.
The client logs applied values to standard error. Test uses agent zero's state when coupling is enabled.

Spoken coupling changes the host speech engine.
A sampler voice profile changes token bias on the GPU; these are separate controls.
Neither mechanism establishes a validated psychological measure.

## The build

`AOTX_CTRL=ON` builds the client with GLFW, OpenGL and vendored ImGui.
`AOTX_AFFECT` controls Trace, Dials, quality means and spoken coupling.
See [build requirements](06-build.md#requirements) for host libraries.

`ctrl_fix` checks client fixes. `ctrl_smoke` runs simulated data with a bounded frame count.
A simulated smoke check does not establish live conversation or recovery acceptance.
The third-party source under `ctrl/vendor/` retains its own text and licenses.

## Conversation tool controls

Open **System tools** to inspect instance defaults, conversation overrides and effective next-turn availability.
Choose **Inherit**, **Off** or **On** for each group.
An explicit conversation choice overrides the default; role grants and mandatory authorization still constrain execution.
Imported tools share one group.

The table reads typed `tools.jsonl` records, not generated text.
Use **Refresh** after a model or role change.
A lost connection retains one query status until refresh or reconnection.

A prompt-capacity refusal appears as an input error and remains in the conversation log.
Use a shorter input or a supported larger capacity, then submit a new message.
[Tool selection](09-modules.md#tool-selection) and [model files](16-model-files.md) describe the underlying runtime behavior.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
