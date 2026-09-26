<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Settings reference

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Settings define startup paths, terminal presentation and device behavior.
A file contains one `key = value` pair per line. A `#` starts a comment.
Numeric values are integers or fixed decimals with at most four decimal places.

<details>
<summary>On this page</summary>

- [Select a file](#select-a-file)
- [Startup and terminal settings](#startup-and-terminal-settings)
- [Device settings](#device-settings)
- [Affect settings](#affect-settings)
- [Change a running instance](#change-a-running-instance)

</details>

## Select a file

Use `--settings FILE` to name a required settings file.
An explicitly named missing file stops startup.
Without that option, boot looks beside the journal directory, or in the current directory when no journal path is given.
An absent default file uses built-in defaults.

The startup log names the file it reads. Invalid lines produce reasons; other valid lines remain available.
An unreadable file stops startup. Explicit command-line options take precedence over corresponding file values.

```ini
journal.dir = build/run
models.dir = models
models.roles = language
tui.on = 1
tick.period_ms = 10
sample.temperature = 0.6
```

## Startup and terminal settings

These values are local configuration. They do not become device SETTING records.
Startup values apply when boot selects its paths and surfaces.
Terminal values apply when the terminal reads its file.

| Key | Default and range | Purpose | Applies |
| --- | --- | --- | --- |
| `journal.dir` | `journal` | journal directory | at the start |
| `models.dir` | `models` | model-store directory | at the start |
| `models.roles` | empty | model roles to load | at the start |
| `modules.dir` | `modules` | directory that holds module directories | at the start |
| `tools.root` | empty | root directory of host file tools | at the start |
| `derive.list` | empty | derived journal outputs | at the start |
| `window.on` | 0; 0 to 1 | start the window | at the start |
| `tui.on` | 0; 0 to 1 | start `aotx_tui` | at the start |
| `tui.escape_ms` | 25; 5 to 500 | wait before a lone Escape is accepted | when the terminal reads the file |
| `tui.color` | `none` | terminal color form | when the terminal reads the file |
| `tui.box` | `ascii` | terminal box form | when the terminal reads the file |
| `tui.splash` | `auto` | terminal splash form | when the terminal reads the file |

## Device settings

Device settings become class A SETTING records. Restore applies those recorded values.
A change takes effect at the boundary shown below, not midway through an active sequence.

| Key | Default and range | Purpose | Applies |
| --- | --- | --- | --- |
| `tick.period_ms` | 10; 1 to 1,000 | milliseconds between ticks | the next tick |
| `decode.budget_ms` | 120; 10 to 10,000 | decode allowance read by the check; no run node consumes it | the next tick |
| `decode.prefill_tokens` | 512; 32 to 512 | prompt tokens admitted in one tick | the next tick |
| `decode.reply_limit` | 256; 1 to 8,191 | reply tokens for a sequence | the next sequence |
| `decode.auto_continue` | 0; 0 to 1 | resume a limited reply until its natural stop | the next tick |
| `sample.temperature` | 0; 0 to 2 | sampling temperature; zero selects the largest logit | the next sequence |
| `sample.top_p` | 1; 0.0001 to 1 | top probability mass | the next sequence |
| `sample.top_k` | 0; 0 to 256 | candidate token count; zero keeps all candidates | the next sequence |
| `sample.min_p` | 0; 0 to 1 | least probability relative to the largest | the next sequence |
| `sample.repeat_penalty` | 1; 0.0001 to 2 | penalty for a token in the repeat window | the next sequence |
| `sample.repeat_window` | 0; 0 to 8,191 | recent tokens checked for repetition | the next sequence |
| `sample.presence_penalty` | 0; -2 to 2 | penalty when the sequence contains the token | the next sequence |
| `sample.frequency_penalty` | 0; -2 to 2 | penalty for each use of the token | the next sequence |
| `sample.seed` | 0; 0 to 2,147,483,647 | fixed sample seed; zero derives one for the turn | the next sequence |
| `decode.think_limit` | -1; -1 to 8,191 | thinking tokens; -1 gives no limit and zero forbids the span | the next sequence |
| `agent.budget` | 8; 1 to 64 | turns per task or operator input, including automatic continuation | the next input |
| `agent.pages` | 0; 0 to 4,096 | default hot page limit; zero takes the profile maximum | the next task |
| `agent.recall_k` | 4; 0 to 16 | warm turns recalled into a prompt | the next task |
| `agent.compact_at` | 128; 8 to 1,024 | warm turns that start compaction | the next task |
| `tool.deadline_ticks` | 500; 1 to 1,000,000 | ticks allowed after a request or grant | the next request |
| `mirror.hz` | 30; 1 to 120 | mirror snapshots in one second | the next frame |
| `tools.mask` | 1023; 0 to 1023 | instance tool selection bits | the next sequence |

## Affect settings

These settings exist only with `AOTX_AFFECT=ON`. Both runtime enable flags default to zero.
A sequence retains the law selected when it opens. Later changes affect later sequences.

| Key | Default and range | Purpose | Applies |
| --- | --- | --- | --- |
| `affect.on` | 0; 0 to 1 | the substrate measures a turn, updates the state and applies it | the next sequence |
| `quality.on` | 0; 0 to 1 | the quality instrument measures a turn and writes its record | the next sequence |
| `affect.probe_gain` | 0; 0 to 1 | weight of the readouts in the drive of the update | the next sequence |
| `affect.decay_fast` | 0.5; 0 to 0.99 | decay of the fast part of the state | the next sequence |
| `affect.decay_slow` | 0.9; 0 to 0.99 | decay of the slow part of the state | the next sequence |
| `affect.gain_fast` | 0.5; 0 to 2 | drive gain of the fast part of the state | the next sequence |
| `affect.gain_slow` | 0.1; 0 to 2 | drive gain of the slow part of the state | the next sequence |
| `affect.cap_valence` | 1; 0 to 1 | cap of the effective valence | the next sequence |
| `affect.cap_arousal` | 1; 0 to 1 | cap of the effective arousal | the next sequence |
| `affect.temperature_gain` | 0; -1 to 1 | temperature change for one unit of effective arousal | the next sequence |
| `affect.voice_gain` | 0; -1 to 1 | voice bias scale for one unit of effective valence | the next sequence |
| `affect.steer_gain` | 0; 0 to 1 | dose of the composite steer for one unit of effective state | the next sequence |
| `affect.budget` | 0.25; 0 to 4 | largest divergence one turn applies, in nats | the next sequence |

## Change a running instance

Enter `settings` to inspect current values and their application boundaries.
Use `set KEY VALUE` for a device setting:

```text
set sample.temperature 0.6
set decode.reply_limit 512
```

Per-agent decode overrides use the [console commands](commands.md).
The `decode.budget_ms` setting is retained for its check; it does not enforce a runtime execution deadline.
The reply limit must still fit the configured sequence and page capacities.

The authoritative key definitions are in `cuda/settings/keys.h`.
See [affect](14-affect.md) for state equations and [tool selection](09-modules.md#tool-selection) for effective tool masks.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
