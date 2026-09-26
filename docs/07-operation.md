<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Operation

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Use one journal directory for each local instance.
A normal run owns a CUDA context, device state, disk writer, feeder and optional clients.
The runtime supervises essential child processes until shutdown completes.

Prepare a checked model store and a compatible build before starting.

[Model files](16-model-files.md) covers existing GGUF files and catalog downloads.
[Support](support.md) states which optional behaviors have qualified model packages.

<details>
<summary>On this page</summary>

- [Start a run](#start-a-run)
- [The settings file](#the-settings-file)
- [The disk-side programs](#the-disk-side-programs)
- [The window](#the-window)
- [The command line](#the-command-line)
- [Replies and agents](#replies-and-agents)
- [The model store](#the-model-store)
- [Conversation memory](#conversation-memory)
- [Tool call forms](#tool-call-forms)
- [Tool requests and file reads](#tool-requests-and-file-reads)
- [The journal a run leaves](#the-journal-a-run-leaves)
- [Restore](#restore)
- [Load runs and the derive list](#load-runs-and-the-derive-list)
- [How a run stops](#how-a-run-stops)
- [Failures and recovery](#failures-and-recovery)
- [The watchdog](#the-watchdog)

</details>

## Start a run

From the repository root:

```sh
build/aotx_boot --journal build/run --models models --roles language --tui
```

Use `--window` for the GPU window, or omit both display options for a headless run.
A headless run still uses the normal CUDA build and disk programs.
The window path must use a GPU compatible with the active display context.

Boot prints its selected settings path, clock sample, runtime identity and memory admission results.
Model startup verifies file digests before use.
A model or allocation failure reports its reason and stops startup.

The default tick period is 10 ms. The configured period sets a pacing target, not a guarantee that all work finishes within it.
A run without an explicit tick count stops at its configured record bound, which defaults to 1,000,000 records.

### Boot options

| Option | Purpose |
| --- | --- |
| `--journal <dir>` | the directory the journal goes in |
| `--models <dir>` | the directory the model files are in |
| `--roles <list>` | roles of the model file list to load, with commas between them |
| `--root <dir>` | the one directory a file read tool may reach |
| `--modules <dir>` | the directory of module directories to import at the start |
| `--restore` | replay the journal before the first input |
| `--window` | show the panels in a window on the display |
| `--tui` | start the terminal program beside the system |
| `--tui-attached` | a terminal started this run and is attached to it already |
| `--ticks <n>` | run this many ticks, then stop; zero runs on |
| `--workload <n>` | records the tick load writes for each tick |
| `--blocks <n>` | blocks of the tick load; the default is 64 |
| `--records <n>` | stop a run that has no tick count at this record count |
| `--derive <list>` | types the drain makes lines from, with commas between them |
| `--settings <file>` | the settings file; the default is `aotx.settings` beside the journal directory |
| `--solo` | run with no disk-side programs |
| `--clock-only` | run the clock module check and stop |
| `--version` | print the version, the profile, the architecture and the slots, then stop |
| `--ccir <file>` | activate and maintain a complete runtime file |
| `--memory-mirror <file>` | maintain a memory checkpoint file |
| `--service-grants <file>` | enable the local service with a binary grant table |
| `--policy <file>` | select a resident creator-policy bundle |
| `--policy-trust <sha256>` | trust this exact native policy revision |

Normal operation requires a journal. `--solo` starts no disk writer or feeder and is intended for bounded checks.
Do not use it when durable state is required.
The settings file can supply paths and startup surfaces; explicit command-line values take precedence.

## The settings file

Create a settings file with one assignment per line:

```ini
journal.dir = build/run
models.dir = models
models.roles = language
tui.on = 1
```

Start with `build/aotx_boot --settings aotx.settings`.
An explicitly named missing file is an error.
An absent default file uses built-in defaults.
The [settings reference](settings.md) lists every key, default and application boundary.

## The disk-side programs

The boot program finds its child executables beside itself.
The drain writes and synchronizes journal blocks. The feeder publishes input and host-tool results.
Restore supplies previously recorded authoritative inputs.
An enabled service broker transports requests without owning their device state.

The runtime treats an essential child exit during execution as a failure, including an unexpected exit code of zero.
It reports the child and status, stops owned work and returns a failure status.
Closing an optional terminal client does not itself fail the runtime.

## The window

The GPU window contains six panels:

| Panel | Contents |
| --- | --- |
| Console | Recent output and the editable command line. |
| Agents | Active agent rows and pending authorization requests. |
| Bus | Recent messages, newest first. |
| Arena | Mapped regions, reserved memory, page use and available memory. |
| Tick | Tick, record, decode and agent counters. |
| Seam | Ring cursors, durable lag, held ticks and dropped output. |

The grid has 160 columns and 50 rows. Printable ASCII input supports editing, history and a four-row expanding command area.
Use Alt+Enter for a line break and Enter to submit.
Tab moves focus between the console and agents panel.
In the agents panel, `y` grants the first pending request and `n` refuses it.

Closing the runtime window requests normal shutdown.
The separate [control client](13-control.md) provides instance management and conversation windows.

## The command line

Enter `help` for the main commands. Common operations are:

```text
say explain the current task
agents
agent 0
stop
settings
quit
```

[Console commands](commands.md) documents task assignment, tool authorization, model replacement and specialized memory controls.
Input accepted through the terminal and window reaches the same device parser.

## Replies and agents

Agent zero is the conductor. The supplied role modules also define workers and verifiers.
The profile sets available agent slots; a task uses eight turns by default.
Tool follow-on turns and automatic continuation consume that same budget.
A new input or explicit continuation starts a new budget.

A language reply uses the loaded model's checked turn wrapper and current sampling settings.
The console streams printable reply text and records the final status.
A second input to a busy conductor is refused.
Stopping a reply ends it at a token boundary and retains a stopped transcript entry.

A model can repeat a fact from its prompt without running a memory tool.
To check a memory-tool operation, inspect matching call and result entries in the transcript.
Memory tools require a ready embedding role; ordinary language replies remain available without it.

## The model store

Fetching, activation and resident loading are separate operations.
The [model store](16-model-files.md#the-model-store) defines them and their file identities.

### Use another model file

Follow [local model selection](16-model-files.md#use-another-model-file) before loading a replacement.
A changed model can invalidate wrappers, fitted controls or automatic-memory qualification.

### Turn wraps

The [turn-wrap contract](16-model-files.md#turn-wraps) defines text spans and accepted model metadata.
Tool call formats use a separate exact-template selection; a text-wrap override does not enable tools.

## Conversation memory

Unbound local conversations use ordered transcripts with hot turns, recalled warm turns and a recorded summary.
Each prompt records its selected turns and page limit.
Restore applies the selection without repeating semantic search.

Hot turns fit within the agent's page bound. Older turns receive embedding vectors for warm recall.
Compaction summarizes the oldest eligible warm range and retains its vector references.
The oldest folded text can leave the circular text store when capacity is exhausted.

[Live memory bindings](20-live-memory.md) instead use selected typed objects and the current input.
The transcript remains an audit record.
[Semantic memory](27-semantic-memory.md) adds qualified source interpretation and corrections.
[Appraisal](35-automatic-appraisal.md) and [task reviews](38-task-reviews.md) remain separately controlled optional consumers.

| Profile | Default pool pages | Default pages per slot | Default sequence tokens | Transcript text per agent |
| --- | ---: | ---: | ---: | ---: |
| `8g` | 512 | 148 | 2048 | 64 KiB |
| `12g` | 1024 | 640 | 2048 | 256 KiB |
| `24g` | 4096 | 320 | 4096 | 1 MiB |
| `48g` | 12288 | 640 | 8192 | 4 MiB |

These are profile defaults; explicit build overrides can change context and cache bounds.
Tokens per page depend on the active model. A larger prompt also needs enough shared physical pages.
See [build capacity](06-build.md#context-and-cache-capacity).

## Tool call forms

The complete model template selects one bounded tool protocol.
The same row controls instructions, parsing, stored calls and result rendering.
Unknown templates disable tool advertisement and parsing while permitting otherwise valid text conversation.

| Form | Accepted call | Result rendering |
| --- | --- | --- |
| `hermes` | Tagged JSON with `name` and `arguments`. | User turn with `tool_response` tags. |
| `llama-json` | Bare JSON with `name` and `parameters`. | `ipython` turn with a JSON string. |
| `qwen-xml` | Function and parameter tags inside `tool_call`. | User turn with `tool_response` tags. |
| `none` | No tool call. | Historical calls remain text. |

The bare JSON form permits one leading `<|python_tag|>` marker, not executable Python.
Tool names and argument keys remain exact when catalog entries change.
[Modules](09-modules.md) and the [tool SDK](10-tool-sdk.md) define authority and execution.

## Tool requests and file reads

A host-tool request reaches the feeder only after its required authorization.
Its deadline starts at admission or operator grant, according to the tool policy.
A late result becomes a recorded device verdict.
Restore applies an already recorded reply without executing the host tool again.

Built-in file tools use the configured root and refuse absolute paths, parent traversal and symbolic-link components.
They require supported file types and bounded results.
`fs_read` includes a digest of served bytes; `fs_stat` reports metadata and a digest without returning file contents.
A trusted external host tool has the account's rights and is not confined by that file-tool root.

## The journal a run leaves

Each boot creates a directory containing numbered segment files and derived output.
The journal also contains bulk payloads, bus output, turn manifests and host-tool requests where enabled.
The journal is the recovery source; most derived files are inspection outputs.
The feeder consumes the requests file for admitted host tools.

Inspect a completed run with the built programs:

```sh
build/aotx_restore --journal build/run --summary
build/aotx_journal manifest build/run
build/aotx_journal requests build/run
```

Turn manifests form a SHA-256 chain. Their verification reports a changed or missing line.
The [journal format](04-journal-format.md) defines exact layouts, checksums and derived files.

## Restore

Stop the original instance before restoring its journal:

```sh
build/aotx_boot --journal build/run --models models --roles language --restore --tui
```

Restore selects the newest boot with complete durable ticks and reapplies its authoritative records.
It retains recorded token and input order without resampling completed outputs.
The replay tick loop is not paced by the normal tick period.
Its elapsed duration is not equal to the original run duration.

The completion report includes records, state hash, decode refusals, rejected input, pages and paced ticks.
A nonzero rejected-input count stops boot.
A reproduced operation refusal can increase `decode_refused` without invalidating replay.
Inspect both counts rather than treating any nonzero counter as the same failure.

For file-only recovery, use the separate [complete runtime procedure](28-runtime-files.md).
A memory checkpoint alone does not carry every asset or control state of a complete runtime.

## Load runs and the derive list

`--workload` generates synthetic records for transport checks.
It is not a conversation-load setting. Use an explicit tick or record bound for these runs.

```sh
build/aotx_boot --journal build/load-check --ticks 100 --workload 12000 --derive console,bus
```

The derive list controls text outputs, not authoritative journal retention.
Its names include `console`, `note`, `bus`, `bulk`, `sequence`, `requests`, `transcript`, `tokens`, `pages` and `none`.
An affect build also accepts `affect` and `quality`.
The turn-manifest chain remains independent of this selection.

## How a run stops

`quit`, SIGINT, SIGTERM and the runtime window's close request initiate normal shutdown.
Boot performs its final flush, closes transport and waits for essential child completion.
A child failure during shutdown changes the runtime exit status to failure.
A stopped optional terminal is not an essential-child failure.

When an essential process fails during execution, boot reports it and skips the normal final GPU flush.
It requests termination of owned child processes, then kills and reaps those that remain after the bounded cleanup interval.
This prevents a failed run from being reported as a successful save.

## Failures and recovery

Preserve the journal and startup output after a failure.
Recover the last complete durable tick; do not infer durability from a displayed reply or HTTP admission alone.
The current GPU state can be newer than the saved journal.

| Symptom | Check |
| --- | --- |
| Missing settings file | Correct the explicit path or deliberately use defaults. |
| Profile or allocation refusal | Check free VRAM, loaded roles, media workspaces and cache capacities. |
| Missing model or changed digest | Check the exact manifest and model file. |
| Disk writer, feeder or service exit | Read the named child error and recover the durable journal. |
| Unavailable automatic memory or control | Check exact model, wrapper and qualification identity. |
| Prompt-capacity refusal | Check complete prompt bytes, sequence tokens and available pages. |
| Rejected restore input | Keep the original file and inspect the recorded schema and dependency failure. |

A driver or CUDA runtime failure can prevent a clean flush.
Forced termination can also leave unsaved work. Both cases require recovery from durable state.

## The watchdog

A display GPU can have a driver launch timeout.
A kernel that exceeds it can terminate the CUDA context.
Bounded token batches and separate raster scheduling reduce long unbroken work, but do not remove the driver's limit.

The pump launches one tick graph at a time and services page requests between ticks.
The raster graph uses a high-priority stream.
An overrun resets the pacing schedule instead of adding delay to every following tick.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
