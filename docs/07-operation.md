# Operation

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| ring | a single-producer, single-consumer ring buffer in pinned host memory |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| drain | the disk-side process that writes the outbound ring to the journal (a log writer) |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| catalog | the GPU-resident registry of imported modules: skills, roles and tools |
| profile | a build-time table-size configuration for one class of card |
| bus | an append-only message log between agents (a message bus) |
| arena | a contiguous memory region for offset-addressed allocations |
| pump | the host glue that launches the device scheduling graph once per tick |

This document states how a run starts and what the window shows. It then states what each
console command does. It ends with what a run leaves on the disk and how a run stops.

## Start a run

`aotx_boot` starts the system. It accepts these options:

| option | what it does |
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

A run needs a journal directory. A run with `--solo` needs none, because it starts no drain. The
default record count is 1,000,000. The pump makes at most 100 ticks in one second.

The start reads the clock module first and prints its sample. It then prints the run identity
and the sizes of the ring, the scratch arena and the host ring. With a model directory it prints
the digest line and the model line. The end of a run prints its counters and the state hash. It
also prints the model megabytes placed after the start.

```
aotx_boot --journal build/run --models models --roles language --window
```

Before the first placement the start reads the card and compares its free memory with the
need of the build profile (`docs/06-build.md`). A card that cannot support the profile stops
the start with the figures and names the profile that fits. An example of the line is
`profile 24g needs 19062 MB; 11335 MB free`. The start writes one CARD record with the card and the build
after the BOOT record.

## The settings file

A settings file contains one `key = value` a line. A `#` at the start of a line starts a
comment. A number is a whole number or a number with at most four decimals. Every key has a
default, a least value and a most value (`cuda/settings/keys.h`). The start reads the file
that `--settings` names, or `aotx.settings` beside the journal directory; a file that is not
there gives every default. A line the reader refuses is printed with its reason, and the
run starts with the rest.

The start prints one `settings:` line that names the file it read, or the path where it found
no file. A run log thus shows where the options came from.

| key | default and range | what it governs | takes effect |
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
| `agent.budget` | 8; 1 to 64 | turns of a task | the next task |
| `agent.pages` | 0; 0 to 4,096 | default hot page limit; zero takes the profile maximum | the next task |
| `agent.recall_k` | 4; 0 to 16 | warm turns recalled into a prompt | the next task |
| `agent.compact_at` | 128; 8 to 1,024 | warm turns that start compaction | the next task |
| `tool.deadline_ticks` | 500; 1 to 1,000,000 | ticks allowed after a request or grant | the next request |
| `mirror.hz` | 30; 1 to 120 | mirror snapshots in one second | the next frame |
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

The last thirteen rows exist only in a build with the `AOTX_AFFECT` option. The two run
settings `affect.on` and `quality.on` are 0 by default. The open of a sequence copies the
eleven other rows and `affect.on` into a law of that sequence. A change therefore never
reaches the reply in hand. `docs/14-affect.md` states what each row governs.

A key that the start reads (the first two rows) makes no record. Every other key goes into
the journal as one SETTING record, a class A record. The record is written when the file
names the key and when a `set` line changes it. A restore replays those records, so a restored system uses the settings
of the run it restores and reads no file.

```
# aotx.settings
tick.period_ms = 20
sample.temperature = 0.6
```

See `docs/11-terminal.md` for the terminal program, its screens and its keys.

## The disk-side programs

The boot program starts `aotx_drain` and `aotx_feed`. Each one sits beside the boot program in
the same directory, and each one receives only the descriptors it must map. A run with `--solo`
starts neither.

```
aotx_drain --ring-fd <fd> --journal <dir> [--bulk-fd <fd>] [--derive <list>]
aotx_feed --inbound-fd <fd> [--keys-fd <fd>] [--root <dir> --requests <file>]
aotx_restore --journal <dir> [--inbound-fd <fd>] [--summary]
```

The drain reads blocks from the host ring and writes journal segments. The feeder writes the
inbound ring. A run with `--window` gives the feeder the read end of the key pipe. A run with
`--root` gives the feeder that root and the requests file of its journal.

## The window

`--window` opens the window and draws the grid of 160 columns by 50 rows. The grid comprises six
panels, and every cell belongs to one of them.

- console: the last lines of the console buffer, with the command line on the last row. The
  buffer contains 256 lines of 160 bytes.
- agents: one row for each agent that is not free, and then the pending requests. The
  panel gives 10 rows to the agents and 3 rows to the requests.
- bus: the last messages of every kind, the newest first, up to 32 of them.
- arena: the region table and the memory budget. The budget reports the mapped bytes, the
  reserved bytes, the free bytes, the total bytes, the ring use and the mapped pages.
- tick: the tick, the records, the blocks, the blocked ticks, the records applied and the
  start time. It ends with the figures of the last statistics record. The last row contains the
  counts of the decode and of the agents.
- seam: the head bytes, the drain bytes, the lag in bytes and in ticks, the block sequence
  and the free bytes. It ends with the figures of the last stall record and the dropped runs.

The editor accepts the code points 32 to 126. It accepts Backspace, Delete, Left, Right, Home and
End. The Up key and the Down key walk the history, which contains 32 lines.

The editor grows to four rows and then follows the cursor. It shows the byte count from its
second row. Alt-Enter adds a line break. Enter gives the text to the parser.

The `Tab` key moves the focus between the console and the agents panel. The panel with the focus
shows a bright title. The editor accepts no key while the focus is on the agents panel. The key
`y` grants the first pending request, and the key `n` refuses it. Each answer writes a
console line that names the request and the answer.

The close request of the window manager ends the run. The close prints the frames drawn, and the
mean and the worst interval between two frames. It then prints the intervals over 20 ms and the
key events the pipe could not accept. No check and no tool of this repository destroys or kills
the window of another program.

## The command line

| command | what it does |
| --- | --- |
| `help` | show the command lines |
| `bus [kind]` | show the last bus messages of a kind |
| `note <text>` | put a note on the bus |
| `finding <source> <text>` | put a finding on the bus |
| `say <text>` | send a message to the conductor agent |
| `stop` | end the reply that runs |
| `continue` | resume a reply that ended at its reply limit |
| `outcome <ok\|error\|refused\|none\|call>` | arm one tool result for the next turn of the conductor agent; that turn runs no tool |
| `spawn <role> [n]` | make n agents of a role; n is 1 to 8 |
| `task <agent\|role> <text> [verify]` | open a task for an agent or for a role |
| `authorize <id>` | let a tool request of that number run |
| `refuse <id>` | stop a tool request of that number |
| `mem` | show the memory regions and the budget |
| `memory` | show the page pool and the limit of each live agent |
| `agents` | show the agents |
| `agent <id>` | show the transcript counts and the summary sequence of one agent |
| `agent <id> pages <n\|auto>` | change the hot memory bound at the next turn |
| `agent <id> compact` | start a compaction turn when the agent is idle |
| `agent <id> stop` | stop the reply of one agent at its next token |
| `agent <id> continue` | resume the reply of one agent that ended at its reply limit |
| `agent <id> decode.<key> <value>` | change one sampling value at the next turn |
| `stats` | show the counts of the last tick |
| `settings` | show the settings and when each takes effect |
| `set <key> <value>` | change a setting; the change is a class A record |
| `model load <role> <name>` | place a model file between two ticks |
| `model fetch <name>` | ask the feeder to fetch one model into the store |
| `models` | show each resident model, its file, digest and placement tick |
| `modules [kind]` | show all catalog modules, or only skill, role or tool modules |
| `module <name>` | show one module in full |
| `skills` | show skill modules |
| `roles` | show role modules |
| `tools` | show tool modules |
| `import <path>` | import a module directory through the feeder |
| `remove <name>` | remove an imported module |
| `quit` | stop the run |

A kind is `finding`, `rank`, `question`, `answer`, `handoff`, `cost` or `note`. A source is
`computed`, `fetched`, `recalled` or `testimony`. A role is `conductor`, `worker` or `verifier`.
An agent is a slot from 0 to one less than the slots of the profile (63 on the reference).
`docs/14-affect.md` states the `outcome` line, which is for scripted runs.

The decode keys are `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_window`,
`presence_penalty`, `frequency_penalty`, `seed` and `think_limit`. The value `absent` removes
the thinking limit. Each accepted line is a class A input line, so restore applies it again.

The list of the `bus` command shows 32 messages, which fills the console once.

One input line can use 32 record parts. The first part is an input record, and each next part
has the fragment mark. The apply joins the parts before it reads the command. A line that
crosses an apply batch remains pending until the next tick. The journal keeps each part in its input order.

One command writes at most 32 output records. A command that reaches the allowance ends with a
line that states the cut. Replay blocks the `quit` command.
A `quit` that a past run typed therefore does not close the run that replays it.

The model file list gives the names and roles accepted by `model load`. These are separate
fields. The only model roles are `language`, `language-q4`, `embedding` and `reranker`.
The command requires a manifest line with the given name under the given role. It refuses a
role with a live sequence and instructs the operator to enter `stop` first. It also refuses a bad digest or a file
that does not fit the weights region.

A profile that keeps one language model releases the old allocation. The region check
includes that room and leaves only the new language model resident.

The 24g and 48g profiles may keep both language descriptors. Placement writes a stall line
before the copy and another after it. The next complete tick records the model and its digest. A
restore checks that digest and places the same file before it continues.

`model fetch` changes the store only. It does not change the resident model. The feeder runs one
fetch at a time and reports its progress and final state on the console. The Models screen
offers fetch for a catalog file that is not on disk. For a file on disk but not in the manifest,
it runs `aotx_models activate <role> <name>`. It sends `model load <role> <name>` only when the
file is on disk and that exact name and role are in the manifest.

## The model store

The model store is the directory that `--models` or `models.dir` names. The repository catalog
is `share/models/catalog.jsonl`. Each entry names the source, revision, license, byte count and
SHA-256 digest. The local `store.jsonl` records verified files, and `manifest.jsonl` records
the active name and role.

Use the disk-side store program before the first start:

```
build/aotx_models --dir models list
build/aotx_models --dir models fetch language
build/aotx_models --dir models activate language language
build/aotx_models --dir models check
```

The 8g profile uses `language-q4` as its default language role. Replace both `language` words
in the fetch and activation commands for that profile. A later file can use another catalog
name with the same role.

The complete store forms are:

```
aotx_models [--dir <dir>] [--catalog <file>] list
aotx_models [--dir <dir>] [--catalog <file>] fetch <name>
aotx_models [--dir <dir>] check
aotx_models [--dir <dir>] [--catalog <file>] activate <role> <name>
aotx_models [--dir <dir>] [--catalog <file>] remove <name>
```

A fetch can resume its part file. It checks the byte count and complete digest before rename.
The remove command removes the file and its local store row. It does not remove a resident
model from a system that runs.
### Use another model file

A model whose layer types, tensor types, and tokenizer are supported needs no source change.
The runtime uses its own CUDA backend. The model architecture name does not select another backend.
Keep the model files and the local catalog outside the repository.

1. Select a model file and a fixed source revision.
2. Run `build/aotx_models inspect <file-or-url>` to read its GGUF header before the full download.
3. Check `general.architecture`, `tokenizer.ggml.pre`, and every tensor type in the tensor table.
4. Record the source byte count and SHA-256 digest.
5. Read the model license.

The quantization name in a filename does not state every tensor type.
One file can contain several tensor types.
The inspector needs no catalog, manifest, or device.
Use a local path or an HTTP(S) URL with a fixed source revision:

```sh
build/aotx_models inspect /path/to/model.gguf
build/aotx_models inspect 'https://host/path/model.gguf'
```

The report gives architecture, pre-tokenizer, tensor count, and each block type count.
It also gives layers, hidden width, vocabulary size, template byte count, and template SHA-256.
`file_bytes` is the complete source size. `header_bytes` ends at the tensor table, before alignment padding.
The template digest is not the complete file digest.

`build_support=no` names unsupported or missing header fields and tensor sets.
`build_support=yes` applies only to the listed header fields and compiled tensor sets.
`run_verified=no` means that weight integrity, device memory, wrap, prefill, and restore still need checks.
An unknown tensor type remains visible by its numeric id and has `supported=no`.

Remote inspection requires `AOTX_FETCH=ON` and a server that supplies exact HTTP byte ranges.
The server must supply a stable strong ETag or Last-Modified value.
The first resolved URL is held for all later ranges; a later redirect is refused.
A changed file, ignored range, short response, or invalid range is refused.

The command reads bounded pieces and can read ahead past the tensor table; `received_bytes` states the total body bytes.
It does not save a model file. TLS checks remain on, and redirects are limited to ten.
Local inspection also works with `AOTX_FETCH=OFF`.

The header limit is 256 MiB. Parser allocations have a combined 512 MiB limit.
Every declared length is checked before a read or allocation.
Malformed or truncated headers print the source and problem, then return status 2.
Read and network errors return status 1. A complete report returns 0, even when `build_support=no`.

#### Fetch with a local catalog

Create a local catalog with one JSON object on each line.
Use `share/models/catalog.jsonl` as the field example, not as the local store.
Replace every bracketed value in this example before use:

```json
{"name":"local-model","role":"language","repository":"<repository>","file":"<file.gguf>","revision":"<fixed revision>","bytes":0,"sha256":"<64 hexadecimal digits>","license":"<license>","quant":"<quantization>","profiles":"12g","verified":false,"source":"<repository>","note":""}
```

Replace `bytes` with the complete file byte count, not the header byte count.
The catalog name is a local label. The role selects a runtime function.
For a language-only store, use `language` and select that role explicitly at boot.

Run these commands after the build in `docs/06-build.md`:

```sh
STORE="$HOME/aotx-models"
CATALOG="$HOME/aotx-catalog.jsonl"
mkdir -p "$STORE"
build/aotx_models --dir "$STORE" --catalog "$CATALOG" list
build/aotx_models --dir "$STORE" --catalog "$CATALOG" fetch local-model
build/aotx_models --dir "$STORE" --catalog "$CATALOG" activate language local-model
build/aotx_models --dir "$STORE" check
```

Fetch verifies the complete file and writes `store.jsonl`.
Activation writes `manifest.jsonl`. It does not load device memory.
Check reads the active manifest, not the catalog or the list of downloaded files.

On an empty store, activate before check.
If check reports `the models manifest does not read`, check that activation completed.
If fetch or activation reports `the catalog has no such model`, check the catalog path and local name.

#### Use a file already on disk

Put the file in the store before the manifest write.
The manifest stores its basename, not its original path.
For a language-only store, use this form:

```sh
build/aotx_manifest write "$STORE" language "$STORE/<file.gguf>" "<source>" "<revision>" "<license>"
build/aotx_models --dir "$STORE" check
```

The manifest writer uses the name as the role.
This form needs no catalog or separate activation command.
Use the catalog form when the local name must differ from the role.

#### Start and check a conversation

Create a settings file with these lines:

```ini
sample.temperature = 0
decode.reply_limit = 64
```

Start the actual boot program:

```sh
build/aotx_boot --models "$STORE" --roles language --modules modules/roles \
  --journal "$HOME/aotx-journal" --settings "$HOME/aotx.settings" \
  --derive transcript,tokens --ticks 0
```

Without `--roles language`, the default profile also requests embedding and reranker files.
A store with only a language file cannot supply those roles.
Check the tensor count, the unplaced count, and the `layers:` line at boot.
Require `spans=pass ends=pass prefill=pass` and `usable=yes` on the file's `wrap:` line.

An affect-enabled build can print `affect composite: the last calibration is not trusted` for a new store.
A store without calibration files has no optional composite steer.
This message does not refuse the language model or prevent ordinary replies with affect off.
If calibration files are installed, check them against `docs/14-affect.md`; the same message also reports invalid calibration data.

Enter `say` followed by the first question.
Wait for the completed reply before the next `say` line.
Make the second question depend on the first reply.
Check that neither reply contains model header or end-token text.
Enter `quit` to stop.
The transcript files under the journal directory retain the exact reply text.

Tool calls can start more turns before the agent completes a user request.
A raw transcript reply alone does not prove that a user request is complete.

A valid wrap does not guarantee that the model follows every instruction or uses tools correctly.
Literal tool-call text in a reply does not prove that a tool ran.
For a text-only check, request an answer without a tool call.

The disk-side store check verifies file identity and prints the wrap.
It does not check all device behavior.
`build/aotx_wrap_load_test "$STORE" language` checks the wrap on the device.
Its invalid-table cases must fail internally; the final test status must be zero.
To check journal restore, compare a third reply after `--restore` with an uninterrupted three-turn conversation.
Use identical inputs and temperature zero for both conversations.

The architecture test requires reference token lists.
It runs the actual device wrap check and counts its result.
Its cache rebuild is not a stopped-process restore; its load-line notice is not an output check.
The real boot comparison supplies the restore check. Check the printed layer sequence separately.

`tests/affect_identity.sh` requires a store with all default profile roles.
It has no role-list argument for a language-only store.


### Turn wraps

Each active manifest entry can hold a `wrap` object. This object takes precedence over
the model template. Without it, the reader accepts only a known complete template.
An unknown template requires a `wrap` block; the refusal names the model file.
The reader does not execute templates.

The object has nine string fields: `system_head`, `system_tail`, `user_head`, `user_tail`,
`assistant_head`, `assistant_tail`, `generation_head`, `think_open`, and `think_close`.
Each string holds at most 64 UTF-8 bytes, with a total of at most 432 bytes.
JSON escapes are accepted. NUL bytes, duplicate keys, and unknown wrap keys are refused.
`end_ids` is an array of one to eight distinct unsigned token ids.
`prefix_length` is optional and defaults to zero.

The prefix consists of the first `prefix_length` bytes of `system_head`.
A prompt with system text emits it as part of that header.
A prompt without system text emits it once before the first user header.
Stored turns do not repeat the prefix; a client supplies date preambles as system text.
Empty thinking spans add no bytes. A generation uses `generation_head`, then both thinking
spans; a stored reply uses `assistant_head` and `assistant_tail` instead.

To make an unknown wrap, read the file's complete `tokenizer.chat_template` and vocabulary metadata.
Identify the literal bytes before and after each text role and before generation.
Keep every required line break. Use the file's vocabulary ids for its end tokens.
Read `tokenizer.ggml.eos_token_id` and any turn-end metadata; do not copy ids from another model.
Check any start-of-text prefix separately, because it must occur only once.

Add all nine span fields and `end_ids` to the active manifest entry.
Use empty strings only for spans the template does not use.
The store program does not extract an unknown template or write this block for you.
Do not execute a template obtained from an untrusted source.

The reader matches known templates by their complete byte length and SHA-256 digest.
A similar architecture or template name is not a match.
If check prints the spans without a manifest override, the complete-template match succeeded.
If a template cannot fit the bounded spans, the current wrap format cannot represent it.


The disk-side `check` command prints every span with escapes and the end-token ids.
The device load checks token order, each end-token id, and a short prefill.
A failed check leaves the file loaded but disables its prompt paths and quality scoring.
Embedding and reranker calls remain available; those roles do not require a language head.

Optional entry keys `probe_numerator` and `probe_denominator` select the probe layer.
The layer is the layer count times this fraction, rounded down; the default fraction is 2/3.
The numerator must be less than the nonzero denominator. The load prints the result.
A fitted probe records this absolute layer and must match the current selection.
Each row in the shared probe catalog must match every loaded language model's width and selected layer.

## Replies

A run loads model files from the directory that `--models` names. With a language model
resident, the command `say <text>` sends the text to the conductor agent. The command wraps the
text with the checked wrap table of the loaded model. A new agent gets the sampling defaults
from the settings table. The neutral defaults select the largest logit, as the earlier greedy
path did, and the reply contains 256 tokens at most.

The console shows a line that starts with `conductor: `, and the reply grows that line as the
tokens come. A newline byte in the reply starts a new line. A control token carries no text of
the reply, so the console never shows its bytes. A reply therefore ends with its last text. At
the end one bus message states the token count and the ticks used by the reply.

One conductor reply runs at a time. A second `say` while the conductor is not idle is refused.
The command `stop` ends the conductor reply. The command `agent <id> stop` ends the reply of the
selected agent. The turn ends normally and its `done` transcript line has status `stopped`. A
`say` with no language model is refused, and a `say` with no conductor agent is refused with the
name of the `spawn` command.

The wrap check derives thinking-token ids from the model's thinking spans.
A thinking limit of zero masks the opening token at the start of a reply. A positive limit permits that many
tokens inside the span. At the limit, only the closing token is permitted. The value `absent`
leaves the span without a limit.

## Agents

An agent is a record, a sequence slot and a share of the arena. The profile sets the slot count.
The 12g profile provides 64, and the 8g profile provides 32. Agent 0 is the conductor. A task gives
an agent 8 turns by default.

The text of a `say` and the text of a `task` can use the prompt byte bound of the profile. The
8g and 12g profiles use 6,144 bytes. The 24g profile uses 12,288 bytes. The 48g profile uses
24,576 bytes. A longer text is refused, and the line names the bound. The word `verify` at the
end of a task activates result verification by a verifier agent.

The catalog starts with nine built-in tools. The device runs `memory_recall`, `memory_write`
and `skill_use`. The feeder runs `fs_read`, `fs_stat`, `fs_list`, `fs_write`, `fs_update` and
`run`. A role manifest selects its tools and the calls that need operator authorization.

The command `agents` and the agents panel show one row for each agent that is not free. A row
contains the identity, the role and the state. It then contains the active task, the tool of a
pending request and the number of that request. It ends with the completed turns, the reply
tokens and the reply tokens each second. A state is `free`, `idle`, `prompt`, `run`, `tool` or
`post`.

The agents panel lists each pending request with its number, its agent, its tool and the
first 40 bytes of its argument. The commands `authorize` and `refuse` answer any request by its
number.

## Conversation memory

Each agent has its own ordered transcript. A turn keeps the input line, the reply, the tool
call and its result, and an authorization answer when they exist. The prompt starts with the
role text. It then has the summary, recalled warm turns, hot turns and the new input text in
that order. Recalled turns show their turn numbers.

The newest turns are hot. Their prompt and key value data use the page limit of the agent. A
role can give `pages` and `pages_least` in its manifest. A role with no `pages` value uses the
`agent.pages` setting. The command `agent <id> pages <n>` changes one agent at its next turn.

The value `auto` uses the pages that the pool can provide when the turn opens. It does not go
below `pages_least`, which is 16 when the role gives no value. It does not go above the profile
maximum. The selection record of the turn states the limit used by the turn.

| profile | pages in the pool | tokens in the pool | most pages for one agent | most tokens in one sequence | transcript text for one agent | transcript text for all agents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8g | 512 | about 7,100 | 148 | 2,048 | 64 KB | 2 MB |
| 12g | 1,024 | about 14,000 | 160 | 2,048 | 256 KB | 16 MB |
| 24g | 4,096 | about 57,000 | 320 | 4,096 | 1 MB | 128 MB |
| 48g | 12,288 | about 172,000 | 640 | 8,192 | 4 MB | 1 GB |

The pool token figures use about 14 tokens for each 2 MB page. The exact page need comes from
the shape of the active model. A long transcript can therefore use much of one card.

A turn that leaves the hot bound becomes warm. The embedding batch makes its vector. Recall
compares the new text with the warm vectors and puts the nearest `agent.recall_k` turns in the
prompt, oldest first. The text stays on the device so the recalled turn is quoted and is not
rewritten.

When the warm count passes `agent.compact_at`, the agent summarizes the oldest half in a new
turn. It uses more than one turn when the text does not fit one prompt. The command
`agent <id> compact` starts the same action. The summary is a finding with computed
provenance.

It gives the first sequence, the last sequence and the count of the folded range.
A new summary corrects the one before it. When the text arena is full, the oldest folded text
leaves first. Its vector and the summary stay on the device.

A prompt that does not fit after the oldest hot turns leave is refused. The console states the
reason, and the refusal count increases. No accepted prompt is cut.

Each prompt writes a class A selection record. It gives the warm turn sequences, the summary
sequence and the page limit used by that prompt. A restore applies this record and does not run
the cosine search again. This keeps the prompt input hash equal to the earlier run.

## Tool requests and file reads

An agent that calls the tool `fs_read` writes a request record. The drain turns that record into
one line of `<journal>/requests.jsonl`:

```
{"request":1000,"agent":0,"turn":1,"tool":"fs_read","arg":"notes/one.txt","deadline":507,"auth":"none","tick":7}
```

The field `auth` is `none` for a tool that needs no authorization, and `granted` for a tool the
operator authorized. A request pending operator authorization makes its line when the record that
grants it comes. A request the operator refuses makes no line.

A request of a tool that needs no authorization carries a deadline of 500 ticks from the tick of
the request. A request pending operator authorization carries no deadline. Its deadline of 500
ticks starts at the tick the operator grants it, so the operator may answer at any time.

A request that gives no answer before its deadline receives a late verdict. The device writes that
verdict as a tool reply of the status `late`, and the bytes of the reply give the reason. The
verdict is a class A record, so a replay applies it at the same place in the order. The feeder
writes no `late` status, because the feeder has no tick.

The feeder reads that file and executes the requests:

```
aotx_feed --inbound-fd 3 --root /home/user/notes --requests build/run/requests.jsonl
```

`--root` names the one directory a file read may reach. It is the security boundary of the
system. Every component of a path is opened with `O_NOFOLLOW`, so a symbolic link at any depth
is refused. A component of two dots is refused. A path that starts at the root of the file
system is refused. A path of more than 64 components is refused.

A path that names anything other than a regular file is refused. A read returns 4,096 bytes at
most, which is the size of the result buffer of an agent. A larger result does not fit the
sequence of the turn beside the text of the role.

The answer is a reply record, or several. One reply is `parts` records with the same agent and
request, from part 0. A part with the status `ok` carries content, in order. A part with any
other status is the last part of the reply, and its bytes are the reason.

Every successful `fs_read` result includes a `sha256` line for the bytes it served. The parser
keeps that line in the tool result that the agent reads. The built-in tool `fs_stat` returns the
size, modification time and digest of a file without returning its contents.

A file that the cap cut gives the parts of its first 4,096 bytes and one more part that states
the cut. A path the root rule refuses gives one part with the status `refused`. A file that is
not there gives one part with the status `error`.

One request is executed one time. The feeder maintains the identities of the last 1,024 requests and
executes no identity twice. A requests file that is already there when the feeder starts is read
from its end. A feeder that starts after a restore therefore executes no request of the run
before it. The device applies the replies that the journal contains.

## The turns of a run

The drain writes one line for each completed turn to `<journal>/manifest/<boot id>.jsonl`:

```
{"agent":0,"turn":1,"input_hash":"1111000000000000","output_hash":"2222000000000000","tokens":7,"finish":"stop","tool":"fs_read","request":1000,"prev":"<64 hexadecimal characters>"}
```

The field `prev` is the SHA-256 digest of the bytes of the line before it, with the end byte of
that line in it. The first line of a file carries 64 zeros. A removed line, or a byte
that changes, therefore breaks every line after it.

## The journal a run leaves

A journal directory contains one directory for each boot, named with the identity of that boot.
That directory contains the segments, which are named `seg-000000.seg` and up, and the console log.
A segment contains 64 MB at most.

Beside the boot directories the journal contains three more entries. The directory `bus` contains one
message file for each day. The directory `bulk` contains the payloads and an index. The directory
`manifest` contains one chain file for each boot, and the file `requests.jsonl` contains the tool
requests of the journal.

When the derive list contains `tokens`, each boot directory also contains `tokens.jsonl`. One
line gives the tick, agent, turn, token index, token identity, selected log probability,
distribution entropy and thinking flag of one emitted token. These records are class B. Restore
does not apply them.

When the list contains `pages`, each boot directory also contains `pages.jsonl`. One line gives
the tick, agent, page, residency, and attention mass at the 64-tick measurement cadence. The
figures do not control eviction.

When the build includes the affect option, the list also takes `affect` and `quality`. These
names write `affect.jsonl` and `quality.jsonl` in each boot directory.

`aotx_journal` prints the records of a journal as text, one record for each line:

```
aotx_journal tokens build/run --boot 00000000cafe0001
aotx_journal manifest build/run
aotx_journal requests build/run
```

The command `tokens` prints the token records of a run. The directory is a boot directory when
it contains segments. If it does not, it is a journal directory: `--boot` names the boot in it, and
with no `--boot` the newest complete boot is read. The first four fields of a line are the token
itself, so a comparison of two runs cuts each line after them. The field `sampled` is one for a
token the model made, and the field `replayed` is one for a token a restore applied again.

The command `manifest` prints the turns of a run and verifies the chain. It recomputes the
digest of each line and compares it with the field that the line after it carries. The command
ends with status 0 when every chain is valid. It ends with status 1 at the first line that breaks a
chain, and the report names that line. With no `--boot` it reads every chain file of the
journal.

The command `requests` prints the tool requests of a journal, one for each line. A line contains
the identity, the agent, the tool, the state of the authorization, the deadline and the path.

## Restore

`--restore` replays the newest complete journal before the first input. The replay sends every
class A record of that journal. It leaves out the boot record and the tick commit record, which
the device makes again on its own. Each replayed record carries a flag that marks it as one the
system applied before.

The apply processes the records of one journal tick in one tick of the restored system. The restore
duration therefore equals the tick count of the system that wrote the journal. A journal of 10,000 ticks requires
10,000 ticks to replay. The pace gives an input of the operator the place in the flow of the
agents it had before. A journal tick with more records than one apply processes spills into the
ticks after it and never merges with the next one. The replay makes its ticks as fast as the
device runs them, and the tick period does not block them.

A restored run executes no tool request that the journal already answers. The reply of such a
request is a record of the journal, and the replay applies it. A pending request appears again
when the replay ends, and the operator answers it as before. A replay whose ring
makes no progress for a million turns of the replay loop ends the run with a line that names
it.

The operator sees one line at the end of the replay. It states the records applied, the state
hash the device computed, the records refused, the pages mapped and the paced ticks. A paced
tick is a replay tick that processed no journal record. A `quit` typed by the past system
does not close the run that replays it.

A restored reply continues its tokens and not its console line. The console line belongs to the
`say` command that opened the sequence. A sequence restored from the token
records has no such line, so the console does not show that reply again.

`aotx_restore --summary` reads a journal and prints its figures without a ring. The line reports
the boot identity, the last tick, the records replayed and the state hash.

## Load runs and the derive list

A run with a tick load writes many records for each tick. The drain makes a line of text for
every record of the types it derives. A load of thousands of records a tick therefore makes
thousands of lines a tick. Give `--derive` to name the types the drain makes lines from:

```
aotx_boot --journal build/run --workload 12000 --derive console,bus
```

The names are `console`, `note`, `bus`, `bulk`, `sequence`, `requests`, `transcript`, `tokens`,
`pages` and `none`, with commas between them. A run that gives no list leaves the drain with its
default, which is every type.
The journal keeps every record, whatever the list contains; the list changes the derived files
only. The name `sequence` makes one line at the end of a reply, with the slot, the token counts
and the ticks. A token record makes no line and stays in the journal segments. The name `bus`
covers the message records and the task and agent events, because all three make message lines.

The chain of turns is not in the list. A turn that makes no line makes a gap in the chain, and a
chain with a gap proves nothing.

A short list gives the drain less to write. Give `--records` or a smaller `--workload` to bound
the size of a journal. The option `--derive` bounds the derived lines only.

## How a run stops

A run stops at the `quit` command, at the close request of the window manager, and at the
signals SIGTERM and SIGINT. Each of them ends the run the same way: the last flush, the closed
rings, completion of the disk-side programs, and the reports. A second signal changes nothing,
because the run is already stopping.

A system with a drawing context must never end at the default action of a signal. The display
server keeps the window of a program that stops in the middle of a frame. An operator who must
end a run that stopped answering sends SIGKILL, and knows what that leaves behind. The last
block is not flushed and the rings stay as they are. The journal contains the system state up to its last
complete tick. A restore reads that journal and gives the state back.

## The watchdog

The display shares the GPU with the system, so the launch watchdog of the driver applies. A
kernel that runs longer than the timeout of the watchdog is killed, and a killed kernel kills
the context of the run.

The code answers that with short kernels and a bounded batch. The plan of a tick admits 512
prompt tokens over every slot. A prompt longer than that is cut into pieces. A piece that does
not fit the budget of the tick remains pending until the next tick. No kernel of the tick graph therefore
grows with the length of a prompt.

The tick graph runs on the pump stream. The raster graph runs on a stream of the highest
priority, so a tick does not block the display. The pump services the page requests of a tick
between two ticks, when no kernel of the tick graph runs. The pace maintains a schedule and not a
delay. A tick that runs long therefore gives the schedule a new start, and does not push the
ticks that follow it.
