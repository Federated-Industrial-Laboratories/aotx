# Model files

This guide describes model file checks, store preparation, and language model use.
The control program and boot program use the same model store.

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

A file whose architecture, layer types, tensor types, and tokenizer are accepted needs no source change.
The runtime uses its own CUDA backend. The model architecture name does not select another backend.
Keep the model files and the local catalog outside the repository.

`share/models/olmoe/manifest.jsonl` contains an entry for one expert model file.
Copy the entry and its named model file into an external store. The entry includes its
turn wrap. Do not use that wrap for a different file. Header support does not prove greedy
token agreement with another backend.

The expert row accepts the `olmoe` routing rule only. Load checks its tensor dimensions and
types. Other routing rules need a separate supported row, even when their tensor names match.

`share/models/qwen2/manifest.jsonl` supplies a pinned Q8_0 entry with explicit role-tag spans.
Copy that entry and its named file into an external store. Its layers use `attention_bias`.
The Q4_K_M file from the same source contains unsupported block type 6; do not substitute it.

The Q8_0 file passes the six architecture checks. Two of four measured 24-token continuations
differ from the processor reference. Do not treat header support as exact token agreement.

The run-time table permits at most 63 bytes in each entry's name and file fields.
Use a short relative filename in the manifest, not a long absolute model path.

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

The default profile also requests embedding and reranker files unless `--roles language` selects language-only operation.
For memory tools, include an embedding model in the store and select `--roles language,embedding` or `--roles language-q4,embedding`.
The embedding vocabulary can differ from the language vocabulary.
A separate table serves the embedding batch when the token IDs or tokenizer family differ.
The shared-table path remains in use when the vocabularies agree.
After a vocabulary split, compatible language tables can grow to the largest vocabulary in either load order.

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
Its cache rebuild is not a stopped-process restore. Two optional arguments name a process-check
script and its output directory. That driver must compare an actual stopped and restored run
and observe the console layer line. Without both arguments, those checks remain untested.

```sh
build/aotx_arch_device_test "$STORE" /path/to/reference tests/arch_process.py /path/to/output
```

The test also compares distinct sequences through prefill and the actual decode child graph.
This batch check does not replace the process check.

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

Do not put the same thinking block in `generation_head` and the separate thinking spans.
For a Qwen3.5 non-thinking prefix, retain the line breaks in `"<think>\n\n"` and `"</think>\n\n"`.

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

## Use the graphical control program

Prepare and check the external store before opening the control program.
For a local file, the manifest entry makes its role active without a catalog action.
Create a settings file with `models.dir` set to the absolute store path.
Use a separate journal and settings file for each model.

```sh
mkdir -p "$HOME/aotx-journal"
build/aotx_ctrl --journal "$HOME/aotx-journal" --settings "$HOME/aotx.settings"
```

1. Open Models from the Windows menu.
2. Find the selected file under Active manifest roles or Model files.
3. Select Inspect to open Model details.
4. Check the file path and build path before reading the report.
5. Read the listed layer types and tensor formats.

The Model details window states whether the selected build accepts the header fields.
It also states that runtime use is not verified.
Inspector output contains the complete bounded report.
A failed or incomplete report displays a refusal. It does not retain an earlier successful result.
A changed file, model entry, or build requires another inspection.

The Models window distinguishes these states:

| displayed state | meaning |
| --- | --- |
| catalog entry | the built-in catalog supplies a fetch and activation source |
| file present | the path is a regular file with the declared size; the digest still needs a file check |
| active manifest role | the manifest assigns this file to the stated role |
| last reported language load | the journal contains the named load event; it does not prove current device residency |

Different files with the same name remain separate rows. Check the file and role before an action.
Fetch and Activate apply to catalog entries. A local-only entry requires an active manifest before startup.
Load sends the existing runtime command for an active file that is present.
Source shows declared source fields; it does not run a new digest check.

### Start through First run

1. Open First run from the Windows menu.
2. Wait for Detect to complete.
3. On Build, select the build directory and the zero-based Card number.
4. On Model, select the language file and check its role.
5. On Activate, check the active role and any companion-model notice.
6. Continue to Start and wait for the running phase and connected socket.
7. Continue to First say and wait for the completed reply.
8. Select Finish, then select the new instance in Instances.

The Start page starts the new instance when it opens.
The first-run journal and settings file are beside the initial journal directory.
Their names are `first-journal` and `first-instance.settings`.
For custom locations, create an instance through New in Instances before starting it.

The selected language role controls startup, including a selected `language-q4` role.
Present, active embedding and reranker roles are included when their assignments are unambiguous.
The wizard refuses an ambiguous role assignment or a selected file that disappears.
It reports unavailable memory tools when no active embedding file is present.
An embedding file alone does not establish correct tool selection by the language model.

### Restore a conversation

1. Send a short statement in the conversation window.
2. Wait for its completed reply.
3. Select Stop in Instances.
4. Wait for the closed phase.
5. Select Restore for that same instance.
6. Wait for replay to finish and the socket to connect.
7. Ask a new question about the earlier statement.
8. Check the saved transcript and the new reply.

Replay can take longer than a fresh start. The journal includes records from idle ticks.
Do not start another copy against the same journal while replay runs.
The CLI form adds `--restore` to the same boot command and journal path.

Use these disk commands to retain the recovery summary and token records:

```sh
build/aotx_restore --journal "$HOME/aotx-journal" --summary
build/aotx_journal tokens "$HOME/aotx-journal" --boot <boot-id>
```

After Restore, the new summary names the previous boot in `restore_of`.
Its `restore_hash` must equal the saved state hash.
Compare the historical manifest rows and replayed token tuples before sending the new question.
The tuple fields are slot, position, token, and flags.
A correct reply about earlier conversation text does not establish a memory-tool operation.

### Check memory through CTRL

Use the checked Qwen3 language and embedding files listed below.
Set `models.roles = language,embedding` in the instance settings before Start.
For the 8g profile, use a suitable checked `language-q4` file and `models.roles = language-q4,embedding`.
Set `sample.seed = 7`; keep temperature zero, affect off, and quality off.

1. Start the instance and wait for its connected socket.
2. Send the write input below in the conversation.
3. Wait for the tool result and the completed reply.
4. Check for a `memory_write` call and an `ok` result with the same request number.
5. Stop the instance and retain its recovery summary.
6. Restore that instance and compare its historical records before sending another input.
7. Send the recall input below.
8. Require a new `memory_recall` call, its successful result, and a final reply containing `4827`.

Write input:

```text
Please save this in memory: my garden gate code is 4827.
```

Recall input:

```text
What is my garden gate code? Please recall it from memory.
```

The transcript is `<journal>/<boot-id>/transcript/0.jsonl` for agent 0.
A tool result uses the request number to identify its call.
The restored write result must match the original result, including its request number and text.
The new recall result must contain the exact stored value; a reply alone does not pass this check.
Do not accept a longer number that only contains the same digits.

## Checked files and settings

The following files have complete digest and local header checks on the 12g build.
The hybrid and expert stores each assign one `language` role.
The memory store assigns `language` and `embedding` roles.

| file | bytes | source revision |
| --- | ---: | --- |
| `Qwen3.5-0.8B-Q8_0.gguf` | 811843840 | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` | 4213512672 | `02ab6ea6894a8418eb14f6d8ee1bfb08bd298080` |
| `Qwen3-4B-Q8_0.gguf` | 4280404704 | `bc640142c66e1fdd12af0bd68f40445458f3869b` |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639150592 | `370f27d7550e0def9b39c1f16d3fbaa13aa67728` |

The complete SHA-256 digests are:

```text
Qwen3.5-0.8B-Q8_0.gguf
0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c
olmoe-1b-7b-0924-instruct-q4_k_m.gguf
8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f
Qwen3-4B-Q8_0.gguf
8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300
Qwen3-Embedding-0.6B-Q8_0.gguf
06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439
```

The sources are `unsloth/Qwen3.5-0.8B-GGUF`, `allenai/OLMoE-1B-7B-0924-Instruct-GGUF`,
`Qwen/Qwen3-4B-GGUF`, and `Qwen/Qwen3-Embedding-0.6B-GGUF`, respectively.
Use the fixed revisions above with those sources.
The hybrid uses its recognized complete template.
The expert file uses the explicit wrap in `share/models/olmoe/manifest.jsonl`.

For the hybrid, put the checked file in a new store, then write its manifest:

```sh
STORE="$HOME/aotx-hybrid-models"
build/aotx_manifest write "$STORE" language "$STORE/Qwen3.5-0.8B-Q8_0.gguf" \
  unsloth/Qwen3.5-0.8B-GGUF 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 Apache-2.0
build/aotx_models --dir "$STORE" check
```

For the expert file, put the checked file in a separate new store:

```sh
STORE="$HOME/aotx-expert-models"
cp share/models/olmoe/manifest.jsonl "$STORE/manifest.jsonl"
build/aotx_models --dir "$STORE" check
```

These commands require the named model files to be present already.
Set `models.dir` to the chosen store before opening CTRL.
Both paths select the `language` role in First run.

The hybrid header lists 18 `linear_delta` layers and six `attention_gated` layers.
Its tensor table contains 133 F32 tensors and 187 Q8_0 tensors.
The expert header lists 16 `ffn_experts` layers.
Its tensor table contains 81 F32, 97 Q4_K, and 17 Q6_K tensors.

The graphical first-run checks use the generated settings and the following effective defaults:

| setting | value |
| --- | ---: |
| `sample.temperature` | 0 |
| `sample.seed` | 0 |
| `decode.reply_limit` | 256 |
| `decode.think_limit` | -1 |
| `affect.on` | 0 |
| `quality.on` | 0 |

The memory check sets seed 7 explicitly; the other values stay as listed.
Each check uses card 1, a separate journal, and a separate settings file.
The card number is machine-specific. Use an available card on the target system.

The hybrid restores its historical turns and token records, then returns the earlier marker `cobalt-731` in a new reply.
Its earlier acknowledgement request receives an unhelpful refusal despite repeating the marker.

The expert file completes its first greeting but adds unrelated text about a hammer.
It restores two historical turns and 251 token tuples without a difference.
Its next reply names the wrong earlier marker. The same three inputs give identical replies without a restart.
These are measured reply-quality limits, not evidence of a failed file check.

The Qwen3 memory check completes a write with testimony provenance and a successful tool result.
Restore preserves the write result, two historical turns, and 975 token tuples.
A new recall returns the saved code in its tool result and final reply.
This result applies to the stated files, inputs, and settings; it does not establish general task reliability.

## Read the memory figures

Attention layers use key-value pages whose storage grows with context.
The hybrid file also uses fixed recurrent state and convolution history for each sequence.
The page count alone does not describe that fixed allocation.
Restore rebuilds both state types from the saved token sequence.

An expert layer routes each token to selected experts.
Its expert tensors still need weight storage, and its attention uses key-value pages.
An active-parameter count does not state the complete file size or device memory requirement.
Use the boot memory check and the reported layer counts for the selected file and profile.

## Header checks and task quality

Keep these results separate:

| check | what it establishes |
| --- | --- |
| Header inspection | listed header fields and tensor sets match the selected build |
| File check | the active file matches its declared byte count and complete digest |
| Runtime check | the selected load, prompt, reply, and restore path completes |
| Task check | the returned answer or tool result meets the requested task |

A successful header check does not establish the other three results.
A completed reply does not prove that a requested tool ran.
Read the transcript call and result records before accepting a memory operation.
The recalled result must contain the stored value, including after restore.

The measured small Qwen files can write and recall a value but assign the wrong provenance to an operator statement.
The measured Qwen2.5-0.5B Q8_0 file also fails a factual question after a refused file read.
The measured Llama3.2-1B Q5_K_M and Q8_0 files write a value but omit the final recall call.
The OLMoE file has no native tool-call protocol. Prose about remembering does not store a finding.
These limits remain model-specific even when file and restore checks pass.
