# Changelog

Each entry names the commits that supply its change. The entries come from the commit record
after the preceding version tag.

## Unreleased

- Complete failed prompt admission without discarding prior conversation state (`e9cdc47`).
- Add instance tool defaults and per-conversation tool choices (`e9cdc47`).
- Add typed GPU state and bounded CCIR file transactions (`130dc50`).
- Select prepared memory on the GPU and record exact choices for replay (`c9415ef`).
- Bind fresh conversations to scoped memory and restore their recorded choices (`ba22afe`).
- Prepare text query vectors with the loaded GPU embedding model (`b9d7cd4`).

The typed store holds at most 256 object versions and 1 MiB of payload.
Text query preparation accepts up to 192 UTF-8 bytes per input.
Prepared queries keep their 2,048-byte input limit.
The current CCIR data-state profile is not a complete runtime package.
See [live memory](docs/20-live-memory.md) and [text requests](docs/21-text-memory.md) for operation and limits.

## 0.3.0

### Release

- Set version 0.3.0.
- Keep the two reranker class logits in registers without changing the score calculation.
- Report process-check failures even when an exception has no message.
- Require the expected illegal-address error in memory fault checks.
- Wait for the current boot's durable restore metadata before checking its journal.
- Keep ASCII window text ahead of Enter when an X11 input method is active.
- Check all local model files and permit each family's pre-tokenizer metadata.
- Correct model download commands and record verified file sources.
- State journal timing and quantized decay limits. Identify earlier benchmark versions.
- Correct the readout normalization formula and timer limits.
- Keep the saved instance name and device at startup.
- Preserve child startup output in the journal boot log and report its path on failure.
- Keep the New conversation control balanced when a persona import starts.
- Bound directory watches in the restore check without reducing its input count.
- Add fixed-input accuracy capture, reference validation, and complete comparison counts.
- State reranker vocabulary requirements and measured limits in model tool use.

### Known limits

- Some checked models make unrelated tool calls or omit the requested call.
- Per-instance and per-conversation tool selection controls are not included.
- A long persona conversation can repeat a prompt-capacity error without completing the input.
- The empirical numerical comparison remains incomplete as an accuracy qualification; see `docs/17-accuracy.md`.

### Layer and state tables

- Bind, select, and capture layers through type rows (`d4746e4`, `08e3df9`, `499d7e6`).
- Derive state allocation and cache pages from layer types (`3e0bd21`, `1cacc21`, `3275948`).
- Check mixed layer selection, compact cache addresses, and fresh descriptor placement (`6db6f20`, `bb156da`, `74c4876`).
- Read tokenizer families from model metadata (`03e742c`).
- Capture attention without query and key head normalization (`4676769`).
- Read bounded model turn wraps from the store (`a577091`).

### Control and operation

- Show custom local model files and keep file presence separate from active roles (`193ad6e`).
- Inspect model headers asynchronously with bounded time and output. Clear results when the model file or executable changes (`193ad6e`).
- Keep the exact selected language model and available companion roles through the wizard (`193ad6e`).
- Add Restore controls and a larger default conversation window (`193ad6e`).
- Add a model file guide with measured hybrid, expert, and memory examples (`193ad6e`).
- Correct replay timeout and failure counts, and add host controls (`193ad6e`).

### Tool history and restore

- Keep native call arguments and render stored tool results as separate turns (`01b8b2d`).
- Use the selected vocabulary for memory embeddings and refuse invalid note provenance (`01b8b2d`).
- Refuse rejected token replay while permitting a reproduced sequence-open refusal (`01b8b2d`).

### Tool call forms

- Select bounded tool call forms from complete model template identities (`38e0515`).
- Use one selected row for parsing, tool instructions, call history, and result turns.
- Support tagged JSON, native bare JSON, and function-parameter forms without template execution.
- Refuse unsupported forms without disabling text conversation.
- Report invalid memory provenance to the model without saving a note.
- Retain every call argument and render stored results as separate tool turns.
- Check native forms, metadata lifetime, and prompt bounds at one slot and all slots.

### Tool completion

- Require a ready embedding pass before memory tools appear in a prompt (`ba46a18`).
- Refuse unavailable memory calls at once and state the required model role.
- Use a separate embedding vocabulary when it differs from the language vocabulary.
- Record each device tool result, including empty results and errors, without duplicate host replies.
- Bound automatic continuation and tool turns by one input budget and report exhaustion on the console.
- Check tool completion, vocabulary selection and turn bounds at one slot and all slots.
- Add a real attach-socket conversation driver that retains replies, tool results and elapsed times.

### Model file checks

- Read Q4_1, Q5_0, Q5_1, Q2_K and Q3_K weights in each matrix and flat reader (`ab3ff3f`).
- Read Q4_K, Q5_K and Q6_K super blocks (`0c5bcbe`).
- Check each packed reader against a double reference at batches of one and 64.
- Add required-file checks that compare each tensor element and refuse missing input.
- Select rotary pairs from a shared family table and refuse unknown families.

- Add `aotx_models inspect` for local files and bounded HTTP header reads.
- Share compiled tensor and tokenizer tables with the header report.
- Bound header allocations and refuse duplicate metadata keys and tensor names.
- Run the real wrap check in the architecture test.
- Name the cache rebuild check correctly and exclude untested results from its tally.
- Add batched expert selection, selected matrix products, and weighted output sums.
- Add full-width query and key norms for the expert layer.
- Supply a file-specific expert model manifest with explicit turn wraps.
- Compare exact tensor names when layer types share descriptor slots.
- Check distinct batches through the decode graph and accept an explicit process-check driver.
- Refuse incompatible expert routing rules, tensor dimensions and tensor types before binding.
- Keep all-slot prefill checks within the token capacity on each profile.
- Add a separate attention bias layer capture and bias-before-rotation kernel.
- Bind complete tensor suffixes and require correctly shaped F32 query, key and value biases.
- Preserve explicit head widths and derive an absent width for the bias layer.
- Supply a pinned Q8_0 model entry with explicit conversation spans.
- Check bias placement, grouped heads, file refusals and token bounds across sequence slots.
- Give each layer row its tensor slot meanings and derive descriptor capacity from row spans.
- Bind layer offsets without pointer arithmetic across descriptor members.
- Check mixed tensor types, model isolation, absent slots and descriptor bounds across layer batches.
- Refuse tensor rows that exceed the inspector name-mask capacity.
- Add fixed F32 recurrent matrices and convolution history outside the page pool.
- Add batched delta layers and gated paged attention with partial rotary turns.
- Rebuild fixed state through prompt replay and check every state byte across chunk partitions.
- Add a Unicode-mark tokenizer pattern and a verified split-pair family row.
- Check mixed layer selection, fixed allocation sizes, gates, carried state and slot isolation.
- Refuse recurrent sequences with slots outside the table and check the fault counts.

## 0.2.8

### Release

- Set version 0.2.8 (the release commit).

### The pair mode and the fixture tool

- Add the pair mode of the quality score tool (`90c62df`).
- Add the fixture tool and the pair script (`51fae80`). The console line `outcome` and the fixture conversations.
- Extend the checks of the score and steer tools (`b3743be`).
- Make the armed result stand in for the call of a turn (`2f054ad`).
- Match the device template and tighten the checks (`1ed271a`).

### The completed request, the cosine readout and the log-odds score

- Make an armed tool result a completed request (`78bf1b3`).
- Read the probes as the cosine of the row (`ed4e381`).
- Score the pair judge by the log-odds in nats (`ce1a1bb`).
- Wait a fixed count of ticks for the quality rows (`8037e69`).
- Arm the ok result on a line with no tool field (`d8a3b2c`) and add the call arm of the outcome line (`b5215f6`).

### The dials and the spoken voice coupling

- Add affect dials and figures (`bb0e2d2`).
- State unavailable dial figures (`a07bf76`).
- Couple agent state to spoken voice (`600655a`).

### The actuator figures

- Carry three actuator figures in the affect trace (`a18c60e`). The budget spent, the entropy
  shift and the class frequency shift of a turn. The record grows to 92 bytes in a body of 192
  and stays derived, so a build without the option is unchanged.
- Show the actuator figures beside their dials (`cec8ba2`).

### The argument line and the profile checks

- Complete a call over the room of the argument line (`e731f73`). A call the parser refuses on
  its bound ends with a tool error result, and the turn after it writes the reply.
- Make the profile checks read the profile (`da10534`). The agent table checks and the pass count
  of a text set read the slots of the build. The probe checks read its language role.

### Documentation

- Document the state, the actuators and the score tool (`b670fbd`). The text of version 0.2.7, which its tag does not hold.
- Document the quality instrument and the dials (`6908222`). The quality page, the affect page, the control page and the measured figures.
- Add the changelog section of version 0.2.8 (`428a809`).
- State what the affect option leaves out (`94e385b`).
- Set the zero point of the readout on plain replies (`339353d`). The measured axes and the findings of the measurement.
- Replace the conduct samples with current output (`0b99d48`).
- Requote the stream samples from one measured turn (`f449d3f`).

## 0.2.7

### Release

- Set version 0.2.7 (the release commit).

### The affect state and its restore

- Add the affect state, its update law and its record (`262ad1b`).
- Derive the affect state line and check the restore (`50e6d62`).
- Keep the affect law with its open sequence (`4e35d20`).

### The calibration

- Add the dialogue neutral set and three conversations (`2c5b704`).
- Add the probe layer and the standardization set (`eef4c3e`).
- Measure the composite vectors in the calibrate mode (`c903de0`).
- Give the small talk conversation no tool outcome (`2eb6ff5`).
- Add the task set and the neutral reply set (`e684e77`).

### The actuators

- Couple affect to sampling and voice bias (`53881a3`).
- Build the bounded affect steer row (`b28feda`).

### The capability instrument

- Add the capability score tool and its check (`f0efd16`).
- Extend the axis check to the composite and the readouts (`f39260d`).

### Documentation

- Document the affect controller, the calibration and the score tool (`517be8d`).

## 0.2.6

### Release

- Set version 0.2.6 (the release commit).

### The affect substrate

- Add the optional affect data contract (`b712dc4`). The build option, the thirteen settings and the three record layouts.
- Add affect and quality drain streams (`752540f`). The two streams, the derive names, the whole-journal output and the identity check.
- Cut the agents table in the allowance test and keep failed identity runs (`738e718`).
- Add the affect sums, the probe loader and the read branch (`afedda6`).
- Add the affect turn node and its check (`b282ab2`).
- Mark verified tasks, refuse bad probe rows and mask the clock record (`12ed10c`).

### The derivation tool and the calibration

- Add the affect fixtures (`cce6375`).
- Add the axis mode of the steer tool (`2d1cfe3`).
- Add the calibrate mode and its check (`e8ff2da`).
- Refuse a flat probe scale, take the longer row and rewrite ten pairs (`094c49f`).

### The quality stream

- Measure quality in the embedding batch (`c6633cc`).
- Connect quality records to completed turns (`ae1a769`).
- Load the refusal phrases from the store and gate the page release (`fbd5f8a`).

### The control client

- Read affect and quality replica lines (`a0d287c`).
- Show affect and quality trace figures (`3213880`).
- Keep two instances apart in the control program (`e7f6e7d`) and build the instance name case without the option (`b3dea6d`).

### Documentation

- Document the affect substrate and the quality stream (`a725a09`). The affect page, the conduct page and the control page.
- Document the affect settings and the affect option (`68ed95d`).

## 0.2.5

### Release

- Set version 0.2.5 (the release commit).

### Decode controls and telemetry

- Add agent decode controls (`f8f32ef`). Class A records set the sampler row of each agent. The token statistics stream, the think budget and the stop command for one agent come with them.
- Add conduct controls and measurement tools (`b4fd0c4`). The steer vector derivation gives a potency figure. The steer add follows each layer residual. The bias profiles and the page map stream come with them.
- Publish the device ring use (`b0fc9f8`) and show the ring occupancy and record the rates (`a59f934`).
- Split the matrix rate fixture (`b570fc7`).

### Control client

- Make first-run startup and voice reliable (`a6fa503`).
- Improve conversation and sync controls (`1feb216`).
- Add model and persona controls (`28e4737`).
- Repair the control program and document it (`8904192`).
- Wrap the reply text at the window edge (`9b8db35`).

### Checks and repairs

- Fix the findings of the review of the quality pass (`5fa9c7c`).
- Harden the steer tools, the conduct loader and the checks (`7097b56`).
- Keep one manifest line for each role (`685c7bb`).
- Name an empty settings file in the replay gate (`96ca1dc`).

### Repository page

- Add the divider graphic to the README (`673d4e9`) and update it (`056546a`).

## 0.2.0

### Release

- Set version 0.2.0 and add the release documents and measured figures (`6086667`).
- Add the repository page (`8306cd8`).
- Add the terms table to the repository page (`608de31`).
- Replace the mark on the repository page (`30b3bb4`).
- Show the banner at full width on the repository page (`5495311`).
- Fix the release findings and update the verified figures (``4b56f51``).

### Profiles and settings

- Add the settings records, the card record and the common key list (`1c97e00`).
- Add the four build profiles and move device table limits into them (`e382048`).
- Add settings files and their disk-side readers (`82b9117`).
- Write a changed device setting at the tick commit (`15515be`).

### Modules and tools

- Add the import and remove record layouts (`d63e39b`).
- Add the device catalog and the import of modules (`9c3a356`, `47fbecb`, `4c7bc3e`).
- Add device and host tool modules, checks, examples and the tool SDK (`5afe360`, `abd1722`, `fa07d81`).
- Join tool nodes to the tick graph and make the tool contracts exact (`c87564f`, `7ec445a`).

### Terminal and conversation

- Add the display mirror and the terminal program (`bf61047`, `ddd5ee7`, `f2016e9`).
- Add long input lines, transcripts and conversation memory (`cd47376`, `4e4b645`, `316a637`).

### Model store

- Add the model record, the store program, run-time model load and store checks (`30ec4ca`, `c905dbd`).
- Make replacement, fetch and catalog rules exact (`d0a5b88`).

## 0.1.0

- 088c239 Add the repository layout, the gates and the clock module
- 4bba818 Add the seam
- 973c096 Add the surface
- 80a57ba Add the model file and the tokenizer
- 582948d Add the model kernels
- 5d730d1 Add decode through the tick
- 2a26563 Add agents and tools
- 7e70542 Add the documentation set
- b7d1257 Fold the block scale once in the PTX product
- a52bf6f Record the late verdict and pace the replay
- 9ecab9a State the late verdict and the pace of a replay
- 6297929 Hold the replay clock until a later tick is seen
- 50df09b Refuse a second granted line after a restore
- ff6ced1 Add the measured figures of one machine
