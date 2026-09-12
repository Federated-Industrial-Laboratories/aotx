# Audio input

The optional audio component accepts speech and other sounds as native model input.
CUDA validates the source, decodes PCM, resamples, builds log-mel values and runs the trained audio encoder.
The language decoder receives the projected audio rows at each source reference.
No speech transcript or generated sound description replaces those rows.

The supported pair is Qwen2-Audio-7B-Instruct with its F16 audio encoder.
The language file uses Q4_K_M weights.
Base text conversations and the separate [image component](29-image-input.md) remain available.

## Prepare the model store

Add the audio language file to `manifest.jsonl` with role `language-audio`.
Create `audio.jsonl` with exactly two entries:

1. Copy that language entry from `manifest.jsonl`.
2. Add the matching encoder entry with role `audio`.

Each entry requires name, role, path, source, revision, license, SHA-256 and byte count.
The parent entry must match exactly. The two names, paths and digests must differ.
The component reader checks all 489 tensor shapes, types and extents before allocation.

| Field | Value |
| --- | --- |
| Source | `mradermacher/Qwen2-Audio-7B-Instruct-GGUF` |
| Revision | `e1e68850ba33e38eafbc3817919c318d9c7e757b` |
| Language file | `Qwen2-Audio-7B-Instruct.Q4_K_M.gguf` |
| Language bytes | 4790394016 |
| Language SHA-256 | `1ef119f39f39258c39e1c95c6e0144da2733f4862c673add9afcf3a9d42f48bc` |
| Encoder file | `Qwen2-Audio-7B-Instruct.mmproj-f16.gguf` |
| Encoder bytes | 1289301536 |
| Encoder SHA-256 | `b52435dead2956f1fc113818c3b5ceb42a940cb487e59163cb1ffc69cae69347` |
| License | Apache-2.0 |

Set the explicit chat wrap described in [model files](16-model-files.md).
Use the Qwen `im_start` and `im_end` message spans, without thinking spans.
The end IDs are 151645 and 151643. The generation head is `<|im_start|>assistant` followed by a newline.
Select the audio parent at startup:

```text
aotx_boot --models STORE --roles language-audio,embedding --journal JOURNAL
```

A combined store can select `language,language-audio,embedding` and include the image pair.
Each request keeps its model, vocabulary, chat wrap and native feature width.
An audio reference selects `language-audio`, including references in retained context.
A prompt that requires both supported media parents is refused.
Use separate conversation slots for requests that require different parents.
Raw trained audio markers are refused without resolved sources.

Store-wide voice, steer and probe assets belong to the default language model.
They do not apply to a separate specialist model through matching token IDs or vector widths.
The optional per-agent affect state and temperature controls remain independent of those assets.
When audio is the default language model, its compatible store assets can apply to it.

## Submit audio

Use the terminal or the local attach interface:

```text
audio load 0 private wav /absolute/path/sound.wav
audio load 0 private pcm16 48000 2 /absolute/path/samples.pcm
audio load 0 private f32 16000 1 /absolute/path/samples.f32
```

Raw samples use little-endian byte order. Channels are interleaved.
The remainder of the command is the pathname; spaces need no quotes.
The file reader rejects symlinks, nonregular files and files changed during transfer.
It reads bytes on the runtime host. It does not interpret samples.

The console gives a source reference and transfer ID.
A ready notice means that native rows are available:

```text
audio: ready [audio:SHA256] transfer TRANSFER_ID
say [audio:SHA256] Describe the sound briefly.
```

Replace `SHA256` with the displayed 64 lowercase hexadecimal characters.
Several audio references can appear in one prompt, in order.
The decoder receives the trained audio markers and rows; the source digest and pathname are omitted.
Missing, malformed, retired or inaccessible sources refuse the complete prompt.
A valid source that is still processing keeps the prompt waiting.

The `private`, `room` and `shared` scopes have the same rules as image sources.
Without a cognitive binding, `private` belongs to the addressed base conversation slot.
An active sequence keeps a feature lease until it ends.
Cancel an owned transfer or retire an unused source:

```text
audio cancel 0 TRANSFER_ID
```

Audio adds these hexadecimal refusal statuses to the common media statuses:

| Status | Meaning |
| --- | --- |
| `9` | The complete canonical signal is exactly zero |
| `a` | Invalid or unsupported audio representation |
| `b` | A sample or numerical result is not finite |

The zero-signal refusal is exact. It is not a speech detector or a quiet-audio accuracy guarantee.
Model replies can be incorrect, including sound labels and speech transcriptions.

## Supported source bytes

WAV input supports RIFF/WAVE PCM16 and IEEE F32, including validated extensible PCM/F32 headers.
Supported rates are 16000, 44100 and 48000 Hz, with one or two channels.
The parser checks the complete declared file extent, chunk padding, sample alignment and format fields.
It refuses duplicate data chunks, inconsistent lengths, unsupported encodings and nonfinite samples.

Each source can contain at most 30 seconds.
A source must produce at least one pooled audio row.
This requires more than 320 canonical samples at 16000 Hz.
Longer recordings can be submitted as separate sources.
Microphone drivers, compressed audio codecs, video decoding and network retrieval are outside this interface.
A producer can submit captured samples through the common media ring.

The raw transport source begins with a 32-byte header:

| Byte | Field |
| --- | --- |
| 0 | Eight bytes `AOTXPCM1` |
| 8 | Encoding: 1 for PCM16, 3 for F32; unsigned 32-bit |
| 12 | Sample rate, unsigned 32-bit |
| 16 | Channel count, unsigned 32-bit |
| 20 | Four zero bytes |
| 24 | Frame count, unsigned 64-bit |

All fields are little-endian. The exact sample bytes follow the header.
The digest covers the header and samples. The raw file commands supply this header automatically.
The common begin record uses format 3 for WAV or 4 for raw PCM; width and height are zero.

Stereo channels are averaged. Resampling operates over the complete source timeline.
Transport chunks do not restart the filter or spectral window.
The encoder preserves valid input lengths and masks padded attention keys.
Each source produces at most 750 rows, each containing 4096 F32 values.
These rows count toward the conversation token and KV limits.

The supplied conductor role has a 160-page quota; the worker role has 32 pages.
Audio prompts can need more pages than these quotas permit.
On the 12g profile, increase the quota for each unbound conversation that needs it:

```text
agent 0 pages 640
agent 1 pages 640
```

Bound conversations use the page quota in their cognitive binding.
Each request must fit both that quota and the shared physical pool.

Select each required language role at startup.
The `model load ROLE NAME` command can replace an active base language role.
It cannot add a language role whose vocabulary, working memory and decoder are absent.

## Device capacity and recovery

The common media profile bounds source descriptors and encoded bytes for both media types.
Audio features and worker storage have separate capacities:

| CMake option | Default | Resource |
| --- | --- | --- |
| `AOTX_AUDIO_FEATURE_ROWS` | 16384 | Resident rows of 4096 F32 values |
| `AOTX_AUDIO_WORKERS` | 1 | Concurrent encoder workspaces |
| `AOTX_AUDIO_SOURCE_FRAMES` | 1440000 | Source frames per workspace |
| `AOTX_KV_SLOT_PAGES` | 0 | Per-slot page capacity; zero selects the build profile |
| `AOTX_KV_POOL_PAGES` | 0 | Shared physical page capacity; zero selects the build profile |

An audio feature row costs 16384 bytes. Model weights and worker buffers require additional VRAM.
Capacities are independent. Increasing source slots does not allocate one encoder workspace per slot.

Requests reserve their complete prompt and reply cache requirement before decoding.
An audio runtime reserves its configured physical KV pool at startup.
Startup refuses allocations that do not fit; it does not silently offload a model or active memory.

An optional `audio.profile` selects capacities within the build limits.
It contains exactly 64 bytes:

| Byte | Field |
| --- | --- |
| 0 | Eight bytes `AOTXAU01` |
| 8 | Schema 1, unsigned 32-bit |
| 12 | Feature rows, unsigned 32-bit |
| 16 | Worker count, unsigned 32-bit |
| 20 | Source frames per worker, unsigned 32-bit |
| 24 | 40 zero bytes |

All fields are little-endian. An omitted directory profile selects the build defaults.
[Complete runtime files](28-runtime-files.md) include the audio pair, audio profile and common source profile.
The required audio flag is 4 in the runtime index feature field.
Activation checks the selected parent, exact component bytes and supported capacities.

## Combined 12 GiB configuration

The checked configuration holds the audio pair, Qwen3.5-0.8B Q8_0 image parent and Qwen3-Embedding-0.6B Q8_0.
It uses the matching F16 vision component and one workspace per media component.
Configure the shared cache before building:

```sh
cmake -S . -B build -DAOTX_KV_POOL_PAGES=640
cmake --build build --parallel
```

Create both capacity profiles in the prepared model store.
Replace `/path/to/store` with that store's absolute path:

```python
from pathlib import Path
import struct

store = Path("/path/to/store")
media = struct.pack("<8sIIQIIIII4xQ24x", b"AOTXIM01", 1,
    128, 67108864, 16384, 1, 4194304, 8192, 4096, 8388608)
audio = struct.pack("<8sIIII40x", b"AOTXAU01", 1, 8192, 1, 1440000)
(store / "media.profile").write_bytes(media)
(store / "audio.profile").write_bytes(audio)
```

These profiles use 64 MiB for shared source bytes, 16384 image rows and 8192 audio rows.
The 640 physical cache pages reserve 1280 MiB.
Set each conversation page quota for its intended prompt and reply size, as described above.
The complete runtime packager includes both profiles.
Available VRAM must cover the complete runtime and other device allocations.

## Source recovery

The GPU retains active source bytes and feature rows. The drain writes the recovery copy behind it.
Recovery rebuilds native audio rows from the recorded canonical bytes before new requests use them.
Source retirement frees GPU capacity but preserves historical source records in the recovery log.
The complete file cap, publication costs and durable-ready rules still apply.
