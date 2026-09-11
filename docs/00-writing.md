# Documentation

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| catalog | the GPU-resident registry of imported modules: skills, roles and tools |
| profile | a build-time table-size configuration for one class of card |
| bus | an append-only message log between agents (a message bus) |

This file is the index of the documentation set. Read the architecture first. Read the build
and operation documents before you start a system.

| document | subject |
| --- | --- |
| `01-architecture.md` | the device, the disk side, the mirror, the terminal and the memory tiers |
| `02-temporal-model.md` | ticks, record classes and restore order |
| `03-seam-contract.md` | byte layouts and publication across the seam |
| `04-journal-format.md` | segment files, record bodies and derived files |
| `05-bus-schema.md` | messages on the device and JSON lines on the disk |
| `06-build.md` | requirements, profiles, build options, checks and gates |
| `07-operation.md` | start, commands, settings, models, agents and restore |
| `08-measured.md` | measurements from the reference card |
| `09-modules.md` | skills, roles, tools, import and the catalog |
| `10-tool-sdk.md` | the device and host tool contracts |
| `11-terminal.md` | terminal options, screens and keys |
| `12-conduct.md` | steer vectors, voice profiles and the conduct commands |
| `13-control.md` | the graphical control program, its windows and its attach |
| `14-affect.md` | the affect substrate, the quality stream and their optional build |
| `15-quality.md` | the conversation quality instrument: the score tool, the pair script and the fixtures |
| `16-model-files.md` | header inspection, file verification, model stores, and model use |
| `17-accuracy.md` | fixed full-row references, calibration bounds and accuracy checks |
| [17-ccir.md](17-ccir.md) | bounded container sections, file transactions, recovery and compaction |
| [18-typed-state.md](18-typed-state.md) | GPU object admission, exact media state, recorded replay and checkpoint export |
| [19-prepared-memory.md](19-prepared-memory.md) | prepared GPU recall, bounded context and saved selection replay |
| [20-live-memory.md](20-live-memory.md) | live conversation bindings, typed input, memory prompts and journal restore |
| [21-text-memory.md](21-text-memory.md) | GPU query preparation from bounded text, recorded vectors and exact replay |
| [22-memory-retention.md](22-memory-retention.md) | retain accepted input, typed vectors, working focus and journal recovery |
| [29-image-input.md](29-image-input.md) | native image input, source scopes, device capacity and portable image runtimes |

## Writing rules

Project comments, documentation, command help, interface strings and commit messages follow
the repository's ASD-STE100 Simplified Technical English rules.
Third-party source and license text retain their original wording.

## Rules

- Write short sentences. A descriptive sentence has 25 words or fewer. A procedural sentence has
  20 words or fewer.
- Write short paragraphs. A paragraph has six sentences or fewer.
- Use one term for one thing.
- Use the active voice and the present tense. Give one instruction in each sentence.
- Do not write in the first person. Do not write conversation.
- Use ASCII punctuation only.
- A comment gives the constraint, the invariant or the reason for the code it is attached to,
  and nothing else.

## The gate

`tools/ste-lint.py` checks sentence length, paragraph length, punctuation and a list of refused
words in `tools/ste-words.txt`. The gate is an approximation of the standard. The full approved
word list of ASD-STE100 is not part of this repository. A sentence that passes the gate can
still break a rule of the standard; a manual check finds the rest.

## Spelling

The repository uses American English spelling, as ASD-STE100 does. The gate refuses common
British spellings.
