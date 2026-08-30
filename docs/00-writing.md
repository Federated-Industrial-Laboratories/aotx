# Documentation

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

## Writing rules

All text in this repository follows ASD-STE100 Simplified Technical English: comments, this
documentation, command help, interface strings and commit messages.

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
