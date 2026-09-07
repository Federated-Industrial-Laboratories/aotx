# Modules

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| drain | the disk-side process that writes the outbound ring to the journal (a log writer) |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| catalog | the GPU-resident registry of imported modules: skills, roles and tools |
| profile | a build-time table-size configuration for one class of card |
| bus | an append-only message log between agents (a message bus) |
| arena | a contiguous memory region for offset-addressed allocations |

A module is one thing an operator installs: a skill, a role or a tool. A module is a
directory with a manifest and the files the manifest names. This document states what a
module directory contains, how a module reaches the device, what the catalog contains, and what
the commands do.

## The module directory

A module directory contains `module.manifest`. The manifest is plain text with one key and one
value a line. A key is in lower case. A value runs to the end of the line. A line that
starts with a number sign is a comment. There is no quoting and there is no escape.

A skill directory may contain `SKILL.md` and no manifest. The feeder then reads that file whole
and sends it as the body, with no manifest beside it. The device splits the file. The head
appears between two lines of three dashes and is the manifest.

The text after the head is the body. The head contains two keys, `name` and `description`, and
no other key. A file with no head is refused. A `module.manifest` beside a `SKILL.md` wins.

This is a complete skill directory file:

```
---
name: arithmetic
description: Gives rules for arithmetic tasks.
---
Check each operation and state the unit of each result.
```

Put it in `skills/arithmetic/SKILL.md`. Type `import skills/arithmetic` at the console.
The `skills` command then lists it. A role names the skill in its `skills` manifest key, or an
agent calls `skill_use` with its name.

The repository carries the three roles of a run in `modules/roles/`. Each one contains a
`module.manifest` and an `overlay.txt` with the duty sentence of the role.

## The manifest keys

Every kind accepts these keys.

| key | meaning |
| --- | --- |
| `kind` | `skill`, `role` or `tool` |
| `name` | 1 to 63 bytes of `a` to `z`, `0` to `9` and the low line; the identity in the catalog |
| `description` | one line the model reads in the tool list or the skill list |
| `version` | free text, which the `modules` command shows |
| `body` | the file that holds the text of a skill or the overlay of a role |

A role also accepts these keys.

| key | meaning |
| --- | --- |
| `model` | a role name of the model file list: `language`, `language-q4`, `embedding` or `reranker`; a role of a run names a language file, because the two small models open no reply |
| `tools` | tool names with commas between them; an unknown name gives no tool |
| `authorise` | tool names that need the operator for this role |
| `budget` | turns for each task or operator input, including automatic continuation; zero takes the setting `agent.budget` |
| `pages` | transcript pages of the role; zero takes the setting |
| `skills` | skill names whose bodies go in every prompt of the role |

A tool also accepts these keys.

| key | meaning |
| --- | --- |
| `side` | `device` or `host` |
| `arguments` | argument keys with commas between them; at most four, string values only |
| `authorise` | `always` or `never` |
| `deadline` | ticks a reply may take; zero takes the setting `tool.deadline_ticks` |
| `timeout` | seconds the feeder lets a program run |
| `module` | the PTX file of a device tool |
| `entry` | the kernel symbol in that file |
| `program` | the executable of a host tool, beside the manifest |
| `example` | one argument line for the check program |
| `sha256` | the digest of the module file |

The import refuses a key that is not valid for the kind. The import also refuses an invalid value.

## The import

The feeder reads a module directory and publishes the bytes as IMPORT records. The head
record names the kind, the name, the file count and the bytes of each file. The parts that
follow carry the text of the manifest and then the text of the body.

Every IMPORT record is class A. The journal contains the bytes, so a restore builds the catalog
from the journal and reads no manifest and no body from a file. A skill or a role that
changes on the disk after an import stays as it was imported. The next import of that name
replaces it.

A device tool is code, and the driver loads code from a file. The host glue therefore opens
the module file of every device tool of the catalog again after a restore. It computes the
digest of that file and compares it with the digest the import carried. A file that changed
is refused with the reason, and the entry enters the state `refused`. A restored run thus
never loads code the journal does not name.

`aotx_boot --modules <dir>` names the directory of module directories. The feeder imports
each directory below it in name order, before the first line of the operator. The walk goes
one level deep, so the value names the directory that contains the modules. The default is the
`modules/roles` directory of the build, which contains the three roles. The console agent uses slot 0 in the tick that the role
named `conductor` is installed.

The feeder reads the console line `import <path>` of its own standard input and imports the
directory. A line typed in the window does not pass the feeder. The device writes one
request record for such a line, and the drain gives that record to the feeder. Both routes
end in the same import records.

A directory the feeder refuses gives one line of the shape `import <path> refused: <reason>`.
The device writes that line to the console and to the bus as a note.

## The catalog

The catalog is the device table of installed modules. It comprises `AOTX_MODULE_SLOTS` entries
and an arena of `AOTX_CATALOGUE_BYTES`, which the build profile gives
(`cuda/profile/12g.cuh`). The arena contains the manifest text and the body of every module as
runs of an offset and a length.

An entry has one of four states.

| state | meaning |
| --- | --- |
| `free` | the entry holds no module |
| `arriving` | the head of an import claimed the entry and the parts fill it |
| `installed` | the commit took the module and the run may use it |
| `refused` | the commit did not take the module; the entry keeps its name and its reason |

The commit runs when every byte of every file has arrived. It reads the manifest on the
device, checks the entry, and gives the entry the state `installed` or `refused`. Each
outcome writes one console line and one bus note with the reason.

An import replaces an existing module of the same name at the commit. The runs of the
replaced module return to a free list, which joins adjacent runs. The arena refuses the import
when no run is available for a file, and reports the required bytes.

A new import cancels an incomplete import of the same name. The runs of the incomplete import
return, and the new head claims the entry. A restore of a partial import leaves such an entry.
The next import of that name clears the entry.

Nine built-in tools reside in the catalog before the first tick. Three run on the
device. Six run on the disk side. They are entries of the same shape as an imported tool.

| tool | side | result |
| --- | --- | --- |
| `memory_recall`, `memory_write`, `skill_use` | device | device text |
| `fs_read` | disk | `sha256: <64 hexadecimal characters>` as the first line, then the file bytes |
| `fs_stat` | disk | the size, modification time and digest, with no file bytes |
| `fs_list`, `fs_write`, `fs_update`, `run` | disk | the result of the operation |

The memory pair requires a ready embedding pass before it appears in a prompt.
Install an embedding model and include `embedding` in the selected `--roles` list.
An unavailable memory call returns an error at once, even if the model calls an unlisted tool.
Device tool results use the same result body as host replies and appear in the transcript.

The `remove` command cannot remove a built-in tool. The three tools that write or run require
operator authorization for every call (`docs/10-tool-sdk.md` shows a role manifest that grants them).

The selected row places tool instructions, the role duty sentence, skill bodies, and the skill list in its system block.
The device lists ready tools that the role permits; a newly installed tool can appear in the next prompt.
The block contains at most `AOTX_CATALOG_LIST_BYTES`, including its closing text and call instructions.
A role that allows more tools than fit gets the first that fit, and the `modules` command
states the count that was cut.
Unsupported templates have no tool list, and role overlays do not change call forms.
The forms and provenance errors are specified in `docs/07-operation.md`.

`skill_use` is a device tool with one argument, `name`. It copies the body of that skill
into the result of the request, and the next prompt of the agent carries it. An unknown
catalog name gives an error result which names it.

The prompt of a turn uses the room that is left after the system block. A result longer
than that room is cut to the room, and the included bytes report the cut. The
`modules` command states the count of the results that were cut.

## What a restore does

A restore replays the IMPORT records and the REMOVE records of the journal. The catalog
after the replay contains the modules that were present at the crash. The restore reads no module
directory. A module the operator changed on the disk after the import does not come back
until the next import of that name.

A run that stopped in the middle of an import leaves a head in the journal with no last
part. Such an import never completes. Replay removes every incomplete import and reports the
count on one console line. The import number is unique while an import is active. Removal
makes the number available again.

## The commands

| command | what it does |
| --- | --- |
| `modules [kind]` | the catalog: name, kind, state, version, and the reason of a refused entry |
| `module <name>` | one module in full: the manifest, and the first lines of the body of a skill |
| `skills` | the skills of the catalog |
| `roles` | the roles of the catalog |
| `tools` | the tools of the catalog |
| `remove <name>` | take one module out of the catalog |
| `import <path>` | the feeder reads the directory and publishes the import |
| `spawn <role>` | make an agent of a role of the catalog |

`remove` writes a class A record at the commit of the tick. It refuses an unknown name, an
arrival, a role in use, a tool in flight or a built-in tool.

Each refusal gives one line with the reason.

## The limits of this version

An imported tool module resides in the catalog and appears in the lists. The `module` command
reports it. A device tool runs as a node of the tick graph. A host tool runs as a program of
the feeder. `docs/10-tool-sdk.md` specifies each contract. The nine built-in tools run.

A call carries every argument value the manifest names, in the order of the manifest.

The record of a turn carries the number of a built-in tool, and zero for an imported tool module.
The console line and the bus note for a module include its name.
