<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Modules

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Install roles, skills and tools as named directories with a manifest and optional body or executable assets.
The feeder transfers file bytes; the GPU validates and commits them to its catalog.
The journal preserves accepted module text and identity for recovery.

<details>
<summary>On this page</summary>

- [The module directory](#the-module-directory)
- [The manifest keys](#the-manifest-keys)
- [The import](#the-import)
- [The catalog](#the-catalog)
- [What a restore does](#what-a-restore-does)
- [The commands](#the-commands)
- [The limits of this version](#the-limits-of-this-version)
- [Tool selection](#tool-selection)

</details>

## The module directory

A module normally contains `module.manifest` with one `key: value` pair per line.
Keys use lowercase text. Values extend to the end of the line, without quoting or escapes.
A line beginning with `#` is a comment.

A skill can instead contain only `SKILL.md` with `name` and `description` in its initial header:

```text
---
name: arithmetic
description: Gives rules for arithmetic tasks.
---
Check each operation and state the unit of each result.
```

Save this file as `skills/arithmetic/SKILL.md`, then enter `import skills/arithmetic` in the console.
Use `skills` to inspect the accepted entry.
A role can include it through its `skills` key, or an agent can request it with `skill_use`.

A separate `module.manifest` takes precedence over the embedded skill header.
The supplied conductor, worker and verifier directories are under `modules/roles/`.
Their `overlay.txt` files contain role instructions.

## The manifest keys

Every kind accepts these keys.

| key | meaning |
| --- | --- |
| `kind` | `skill`, `role` or `tool` |
| `name` | 1 to 63 bytes of `a` to `z`, `0` to `9` and underscore; the identity in the catalog |
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
| `pages_least` | minimum pages for automatic allocation; default 16 |
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

The feeder reads the manifest and body, then publishes an IMPORT head and bounded parts.
The device validates the completed content before replacing an installed entry.
An invalid key, value, length or incomplete transfer cannot become an installed module.
The console and bus report acceptance or the refusal reason.

`--modules DIR` imports each immediate child directory in name order at startup.
It does not recursively search deeper groups.
The default directory is the build's `modules/roles` path; a moved installation must supply a valid path or packaged assets.
Installing the conductor role creates the console agent in slot zero.

Standard input, attached terminal input and window input reach the same import operation through their respective transport routes.
They do not execute the path as a shell command.
An import of an existing name replaces that module at commit.
A new import can also clear an incomplete arrival of that name.

## The catalog

The catalog contains profile-bounded entries and a byte arena for manifests and bodies.
`AOTX_MODULE_SLOTS` and `AOTX_CATALOGUE_BYTES` define those limits.
Freed adjacent arena runs can combine; an import without enough contiguous room reports its required bytes.

| State | Meaning |
| --- | --- |
| `free` | No module occupies the entry. |
| `arriving` | An import is receiving its parts. |
| `installed` | Validation completed and the runtime can use the module. |
| `refused` | The entry retains its name and refusal reason. |

Nine built-in tools exist before the first tick:

| Tool | Execution | Result |
| --- | --- | --- |
| `memory_recall`, `memory_write`, `skill_use` | Device | Bounded text from device state. |
| `fs_read` | Disk | Digest followed by served file bytes. |
| `fs_stat` | Disk | Size, modification time and digest. |
| `fs_list`, `fs_write`, `fs_update`, `run` | Disk | Bounded operation result. |

Memory tools require a ready embedding pass.
If that requirement is absent, a requested memory call returns an error and ordinary language conversation remains available.
Device results use the shared result framing and appear in transcripts.
Built-in tools cannot be removed. File writes and process execution retain mandatory operator authorization.

The model prompt lists only ready tools that its role, template and effective selection permit.
The system block also carries the role overlay and selected skill text.
Catalog-list and prompt bounds still apply; status output reports omitted definitions or shortened results.
A skill result contains the exact installed body, or an error when the skill is absent.

## What a restore does

Restore reapplies recorded IMPORT and REMOVE operations.
Changed manifest or body files do not silently alter restored catalog text.
An incomplete recorded import never becomes installed; replay clears incomplete entries and reports them.

Executable assets have a separate identity check.
A device tool's PTX must match the digest recorded by its import before the driver loads it.
Directory-based recovery needs that exact file.
The complete runtime profile packages data-only roles and skills; it refuses imported tool programs.
Host executables remain trusted programs with the account's rights.

## The commands

| Command | Effect |
| --- | --- |
| `modules [kind]` | List names, kinds, states, versions and refusal reasons. |
| `module NAME` | Inspect the manifest and body preview. |
| `skills`, `roles`, `tools` | List the corresponding module kind. |
| `import PATH` | Transfer and validate a module directory. |
| `remove NAME` | Remove an eligible installed module. |
| `spawn ROLE` | Create an agent with an installed role. |

Removal refuses an unknown name, incomplete arrival, role in use, active tool or built-in tool.
Accepted removal is a class A journal operation.

## The limits of this version

Imported device tools execute as tick-graph nodes. Imported host tools execute through the feeder.
A call carries manifest-declared arguments in their declared order.
The [tool SDK](10-tool-sdk.md) defines the exact ABI, output bounds and checks.

Turn records retain numeric IDs for built-in tools and zero for imported modules.
Console and bus output retain the imported module name.
Installation and successful structural checks do not establish that executable code is safe or semantically correct.

## Tool selection

Tool selection is optional. Its default enables every tool that the role grants and the model can use.
The selection does not grant a tool that the role excludes or remove required operator authorization.

Use these commands to set instance defaults and conversation choices:

```text
tool all off
tool memory_recall on
agent 1 tools memory_recall on
agent 2 tools memory_recall off
agent 2 tools memory_recall inherit
agent 1 tools
```

An explicit conversation choice overrides the instance default in either direction.
`agent <id> tools all inherit` removes every conversation override.
`agent <id> tools all off` disables every model-callable tool for that conversation.
Changes apply when the next turn starts. An active turn retains its selection.

The names are `memory_recall`, `memory_write`, `fs_read`, `fs_list`, `fs_write`,
`fs_update`, `run`, `skill_use`, `fs_stat`, and `imported`.
The `imported` group controls all imported tools together. Their individual role grants still apply.
`tool all off` includes this group.

The prompt omits disabled definitions. With all tools off, it also omits tool-use instructions.
A generated call to a disabled tool receives a refused result. The tool does not execute.
Disabling memory tools does not disable conversation history, internal recall, or compaction.
Role overlays and installed skill bodies retain their original text.

The `tools.mask` setting stores instance defaults. Its default is 1023; zero disables all groups.
Bits zero through nine follow the name order above. For a fresh boot, put the setting in the settings file.

Conversation choices use class A `tools.agent.<id>` settings records.
Each group has two bits: zero inherits, one disables, and two enables. Three is invalid.
The journal restores these choices. Use the named agent commands to change them.

In CTRL, open **System tools** in a conversation.
The table shows the instance default, the conversation override, and the effective next-turn state.

**Unavailable** means the selection is enabled but no granted, available tool exists in that group.
Use **Refresh** after a model or role change to read its current availability.


CTRL reads tool status from the per-boot `tools.jsonl` stream.
Console text and generated replies cannot change the displayed policy.
The console derivation option enables this stream. A stopped connection keeps one status message until a refresh or reconnection.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
