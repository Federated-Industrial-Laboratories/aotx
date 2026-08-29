# Modules

A module is one thing an operator installs: a skill, a role or a tool. A module is a
directory with a manifest and the files the manifest names. This document states what a
module directory holds, how a module reaches the device, what the catalog holds, and what
the commands do.

## The module directory

A module directory holds `module.manifest`. The manifest is plain text with one key and one
value a line. A key is in lower case. A value runs to the end of the line. A line that
starts with a number sign is a comment. There is no quoting and there is no escape.

A skill directory may hold `SKILL.md` and no manifest. The importer then reads the head of
that file, which stands between two lines of three dashes. That head gives two keys, `name`
and `description`, and no other key. The text after the head is the body of the skill. A
`module.manifest` beside a `SKILL.md` wins.

The repository carries the three roles of a run in `modules/roles/`. Each one holds a
`module.manifest` and an `overlay.txt` with the duty sentence of the role.

## The manifest keys

Every kind takes these keys.

| key | meaning |
| --- | --- |
| `kind` | `skill`, `role` or `tool` |
| `name` | 1 to 63 bytes of `a` to `z`, `0` to `9` and the low line; the identity in the catalog |
| `description` | one line the model reads in the tool list or the skill list |
| `version` | free text, which the `modules` command shows |
| `body` | the file that holds the text of a skill or the overlay of a role |

A role takes these keys as well.

| key | meaning |
| --- | --- |
| `model` | `language` or `language-q4` |
| `tools` | tool names with commas between them; an unknown name gives no tool |
| `authorise` | tool names that need the operator for this role |
| `budget` | turns for each task; zero takes the setting `agent.budget_turns` |
| `pages` | transcript pages of the role; zero takes the setting |
| `skills` | skill names whose bodies go in every prompt of the role |

A tool takes these keys as well.

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

A key that the kind does not take refuses the import. A value that the key does not take
refuses the import as well.

## The import

The feeder reads a module directory and publishes the bytes as IMPORT records. The head
record names the kind, the name, the file count and the bytes of each file. The parts that
follow carry the text of the manifest and then the text of the body.

Every IMPORT record is class A. The journal holds the bytes, so a restore builds the catalog
from the journal and opens no file. A module that changes on the disk after an import stays
as it was imported. The next import of that name replaces it.

`aotx_boot --modules <dir>` names the directory of module directories. The feeder imports
each directory below it in name order, before the first line of the operator. The walk goes
one level deep, so the value names the directory that holds the modules. The default is the
`modules/roles` directory of the build, which holds the three roles. The agent of the console takes slot 0 in the tick that the role
named `conductor` is installed.

The console line `import <path>` reaches the feeder and not the device. A line of that shape
that reaches the device gives one line which says so.

## The catalog

The catalog is the device table of installed modules. It holds `AOTX_MODULE_SLOTS` entries
and an arena of `AOTX_CATALOGUE_BYTES`, which the build profile gives
(`cuda/profile/12g.cuh`). The arena holds the manifest text and the body of every module as
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

An import of a name that stands replaces that module whole at the commit. The runs of the
module that went go back to a free list of runs, which joins runs that touch. An arena that
holds no run for a file refuses the import and states the figure.

The device puts four built-in tools in the catalog before the first tick: `memory_recall`,
`memory_write`, `skill_use` and `fs_read`. They are entries of the same shape as an imported
tool. A built-in tool does not go out with `remove`.

The prompt of a turn starts with the duty sentence of the role. It then holds the bodies of
the skills the role names, the tool list and the skill list. A kernel builds the tool list
from the entries the mask of the role allows. A tool installed in one tick therefore stands in a prompt of the
next tick, with no host in the path. The block takes `AOTX_CATALOG_LIST_BYTES` at the most.
A role that allows more tools than fit gets the first that fit, and the `modules` command
states the count that was cut.

`skill_use` is a device tool with one argument, `name`. It copies the body of that skill
into the result of the request, and the next prompt of the agent carries it. A name the
catalog does not hold gives an error result which names it.

## What a restore does

A restore replays the IMPORT records and the REMOVE records of the journal. The catalog
after the replay holds the modules the run held at the crash. The restore reads no module
directory. A module the operator changed on the disk after the import does not come back
until the next import of that name.

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

`remove` writes a class A record at the commit of the tick. The catalog holds a name back
for four reasons. No module holds that name. An agent runs on that role. A request of that
tool is in flight. The entry is a built-in tool.

Each refusal gives one line with the reason.

## The limits of this version

A tool that comes in as a module goes in the catalog, stands in the lists and shows with
`module`. A call to one gives an error result. The node of a device tool and the program of a host
tool come with the tool module contract. The four built-in tools run.

A call carries one argument value. A tool of one value key beside `provenance` therefore
gives its whole call.

The record of a turn carries the number of a built-in tool, and zero for a tool that came in
as a module. The console line and the bus note of a module name it.
