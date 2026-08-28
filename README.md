# aotx

A local inference operating system designed in CUDA. The GPU holds the authoritative state:
agents, models, a message bus, a text interface and a command line. The disk holds a copy that
is one tick behind. The GPU never waits for the disk.

## Status

Version 0.0. The repository holds the seam: the device ring, the per-tick flush into a pinned
host ring, and the disk-side drain, feeder and restore programs. No release exists.

## Requirements

- CUDA Toolkit 13.2 or later, and a driver that supports it
- A GPU with compute capability 8.6
- CMake 3.28 or later, Ninja
- Python 3.10 or later, for the gates

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

## Tests

`ctest --test-dir build` runs every check that needs no desktop. The interface is part of
it. The check `raster_headless` makes its drawing context on a display device. It then runs
the raster graph for 60 frames through the pixel buffer path. It compares the pixels with
the figure the host computes. No window opens.

Three groups of checks are registered only when a build asks for them:

```
cmake -S . -B build -G Ninja -DAOTX_DISPLAY_TESTS=ON     # the window check, label display
cmake -S . -B build -G Ninja -DAOTX_FAULT_TESTS=ON       # the guard gap checks
cmake -S . -B build -G Ninja -DAOTX_SANITIZER_TESTS=ON   # the sanitizer gate, label sanitizer
```

The sanitizer gate (`sanitizer_memcheck` and `sanitizer_racecheck`) runs
`tools/sanitizer-gate.sh`. Each arm takes its own checks. The memcheck arm reads every
access to memory, so it takes the checks that drive the tick graph: `seam`, `decode` and
`agent`. The racecheck arm reads every access to shared memory, so it takes the checks that
use shared memory by hand: `seam`, `ui` and `matrix`. Each run takes minutes; run them alone
with `ctest --test-dir build -L sanitizer`. The gate sets `AOTX_SANITIZER`, which those
checks read to take fewer ticks and to leave their rate cases out, and it opens no window.

The window check opens a window on the display of the operator; run it alone with
`ctest --test-dir build -L display`. The same program sends the close request of a window
manager to a window of a title. A script stops a run with a window that way:

```
build/aotx_window_test --close AOTX-1
```

No check and no tool of this repository destroys or kills the window of another program.

## Replies

A run loads model files from the directory that `--models` names. The option `--roles` names
the roles to load, with commas between them: `embedding`, `reranker`, `language` and
`language-q4`. A run that gives no list loads `embedding,reranker,language`.

```
aotx_boot --journal build/run --models models --roles language
```

With a language model resident, the command `say <text>` sends the text to it. The command
wraps the text in the chat template that the model file carries, with thinking off. It then
opens a sequence on the slot of the conductor. The console shows a line that starts with
`conductor: `, and the reply grows that line as the tokens come. A newline byte in the reply
starts a new line. At the end of the reply one bus message states the token count and the
ticks the reply took.

One reply runs at a time. A second `say` while the conductor is not idle is refused. The
command `stop` ends the reply that runs.

## Agents

An agent is a record, a sequence slot and a share of the arena. Agent 0 is the conductor,
which the command `say` sends its text to. The commands that make and drive agents are:

```
spawn <role> [n]                    make n agents of a role; n is 1 to 8
task <agent|role> <text> [verify]   open a task for an agent or for a role
authorise <id>                      let a tool request of that number run
refuse <id>                         stop a tool request of that number
agents                              show the agents
```

A role is `conductor`, `worker` or `verifier`. The word `verify` at the end of a task asks a
verifier agent to judge the result. The text of a `say` and the text of a `task`
hold 160 bytes at most. A longer text is refused, and the line names the bound.

The command `agents` and the agents panel show one row for each agent that is not free. A
row holds the identity, the role and the state. It then holds the task in hand, the tool of
a request that waits and the number of that request. It ends with the turns taken, the reply
tokens and the reply tokens each second.

A tool that reads a file waits for the operator. The agents panel lists each request that
waits with its number, its agent, its tool and the first bytes of its argument. The `Tab`
key moves the focus between the console and the agents panel, and the panel with the focus
shows a bright title.

The command line takes no key while the focus is on the panel. The key `y` grants the first
request that waits and the key `n` refuses it. Each answer writes a console line that names
the request and the answer. The commands `authorise` and `refuse` answer any request by its
number.

## Load runs

A run with a tick load writes many records for each tick. The drain makes a line of text for
every record of the types it derives. At 12,000 records a tick, those lines fill a disk in
minutes. Give `--derive` to name the types the drain makes lines from:

```
aotx_boot --journal build/run --workload 12000 --derive console,bus
```

The names are `console`, `note`, `bus`, `bulk`, `sequence`, `requests` and `none`, with
commas between them. A run that gives no list leaves the drain with its default, which is
every type. The journal keeps every record, whatever the list holds; the list changes the
derived files only. The name `sequence` makes one line at the end of a reply, with the slot,
the token counts and the ticks. A token record makes no line and stays in the journal
segments. The name `bus` covers the message records and the task and agent events, because
all three make message lines.

The chain of turns is not in the list. A turn that makes no line makes a gap in the chain,
and a chain with a gap proves nothing.

A console record that carries the fragment flag continues the line before it, in the console
log and on the terminal. A reply that comes one record at a time therefore reads as one line.

A short list makes the drain faster, and a faster drain lets the tick load make more
records. Measured on one machine at `--workload 12000` for 30 seconds:

| derive | journal | derived lines | ticks held |
| --- | --- | --- | --- |
| default | 755 MB | 482 MB | 2851 of 2944 |
| `console,bus` | 6.4 GB | 0 | 730 of 2943 |

Give `--records` or a smaller `--workload` to bound the size of a journal. The option
`--derive` bounds the derived lines only.

A run stops at the `quit` command, at the close request of the window manager, and at the
signals SIGTERM and SIGINT. Each of them ends the run the same way: the last flush, the
closed rings, the wait for the disk side programs, and the reports. A second signal changes
nothing, because the run is already stopping. An operator who must end a run that stopped
answering sends SIGKILL, and knows what that leaves behind.

## Tool requests and file reads

An agent that calls the tool `fs_read` writes a request record. The drain turns that record
into one line of `<journal>/requests.jsonl`:

```
{"request":1000,"agent":0,"turn":1,"tool":"fs_read","arg":"notes/one.txt","deadline":507,"auth":"none","tick":7}
```

The field `auth` is `none` for a tool that needs no authorization and `granted` for a tool
the operator authorized. A request that waits for the operator makes its line when the
record that grants it comes. A request the operator refuses makes no line.

Every field of that line comes from the request and none from the record that grants it. The
field `tick` is therefore the tick the request was made at, which is the tick the `deadline`
beside it counts from.

The feeder reads that file and executes the requests:

```
aotx_feed --inbound-fd 3 --root /home/user/notes --requests build/run/requests.jsonl
```

`--root` names the one directory a file read may reach. It is the security boundary of the
system. Every component of a path is opened with `O_NOFOLLOW`, so a symbolic link at any
depth is refused. A component of two dots is refused. A path that starts at the root of the
file system is refused.

A path that names anything other than a regular file is refused. A read takes 4,096 bytes at
most, which is the size of the result buffer of an agent. A larger result does not fit the
sequence of the turn beside the text of the role.

The answer is a reply record, or several. One reply is `parts` records with the same agent
and request, from part 0. A part with the status `ok` carries content, in order. A part with
any other status is the last part of the reply, and its bytes are the reason.

A file that the cap cut gives the parts of its first 4,096 bytes and one more part that
states the cut. A path the root rule refuses gives one part with the status `refused`. A file
that is not there gives one part with the status `error`.

One request is executed one time. The feeder holds the identities of the last 1,024 requests
and executes no identity twice. A requests file that is already there when the feeder starts
is read from its end. A feeder that starts after a restore therefore executes no request of
the run before it. The device applies the replies that the journal holds.

The feeder writes no `late` status, because it does not know the tick. The deadline of a
request is measured on the device, which holds the tick.

## The turns of a run

The drain writes one line for each completed turn to `<journal>/manifest/<boot id>.jsonl`:

```
{"agent":0,"turn":1,"input_hash":"1111000000000000","output_hash":"2222000000000000","tokens":7,"finish":"stop","tool":"fs_read","request":1000,"prev":"<64 hexadecimal characters>"}
```

The field `prev` is the SHA-256 digest of the bytes of the line before it, with the end byte
of that line in it. The first line of a file carries 64 zeros. A line that is taken out, or a
byte that changes, therefore breaks every line after it.

## Reading a journal

`aotx_journal` prints the records of a journal as text, one record for each line:

```
aotx_journal tokens build/run --boot 00000000cafe0001
aotx_journal manifest build/run
aotx_journal requests build/run
```

The command `tokens` prints the token records of a run. The directory is a boot directory when
it holds segments. If it does not, it is a journal directory: `--boot` names the boot in it,
and with no `--boot` the newest complete boot is read. The first four fields of a line are the
token itself, so a comparison of two runs cuts each line after them. The field `sampled` is
one for a token the model made. The field `replayed` is one for a token a restore applied
again.

The command `manifest` prints the turns of a run and verifies the chain. It recomputes the
digest of each line and compares it with the field that the line after it carries. The
command ends with status 0 when every chain holds. It ends with status 1 at the first line
that breaks a chain, and the report names that line. With no `--boot` it reads every chain
file of the journal.

The command `requests` prints the tool requests of a journal, one for each line. A line
holds the identity, the agent, the tool, the state of the authorization, the deadline and
the path.

## Gates

Run `tools/gate.sh` before each commit. It examines the staged files with three checks:
`tools/ste-lint.py` (text register), `tools/size-gate.py` (file size), `tools/seam-gate.py`
(host and device separation). Give paths to examine files that are not staged.

## Layout

```
cuda/    device modules, one directory for each module; host glue files end in _host.cu
ptx/     kernels written in PTX and loaded as modules
disk/    C programs and one library for the disk side; no CUDA dependency
tests/   one test program for each module; each test runs at N=1 and N=64
docs/    reference documentation; start at docs/00-writing.md
tools/   the gates
```

## Where to start reading

Read `docs/00-writing.md` for the writing rules, then `cuda/seam/wire.h` for the layouts that
cross the seam.

## License

Apache License, Version 2.0. See LICENSE.
