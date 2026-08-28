# Operation

This document states how a run starts and what the window shows. It then states what each
console command does. It ends with what a run leaves on the disk and how a run stops.

## Start a run

`aotx_boot` starts the system. It takes these options:

| option | what it does |
| --- | --- |
| `--journal <dir>` | the directory the journal goes in |
| `--models <dir>` | the directory the model files are in |
| `--roles <list>` | roles of the model file list to load, with commas between them |
| `--root <dir>` | the one directory a file read tool may reach |
| `--restore` | replay the journal before the first input |
| `--window` | show the panels in a window on the display |
| `--ticks <n>` | run this many ticks, then stop; zero runs on |
| `--workload <n>` | records the tick load writes for each tick |
| `--blocks <n>` | blocks of the tick load; the default is 64 |
| `--records <n>` | stop a run that has no tick count at this record count |
| `--derive <list>` | types the drain makes lines from, with commas between them |
| `--solo` | run with no disk-side programs |
| `--clock-only` | run the clock module check and stop |

A run needs a journal directory. A run with `--solo` needs none, because it starts no drain. The
default record count is 1,000,000. The pump makes at most 100 ticks in one second.

The start reads the clock module first and prints its sample. It then prints the run identity
and the sizes of the ring, the scratch arena and the host ring. With a model directory it prints
the digest line and the model line. The end of a run prints the ticks, the records, the blocks,
the ticks held, the records applied and the state hash.

```
aotx_boot --journal build/run --models models --roles language --window
```

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

`--window` opens the window and draws the grid of 160 columns by 50 rows. The grid holds six
panels, and every cell belongs to one of them.

- console: the last lines of the console buffer, with the command line on the last row. The
  buffer holds 256 lines of 160 bytes.
- agents: one row for each agent that is not free, and then the requests that wait. The
  panel gives 10 rows to the agents and 3 rows to the requests.
- bus: the last messages of every kind, the newest first, up to 32 of them.
- arena: the region table and the memory budget. The budget holds the mapped bytes, the
  held bytes, the free bytes, the total bytes, the ring use and the mapped pages.
- tick: the tick, the records, the blocks, the ticks held, the records applied and the
  start time. It ends with the figures of the last statistics record. The last row holds the
  counts of the decode and of the agents.
- seam: the head bytes, the drain bytes, the lag in bytes and in ticks, the block sequence
  and the free bytes. It ends with the figures of the last stall record and the dropped runs.

The editor takes the code points 32 to 126. It takes Backspace, Delete, Left, Right, Home and
End. The Up key and the Down key walk the history, which holds 32 lines. The Enter key gives the
line to the parser.

The `Tab` key moves the focus between the console and the agents panel. The panel with the focus
shows a bright title. The editor takes no key while the focus is on the agents panel. The key
`y` grants the first request that waits, and the key `n` refuses it. Each answer writes a
console line that names the request and the answer.

The close request of the window manager ends the run. The close prints the frames drawn, and the
mean and the worst interval between two frames. It then prints the intervals over 20 ms and the
key events the pipe could not take. No check and no tool of this repository destroys or kills
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
| `spawn <role> [n]` | make n agents of a role; n is 1 to 8 |
| `task <agent\|role> <text> [verify]` | open a task for an agent or for a role |
| `authorise <id>` | let a tool request of that number run |
| `refuse <id>` | stop a tool request of that number |
| `mem` | show the memory regions and the budget |
| `agents` | show the agents |
| `stats` | show the counts of the last tick |
| `quit` | stop the run |

A kind is `finding`, `rank`, `question`, `answer`, `handoff`, `cost` or `note`. A source is
`computed`, `fetched`, `recalled` or `testimony`. A role is `conductor`, `worker` or `verifier`.
An agent is a slot from 0 to 63. The list of the `bus` command shows 32 messages, which fills
the console once.

One line writes at most 32 records. A line that reaches the allowance ends with a line that
states the cut. The `quit` command holds the run while a replay of the journal runs. A `quit`
that a past run typed therefore does not close the run that replays it.

## Replies

A run loads model files from the directory that `--models` names. With a language model
resident, the command `say <text>` sends the text to the conductor agent. The command wraps the
text in the chat template that the model file carries, with thinking off. That wrap gives
temperature 0.7, top_k 20 and top_p 0.8, and the reply holds 256 tokens at most.

The console shows a line that starts with `conductor: `, and the reply grows that line as the
tokens come. A newline byte in the reply starts a new line. The console never shows the bytes of
a control token of the model, so a reply ends with its last text. At the end one bus message
states the token count and the ticks the reply took.

One reply runs at a time. A second `say` while the conductor is not idle is refused. The command
`stop` ends the reply that runs. A `say` with no language model is refused, and a `say` with no
conductor agent is refused with the name of the `spawn` command.

## Agents

An agent is a record, a sequence slot and a share of the arena. The table holds 64 slots. Agent
0 is the conductor, which the command `say` sends its text to. A task gives an agent 8 turns.

The text of a `say` and the text of a `task` hold 160 bytes at most. A longer text is refused,
and the line names the bound. The word `verify` at the end of a task asks a verifier agent to
judge the result.

The system holds three tools. `memory_recall` and `memory_write` run on the device. The tool
`fs_read` crosses the seam to the feeder. The conductor and the worker may call all three, and
their `fs_read` waits for the operator. The verifier may call `memory_recall` alone.

The command `agents` and the agents panel show one row for each agent that is not free. A row
holds the identity, the role and the state. It then holds the task in hand, the tool of a
request that waits and the number of that request. It ends with the turns taken, the reply
tokens and the reply tokens each second. A state is `free`, `idle`, `prompt`, `run`, `tool` or
`post`.

The agents panel lists each request that waits with its number, its agent, its tool and the
first 40 bytes of its argument. The commands `authorise` and `refuse` answer any request by its
number.

## Tool requests and file reads

An agent that calls the tool `fs_read` writes a request record. The drain turns that record into
one line of `<journal>/requests.jsonl`:

```
{"request":1000,"agent":0,"turn":1,"tool":"fs_read","arg":"notes/one.txt","deadline":507,"auth":"none","tick":7}
```

The field `auth` is `none` for a tool that needs no authorization, and `granted` for a tool the
operator authorized. A request that waits for the operator makes its line when the record that
grants it comes. A request the operator refuses makes no line.

A request of a tool that needs no authorization carries a deadline of 500 ticks from the tick of
the request. A request that waits for the operator carries no deadline while it waits. Its
deadline of 500 ticks starts at the tick the operator grants it, so the operator may take any
time to answer.

The feeder reads that file and executes the requests:

```
aotx_feed --inbound-fd 3 --root /home/user/notes --requests build/run/requests.jsonl
```

`--root` names the one directory a file read may reach. It is the security boundary of the
system. Every component of a path is opened with `O_NOFOLLOW`, so a symbolic link at any depth
is refused. A component of two dots is refused. A path that starts at the root of the file
system is refused. A path of more than 64 components is refused.

A path that names anything other than a regular file is refused. A read takes 4,096 bytes at
most, which is the size of the result buffer of an agent. A larger result does not fit the
sequence of the turn beside the text of the role.

The answer is a reply record, or several. One reply is `parts` records with the same agent and
request, from part 0. A part with the status `ok` carries content, in order. A part with any
other status is the last part of the reply, and its bytes are the reason.

A file that the cap cut gives the parts of its first 4,096 bytes and one more part that states
the cut. A path the root rule refuses gives one part with the status `refused`. A file that is
not there gives one part with the status `error`.

One request is executed one time. The feeder holds the identities of the last 1,024 requests and
executes no identity twice. A requests file that is already there when the feeder starts is read
from its end. A feeder that starts after a restore therefore executes no request of the run
before it. The device applies the replies that the journal holds. The feeder writes no `late`
status, because the device holds the tick.

## The turns of a run

The drain writes one line for each completed turn to `<journal>/manifest/<boot id>.jsonl`:

```
{"agent":0,"turn":1,"input_hash":"1111000000000000","output_hash":"2222000000000000","tokens":7,"finish":"stop","tool":"fs_read","request":1000,"prev":"<64 hexadecimal characters>"}
```

The field `prev` is the SHA-256 digest of the bytes of the line before it, with the end byte of
that line in it. The first line of a file carries 64 zeros. A line that is taken out, or a byte
that changes, therefore breaks every line after it.

## The journal a run leaves

A journal directory holds one directory for each boot, named with the identity of that boot.
That directory holds the segments, which are named `seg-000000.seg` and up, and the console log.
A segment holds 64 MB at most.

Beside the boot directories the journal holds three more entries. The directory `bus` holds one
message file for each day. The directory `bulk` holds the payloads and an index. The directory
`manifest` holds one chain file for each boot, and the file `requests.jsonl` holds the tool
requests of the journal.

`aotx_journal` prints the records of a journal as text, one record for each line:

```
aotx_journal tokens build/run --boot 00000000cafe0001
aotx_journal manifest build/run
aotx_journal requests build/run
```

The command `tokens` prints the token records of a run. The directory is a boot directory when
it holds segments. If it does not, it is a journal directory: `--boot` names the boot in it, and
with no `--boot` the newest complete boot is read. The first four fields of a line are the token
itself, so a comparison of two runs cuts each line after them. The field `sampled` is one for a
token the model made, and the field `replayed` is one for a token a restore applied again.

The command `manifest` prints the turns of a run and verifies the chain. It recomputes the
digest of each line and compares it with the field that the line after it carries. The command
ends with status 0 when every chain holds. It ends with status 1 at the first line that breaks a
chain, and the report names that line. With no `--boot` it reads every chain file of the
journal.

The command `requests` prints the tool requests of a journal, one for each line. A line holds
the identity, the agent, the tool, the state of the authorization, the deadline and the path.

## Restore

`--restore` replays the newest complete journal before the first input. The replay sends every
class A record of that journal. It leaves out the boot record and the tick commit record, which
the device makes again on its own. Each replayed record carries a flag that marks it as one the
system applied before.

The operator sees one line at the end of the replay. It states the records applied, the state
hash the device computed, the records refused and the pages mapped. A `quit` typed by the past
run does not close the run that replays it.

A restored reply continues its tokens and not its console line. The console line belongs to the
`say` command that opened the sequence. A sequence that a restore gave back from the token
records has no such line, so the console does not show that reply again.

`aotx_restore --summary` reads a journal and prints its figures without a ring. The line holds
the boot identity, the last tick, the records replayed and the state hash.

## Load runs and the derive list

A run with a tick load writes many records for each tick. The drain makes a line of text for
every record of the types it derives. A load of thousands of records a tick therefore makes
thousands of lines a tick. Give `--derive` to name the types the drain makes lines from:

```
aotx_boot --journal build/run --workload 12000 --derive console,bus
```

The names are `console`, `note`, `bus`, `bulk`, `sequence`, `requests` and `none`, with commas
between them. A run that gives no list leaves the drain with its default, which is every type.
The journal keeps every record, whatever the list holds; the list changes the derived files
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
rings, the wait for the disk-side programs, and the reports. A second signal changes nothing,
because the run is already stopping.

A run that holds a drawing context must never end at the default action of a signal. The display
server keeps the window of a program that stops in the middle of a frame. An operator who must
end a run that stopped answering sends SIGKILL, and knows what that leaves behind. The last
block is not flushed and the rings stay as they are. The journal holds the run up to its last
complete tick. A restore reads that journal and gives the state back.

## The watchdog

The display shares the GPU with the system, so the launch watchdog of the driver applies. A
kernel that runs longer than the timeout of the watchdog is killed, and a killed kernel kills
the context of the run.

The code answers that with short kernels and a bounded batch. The plan of a tick admits 512
prompt tokens over every slot. A prompt longer than that is cut into pieces. A piece that does
not fit the budget of the tick waits for the next tick. No kernel of the tick graph therefore
grows with the length of a prompt.

The tick graph runs on the pump stream. The raster graph runs on a stream of the highest
priority, so the display does not wait for a tick. The pump answers the page requests of a tick
between two ticks, when no kernel of the tick graph runs. The pace holds a schedule and not a
delay. A tick that runs long therefore gives the schedule a new start, and does not push the
ticks that follow it.
