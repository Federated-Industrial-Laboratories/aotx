# The control program

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| instance | one system with its own journal directory, started or attached by the control program |
| replica | the disk files of a journal directory, as the control program reads them |

`aotx_ctrl` is the graphical control program. It starts, attaches to and stops systems, and it
shows their state in windows. It reads the replica of each journal directory, and it sends the
console commands over the attach socket. It uses GLFW, OpenGL and the vendored ImGui sources.

## Start it

Run the program from the build directory:

```
build/aotx_ctrl [--journal <dir>] [--settings <file>] [--sim] [--frames <n>] [--help]
```

| option | effect |
| --- | --- |
| `--journal <dir>` | the journal directory of the system to control |
| `--settings <file>` | the settings file; its `journal.dir` key names the journal directory |
| `--sim` | run against simulated data, with no system |
| `--frames <n>` | stop after this count of frames; the smoke check uses it |
| `--help` | print the usage text and stop |

With no journal option, the settings file names the journal directory. Without one, the program
opens the last journal it bound, which it keeps in `$XDG_CONFIG_HOME/aotx/journal` or
`~/.config/aotx/journal`. With no such journal, it makes a new one in `$XDG_DATA_HOME/aotx` or
`~/.local/share/aotx`. The window layout is kept in `ctrl-layout.ini` beside the journal name.

## The windows

The Windows menu opens and closes each window. The View menu rebuilds the layout.

| window | content |
| --- | --- |
| Instances | the known systems, their phase words and the Start, Attach, Stop and Remove controls |
| Control | the tick figures, the requests of the agents and the Start, Stop, Grant and Refuse controls |
| Models | the catalog with Fetch and Use, the role assignments, the model controls, the presets and the conduct |
| Modules | the imported skills, roles and tools, and a directory import |
| Sync | the module and voice profile files whose disk copies changed, and a Sync act for each |
| Settings | the settings file, with Save for a file key and Apply for a device key of a running system |
| Monitor | the tick, ring, memory, page map and agent figures of the selected system |
| Transcripts | the stored runs and the transcript of each conversation |
| Voice | the speech engine controls and the voice of each agent |
| First run | the six pages Detect, Build, Model, Activate, Start and First say |
| conversation | one chat with one agent; New conversation in the Windows menu opens one |

A conversation sends `say` lines to the console. Its Continue control resumes a reply that
ended at its reply limit. The conversation of agent 0 sends `continue`. A worker conversation
sends `agent <id> continue`. The New control of a conversation starts a worker conversation.

## The instance phases

The phase word of an instance comes from the `phase` file of its journal. A phase word `placing`
or `replaying` is live for 30 seconds after its file was written. A phase word `running` is live
while the attach socket answers or while the program owns the child. A `running` word with no
socket and no child is stale: the instance shows as stopped, and a Start is permitted.

## The attach

The feeder owns `<journal>/aotx.sock`. The program connects there, receives a read-only mirror
descriptor and reads the newest complete frame. A lost connection is stated once, and a retry
starts every 2 seconds. A refusal of the attach, for example when the system runs for another
user, is stated once as an error. The same refusal from a later retry is not stated again.

## The speech engine

The Voice window speaks the replies and the lifecycle lines. The engine needs the programs
`piper` and `pw-play` on the search path, and at least one `.onnx` voice file in
`~/.local/share/piper-voices`. Without one of them the window states the refusal and the
controls are off. A close of the program drops the queued lines and ends the line in synthesis.

## The build

The option `AOTX_CTRL` builds the program; it is ON. The build needs pkg-config and GLFW 3, as
`docs/06-build.md` lists. The check `ctrl_fix` runs the fix cases, and the check `ctrl_smoke`
runs the program with `--sim --frames 300`. The directory `ctrl/vendor/` holds the ImGui
sources, and the gates do not read it.
