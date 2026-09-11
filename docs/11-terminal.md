# The terminal

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| replay, restore | recovery by re-application of the journal |
| feeder | the disk-side process that publishes host input to the inbound ring (an input publisher) |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| bus | an append-only message log between agents (a message bus) |
| arena | a contiguous memory region for offset-addressed allocations |

`aotx_tui` draws the six live panels from the display mirror in a terminal. It can attach to a
system that runs, or it can start a system from its System screen. It uses POSIX terminal calls
and ECMA-48 control sequences. It needs no terminal information library.

## Start it

Run the program from the build directory and give it the journal directory of the system:

```
build/aotx_tui --attach build/run
```

The feeder owns `<journal>/aotx.sock`. The terminal connects there, receives a read-only mirror
descriptor and begins with the newest complete frame. The status line shows `connecting` while
the attach is pending. When no system runs, the splash and the System screen remain available.

The complete command form is:

```
aotx_tui [--attach <journal>]... [--journal <dir>] [--settings <file>] [--no-splash]
```

| option | meaning |
| --- | --- |
| `--attach <journal>` | attach through this journal directory; repeat for several systems |
| `--journal <dir>` | use this journal directory when the System screen starts a system |
| `--settings <file>` | read, show and pass this settings file |
| `--no-splash` | open with no splash art |

With no journal option, `journal.dir` supplies the directory. A relative `journal.dir` starts at
the directory that contains the settings file. The terminal retries an attach every 200 ms. While
it is detached, it reads `<journal>/phase` and shows model placement, journal replay or running
state with the elapsed seconds. A socket-close message remains until the next key or a successful
attach.

The supported floor is 80 columns by 24 rows. A larger terminal shows more of the 160 by 50 cell
picture. A smaller terminal is not supported. A 160 by 52 terminal shows the full picture, its
status line and its key bar.

## The live picture

The work area is the same cell grid as the window: Console, Agents, Bus, Arena, Tick and Seam.
When the Console has focus, the viewport follows the editor cursor. Alt with an arrow pans the
picture by hand and suspends that follow. Ctrl-L, or the next line sent with Enter, resumes it.
The Panels rows of the Menu move directly to Console, Agents, Bus, Models, Tools or Settings.
They send no command line.

Console keys go to the feeder as the same key frame that the window uses. The device edits the
line, and the next mirror frame returns the text and cursor. Tab moves focus between Console and
Agents. With Agents focused, `y` authorizes the first pending request and `n` refuses it.
The feeder processes `import <path>`, `model fetch <name>` and [image commands](29-image-input.md) when Enter completes the line.
Both attached input and standard input use the same operation check.

## Screens

F1 through F11 open the principal screens. The same function key closes its open screen. Escape
closes any screen. Arrow keys move one row, Page Up and Page Down move one page, and Home and End
move to the bounds. Enter activates the selected row or begins an edit where the row accepts text.

| key | screen | use |
| --- | --- | --- |
| F1 | Help | show the command lines and send `help` |
| F2 | Menu | open another screen or move to a panel |
| F3 | Agents | list agents and answer requests |
| F4 | Bus | show all messages or one message kind |
| F5 | Models | list the model store |
| F6 | Tools | list tool modules, import, remove or open the picker |
| F7 | Skills | list skill modules, import, remove or open the picker |
| F8 | Settings | show or edit every setting |
| F9 | System | start, restore or stop a system and read its boot output |
| F10 | Quit | confirm that the terminal should close |
| F11 | Session | select an agent, read its transcript and send a multi-line prompt or task |

The Bus screen's `all` row sends `bus`. Its other rows send `bus` with one of `finding`, `rank`,
`question`, `answer`, `handoff`, `cost` or `note`.

On Tools and Skills, Enter imports the selected path, `m` sends `module` for the selected name,
`x` removes it and `p` opens the path picker. The picker also opens from a path row. The picker
lists directories first, refuses a path outside its root and imports the selected path with
Enter.

On Models, select a row and press Enter. The terminal fetches a missing file, activates a verified
file or loads an active file into an attached system. The row state and
the manifest decide the action.

The Settings screen writes the file when no system runs. With a system attached it sends `set`
and keeps the file for the next start. The range and the time of effect appear beside each key.

The Session screen reads the derived transcript of the selected agent. Up and Down select the
agent. Left and Right select an attached system. Enter adds a line in the editor. Ctrl-Enter
sends the text to the conductor, and Alt-Enter sends a task to the selected agent.

The keys `y` and `n` answer a pending request. The key `p` changes the agent page limit,
`c` starts compaction and `s` makes an agent of a named role. Page Up and Page Down move through
the transcript. Enter expands a long tool result when the editor is empty.

## Start a system

Start `aotx_tui` with `--journal` and, when needed, `--settings`. Press F9. The System screen
shows the build, settings file, model directory, roles, journal directory, window choice and the
last lines of `boot.log`. Select Start and press Enter. Select Restore, or press `r`, to replay
the newest complete journal before the system accepts new input.

The terminal starts the sibling `aotx_boot` program with `--tui-attached`. Boot output stays in
`<journal>/boot.log`. A feeder that cannot listen is a boot failure; its reason appears on the
System screen and the boot returns a nonzero status. After the socket becomes ready, the terminal
attaches without another command.

Select Stop, or press `x`, to send `quit` to an attached system. Closing `aotx_tui` does not stop
a system by itself.

## Terminal settings

These keys are read from the file named by `--settings`:

| key | default | accepted values |
| --- | --- | --- |
| `tui.on` | 0 | 0 or 1; start the terminal beside the boot |
| `tui.color` | `none` | `none` or `16` |
| `tui.box` | `ascii` | `ascii` or `utf8` |
| `tui.splash` | `auto` | `auto`, `braille`, `ascii` or `off` |
| `tui.escape_ms` | 25 | 5 to 500 ms |
| `mirror.hz` | 30 | 1 to 120 snapshots in one second |

The short Escape interval lets the decoder distinguish a lone Escape from the first byte of an Alt
key or a control sequence. A slow link may need a larger `tui.escape_ms`. `mirror.hz` governs the
publisher and applies at the next frame. The terminal draws at its own bounded rate and
uses the newest complete mirror snapshot.
