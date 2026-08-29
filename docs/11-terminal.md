# The terminal

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
it waits. When no system runs, the splash and the System screen remain available.

The complete command form is:

```
aotx_tui [--attach <journal>] [--journal <dir>] [--settings <file>] [--no-splash]
```

| option | meaning |
| --- | --- |
| `--attach <journal>` | attach to a system through this journal directory |
| `--journal <dir>` | use this journal directory when the System screen starts a system |
| `--settings <file>` | read, show and pass this settings file |
| `--no-splash` | open with no splash art |

With no journal option, `journal.dir` supplies the directory. A relative `journal.dir` starts at
the directory that holds the settings file. The terminal retries an attach every 200 ms. While
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
Agents. With Agents focused, `y` authorizes the first request that waits and `n` refuses it.

## Screens

F1 through F10 open the principal screens. The same function key closes its open screen. Escape
closes any screen. Arrow keys move one row, Page Up and Page Down move one page, and Home and End
move to the bounds. Enter takes the selected row or begins an edit where the row accepts text.

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

The Bus screen's `all` row sends `bus`. Its other rows send `bus` with one of `finding`, `rank`,
`question`, `answer`, `handoff`, `cost` or `note`.

On Tools and Skills, Enter imports the selected path, `m` sends `module` for the selected name,
`x` removes it and `p` opens the path picker. The picker also opens from a path row. The picker
lists directories first, refuses a path outside its root and imports the selected path with
Enter.

The Settings screen writes the file when no system runs. With a system attached it sends `set`
and keeps the file for the next start. The range and the time of effect appear beside each key.

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

The short Escape wait lets the decoder distinguish a lone Escape from the first byte of an Alt
key or a control sequence. A slow link may need a larger `tui.escape_ms`. `mirror.hz` governs the
publisher and takes effect at the next frame. The terminal draws at its own bounded rate and
uses the newest complete mirror snapshot.
