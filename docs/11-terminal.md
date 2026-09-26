<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Terminal client

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

`aotx_tui` displays the GPU cell mirror and sends input through the feeder.
It can attach to running instances or start an instance from its System screen.
The client uses POSIX terminal calls and ECMA-48 control sequences without a terminal-information library.

<details>
<summary>On this page</summary>

- [Start it](#start-it)
- [The live picture](#the-live-picture)
- [Screens](#screens)
- [Start a system](#start-a-system)
- [Terminal settings](#terminal-settings)

</details>

## Start it

Run from the repository root after building:

```sh
build/aotx_tui --attach build/run
```

The complete command form is:

```text
aotx_tui [--attach JOURNAL]... [--journal DIR] [--settings FILE] [--no-splash]
```

| option | meaning |
| --- | --- |
| `--attach <journal>` | attach through this journal directory; repeat for several systems |
| `--journal <dir>` | use this journal directory when the System screen starts a system |
| `--settings <file>` | read, show and pass this settings file |
| `--no-splash` | open with no splash art |

The feeder owns `<journal>/aotx.sock` and supplies a read-only mirror descriptor.
The client starts from the newest complete snapshot and reports pending or failed attachment.
Without a running instance, the splash and System screen remain available.

Mirror layout 7 provides 256 agent rows and 256 module rows; unused rows remain empty.
Rebuild terminal, feeder and control monitor together when that layout changes.
An incompatible layout is refused.

The client retries attachment every 200 ms.
While detached, it reads the journal's `phase` file for placement, replay and running status.
A socket-close message remains until the next key or successful attachment.
A relative `journal.dir` resolves from the settings file's directory.

Use a terminal of at least 80 columns by 24 rows.
A 160-by-52 terminal shows the complete grid, status line and key bar.

## The live picture

The work area contains the same Console, Agents, Bus, Arena, Tick and Seam panels as the GPU window.
The viewport follows the console cursor.
Alt+Arrow pans manually; Ctrl+L or a submitted line restores cursor following.
Menu panel selections move the viewport without sending commands.

Tab moves focus between Console and Agents.
In Agents, `y` grants the first pending request and `n` refuses it.
The feeder recognizes import, model-fetch and media operations when a submitted line requires disk input.
[Image](29-image-input.md) and [audio](30-audio-input.md) guides define their file commands.

## Screens

Press a function key to open its screen; press it again or Escape to close.
Arrow keys move by row, Page Up and Page Down by page, and Home and End to the bounds.
Enter activates or edits the selected row.

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

### Models, modules and settings

The Models screen chooses fetch, activation or resident loading from the selected row's actual file and manifest state.
These are separate operations; inspect the row before activation.

In Tools and Skills, Enter imports a path, `m` inspects the named module and `x` removes it.
Press `p` for the path picker. It lists directories first and refuses paths outside its root.

Settings writes the file when detached.
When attached, it sends device changes with `set` and retains file values for the next startup.
Each row shows its allowed range and application boundary.

### Session input

The Session screen reads the selected agent's derived transcript.
Up and Down select an agent; Left and Right select an attached instance.
Enter adds a line in the editor. Ctrl+Enter sends a conductor message; Alt+Enter submits a task to the selected agent.

Use `y` or `n` to answer a pending request, `p` to change pages and `c` to request compaction.
Use `s` to create an agent with a named role.
Page Up and Page Down scroll the transcript.
With an empty editor, Enter expands a long tool result.

## Start a system

1. Start the terminal with the intended `--journal` and `--settings` values.
2. Press F9 to inspect build, model, role, journal and window settings.
3. Select Start for a new boot, or Restore for the saved journal.
4. Read `boot.log` in the System screen until startup completes or reports a failure.

The terminal starts its sibling `aotx_boot` with `--tui-attached` and attaches when the socket is ready.
A feeder that cannot listen causes boot failure.
To stop an attached instance, select Stop or press `x` on the System screen.
Closing the terminal alone does not stop that instance.

## Terminal settings

| key | default | accepted values |
| --- | --- | --- |
| `tui.on` | 0 | 0 or 1; start the terminal beside the boot |
| `tui.color` | `none` | `none` or `16` |
| `tui.box` | `ascii` | `ascii` or `utf8` |
| `tui.splash` | `auto` | `auto`, `braille`, `ascii` or `off` |
| `tui.escape_ms` | 25 | 5 to 500 ms |
| `mirror.hz` | 30 | 1 to 120 snapshots in one second |

A longer Escape interval can help distinguish complete control sequences on a slow connection.
`mirror.hz` controls device snapshot publication at the next frame.
The terminal renders at its own bounded rate and reads the newest complete snapshot.
See the [settings reference](settings.md) for startup and device values.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
