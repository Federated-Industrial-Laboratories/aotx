#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# parity-gate.py: the surface parity gate.
#
# The gate holds the terminal program and the command parser together. It reads the action
# table of the terminal program and the help table of the parser, and it refuses a
# difference between them. Every action of a screen is one command line, so a screen can
# never send a line the device does not parse.
#
# Three arms:
#   1. every command line an action row sends names a command the help table has;
#   2. every command the help table names is a command the parser dispatches;
#   3. every key the key bar names is a key that one screen of the screen table takes,
#      and every screen an action row names is a row of the screen table.
#
#   parity-gate.py                     examine the tables of the repository.
#   parity-gate.py --actions PATH      the action table to read.
#   parity-gate.py --help-table PATH   the help table of the parser to read.
#   parity-gate.py --parser PATH       the parser to read the dispatch from.
# Output: one line for each finding, then a summary line with the counts.
# Exit codes: 0 clean, 1 findings, 2 usage or environment error.

import re
import sys
from pathlib import Path

ACTIONS = "disk/tui/actions.h"
HELP_TABLE = "cuda/cli/help.cuh"
PARSER = "cuda/cli/parse.cu"

# A table of rows of text fields, ended by a row that holds no text.
TABLE = r"{name}\s*\[\s*\]\s*=\s*\{{(.*?)\n\s*\}};"
ROW2 = re.compile(r'\{\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\}')
ROW3 = re.compile(r'\{\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\}')
ROW1 = re.compile(r'\{\s*"([^"]*)"\s*\}')
# One line of the help table. A command line has two spaces before the name; a line of
# six spaces carries a note about the line above it.
HELP_LINE = re.compile(r'return\s+"  (\S+)')
DISPATCH = re.compile(r'aotx_cli_is\(\s*first\s*,\s*"([^"]+)"\s*\)')


def read(path, what):
    if not path.is_file():
        print(f"parity-gate: no {what} at {path}", file=sys.stderr)
        sys.exit(2)
    return path.read_text(encoding="utf-8")


def table_of(text, name, row):
    found = re.search(TABLE.format(name=name), text, re.DOTALL)
    if found is None:
        return None
    return row.findall(found.group(1))


def command_of(line):
    # The words before the first placeholder are the command. A placeholder starts with
    # the character <, which no command word holds.
    words = []
    for word in line.split():
        if word.startswith("<") or word.startswith("["):
            break
        words.append(word)
    return words[0] if words else ""


def main(argv):
    root = Path(__file__).resolve().parent.parent
    actions_path = root / ACTIONS
    help_path = root / HELP_TABLE
    parser_path = root / PARSER
    at = 1
    while at < len(argv):
        if argv[at] == "--actions" and at + 1 < len(argv):
            actions_path = Path(argv[at + 1])
        elif argv[at] == "--help-table" and at + 1 < len(argv):
            help_path = Path(argv[at + 1])
        elif argv[at] == "--parser" and at + 1 < len(argv):
            parser_path = Path(argv[at + 1])
        else:
            print(f"parity-gate: the option {argv[at]} is not known", file=sys.stderr)
            return 2
        at += 2

    help_text = read(help_path, "help table")
    parser_text = read(parser_path, "parser")
    commands = set(HELP_LINE.findall(help_text))
    dispatched = set(DISPATCH.findall(parser_text))
    if not commands:
        print(f"parity-gate: the help table at {help_path} names no command",
              file=sys.stderr)
        return 2

    findings = []
    for name in sorted(commands - dispatched):
        findings.append(f"the help table names {name}, which the parser does not dispatch")

    if not actions_path.is_file():
        # The terminal program is not in this build. The gate states the skip, because a
        # gate that says nothing is not evidence.
        print(f"parity-gate: no action table at {actions_path}; the terminal is not built")
        print(f"parity-gate: {len(commands)} commands, {len(findings)} findings")
        return 1 if findings else 0

    actions_text = read(actions_path, "action table")
    screens = table_of(actions_text, "aotx_tui_screens", ROW2)
    keys = table_of(actions_text, "aotx_tui_keys", ROW2)
    menu = table_of(actions_text, "aotx_tui_menu", ROW3)
    bus_kinds = table_of(actions_text, "aotx_tui_bus_kinds", ROW1)
    actions = table_of(actions_text, "aotx_tui_actions", ROW3)
    for name, rows in (("aotx_tui_screens", screens), ("aotx_tui_keys", keys),
                       ("aotx_tui_menu", menu), ("aotx_tui_bus_kinds", bus_kinds),
                       ("aotx_tui_actions", actions)):
        if rows is None:
            print(f"parity-gate: {actions_path} holds no table {name}", file=sys.stderr)
            return 2

    screen_names = {row[0] for row in screens}
    screen_keys = {row[1] for row in screens if row[1] != ""}
    for key, label in keys:
        if key not in screen_keys:
            findings.append(f"the key bar names {key} ({label}), which no screen takes")
    for label, screen, panel in menu:
        if screen != "" and screen not in screen_names:
            findings.append(f"the Menu row {label} names {screen}, which is not a screen")
        if (screen == "") == (panel == ""):
            findings.append(f"the Menu row {label} must name one screen or one panel")
    kind_start = parser_text.find("unsigned int aotx_cli_kind(")
    kind_end = parser_text.find("__device__ const char *aotx_cli_kind_name", kind_start)
    if kind_start < 0 or kind_end < 0:
        print(f"parity-gate: {parser_path} holds no bus kind parser", file=sys.stderr)
        return 2
    parser_kinds = set(re.findall(r'aotx_cli_is\(word,\s*"([^"]+)"\)',
                                  parser_text[kind_start:kind_end]))
    table_kinds = set(bus_kinds)
    for word in sorted(table_kinds - parser_kinds):
        findings.append(f"the Bus screen names {word}, which the parser does not take")
    for word in sorted(parser_kinds - table_kinds):
        findings.append(f"the parser takes the bus kind {word}, which the Bus screen omits")
    sends = 0
    for screen, key, line in actions:
        if screen not in screen_names:
            findings.append(f"the action of {screen} names a screen that no row has")
        if line == "":
            # An action that changes the disk and sends nothing. It reaches no parser.
            continue
        sends += 1
        command = command_of(line)
        if command not in commands:
            findings.append(f"the screen {screen} sends {command}, which the parser"
                            " does not name")

    for line in findings:
        print(f"parity-gate: {line}")
    print(f"parity-gate: {len(commands)} commands, {len(screens)} screens, "
          f"{len(keys)} keys, {len(menu)} Menu rows, {len(bus_kinds)} bus kinds, "
          f"{len(actions)} actions, {sends} of them send a line, "
          f"{len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
