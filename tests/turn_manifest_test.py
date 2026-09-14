#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check turn equality, agent order and raw manifest chain fault detection.
# Owns: Distinct rows and changed byte strings at batch sizes one and 64.
# Threading: One CPU test process; no runtime or model is started.
# Lifetime: The process tests valid chains and independent field and order changes.

# Inputs: None. Output: Check and failure counts. Exit: 0 pass, 1 failed check.
import hashlib
import json
import sys

sys.dont_write_bytecode = True
from turn_manifest import FIELDS, manifest, same_turns

checks = failures = 0


def check(value, label):
    global checks, failures
    checks += 1
    if not value:
        failures += 1
        print("failed: " + label, file=sys.stderr)


def chain(rows):
    previous, lines = "0" * 64, []
    for row in rows:
        line = json.dumps(dict(row, prev=previous), separators=(",", ":")).encode() + b"\n"
        lines.append(line)
        previous = hashlib.sha256(line).hexdigest()
    return b"".join(lines)


def rows(count):
    return [dict(agent=agent, turn=turn, input_hash=f"{((agent + 1) << 32) | turn:016x}",
                 output_hash=f"{((turn + 7) << 40) | (agent + 19):016x}", tokens=agent + turn + 3,
                 finish="stop", tool="none", request=agent * 17 + turn)
            for turn in range(1, 4) for agent in range(count)]


def refused(original, changed, label):
    check(not same_turns(original, changed), label)
    check(not same_turns(changed, original), "original " + label)


def invalid(original, changed, label):
    refused(original, changed, label)
    check(not same_turns(changed, changed), "invalid identical inputs: " + label)


def batch(count):
    before = checks
    values = rows(count)
    original = chain(values)
    check(same_turns(original, original), "complete identical manifest")
    alternate = [row for turn in range(1, 4) for row in reversed(values) if row["turn"] == turn]
    check(same_turns(original, chain(alternate)), "independent agent order")
    check((original != chain(alternate)) == (count > 1), "agent order control changes raw bytes")
    check(len(manifest(original)) == count, "all distinct agents remain present")
    for index, row in enumerate(values):
        for field in FIELDS:
            changed = [dict(item) for item in values]
            value = row[field]
            if field in ("input_hash", "output_hash"):
                value = ("1" if value[0] == "0" else "0") + value[1:]
            elif field == "finish":
                value = "limit"
            elif field == "tool":
                value = "fs_read"
            else:
                value += count + 7
            changed[index][field] = value
            refused(original, chain(changed), "changed " + field)
        refused(original, chain(values[:index] + values[index + 1:]), "missing turn")
        invalid(original, chain(values[:index] + [row] + values[index:]), "duplicate turn")
    for agent in range(count):
        changed = list(values)
        changed[agent], changed[count + agent] = changed[count + agent], changed[agent]
        refused(original, chain(changed), "changed order within an agent")
    for index in (0, len(values) - 1):
        for field in FIELDS:
            changed = [dict(item) for item in values]
            del changed[index][field]
            invalid(original, chain(changed), "missing " + field)
        changed = [dict(item) for item in values]
        changed[index]["extra"] = 1
        invalid(original, chain(changed), "extra field")
        for field in ("agent", "turn", "tokens", "request"):
            changed = [dict(item) for item in values]
            changed[index][field] = True
            invalid(original, chain(changed), "boolean " + field)
        for value in (-1, 0x100000000, 1.5, "1", None):
            changed = [dict(item) for item in values]
            changed[index]["agent"] = value
            invalid(original, chain(changed), "invalid integer field")
        changed = [dict(item) for item in values]
        changed[index]["input_hash"] = "z" * 16
        invalid(original, chain(changed), "invalid content hash")
        changed[index] = dict(values[index], finish=1)
        invalid(original, chain(changed), "invalid text field")
    invalid(original, b" " + original, "raw line prefix changes the next digest")
    invalid(original, original.replace(b"\n", b"\r\n", 1), "raw newline changes the next digest")
    invalid(original, original[:-1], "missing final newline")
    invalid(original, original + b"\n", "blank row")
    invalid(original, original.replace(b'"prev":"0', b'"prev":"1', 1), "wrong first chain value")
    lines = original.splitlines(keepends=True)
    middle = json.loads(lines[1]); middle["prev"] = "f" * 64
    invalid(original, lines[0] + json.dumps(middle).encode() + b"\n" + b"".join(lines[2:]), "broken later chain")
    last = json.loads(lines[-1])
    for field in FIELDS + ("prev",):
        duplicate = json.dumps(last[field]).encode()
        prefix = b'{"' + field.encode() + b'":' + duplicate + b','
        invalid(original, b"".join(lines[:-1]) + prefix + lines[-1][1:], "duplicate JSON key " + field)
    del last["prev"]
    invalid(original, b"".join(lines[:-1]) + json.dumps(last).encode() + b"\n", "missing previous digest")
    for raw in (b"{\n", b"[]\n", b"null\n", b"1\n", b"\xff\n", b"{}\n"):
        invalid(original, raw, "malformed row")
    print(f"turn manifest N={count}: {checks - before} checks")


def main():
    check(same_turns(b"", b""), "two empty manifests")
    check(not same_turns("", b""), "manifest inputs are bytes")
    for count in (1, 64):
        batch(count)
    print(f"turn manifest: {checks} checks, {failures} failures")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
