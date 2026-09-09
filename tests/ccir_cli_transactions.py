#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check CLI section retention, tail IDs and conditional updates with real processes.
# Input: CLI, fixture and preload paths. Output: check counts. Exit: 0 success, 1 failure.
import hashlib
import os
import pathlib
import select
import struct
import subprocess
import sys
import tempfile

cli, fixture, preload = map(lambda item: str(pathlib.Path(item).resolve()), sys.argv[1:])
checks = 0


def check(value):
    global checks
    checks += 1
    if not value:
        raise AssertionError(f"check {checks} failed")


def run(program, *args, code=0):
    result = subprocess.run([program, *map(str, args)], capture_output=True,
                            text=True, timeout=20)
    check(result.returncode == code)
    return result.stdout


def sections(path):
    raw = path.read_bytes()
    roots = [raw[4096:8192], raw[8192:12288]]
    roots = [root for root in roots if root[:8] == b"AOTXROOT"]
    root = max(roots, key=lambda item: struct.unpack_from("<Q", item, 16)[0])
    check(hashlib.sha256(root[:4064]).digest() == root[4064:])
    generation, offset = struct.unpack_from("<QQ", root, 16)
    commit = raw[offset:offset + 256]
    check(hashlib.sha256(commit).digest() == root[48:80])
    directory, count = struct.unpack_from("<QI", commit, 72)
    rows = raw[directory:directory + count * 128]
    check(hashlib.sha256(rows).digest() == commit[96:128])
    found = {}
    for index in range(count):
        row = rows[index * 128:(index + 1) * 128]
        kind, schema, flags = struct.unpack_from("<IHH", row)
        identity = row[8:24]
        start, length = struct.unpack_from("<QQ", row, 24)
        payload = raw[start:start + length]
        check(hashlib.sha256(payload).digest() == row[56:88])
        check(identity not in found)
        found[identity] = (kind, schema, flags, payload)
    check(len(found) == count)
    return generation, found


def optional(rows):
    return {identity: row for identity, row in rows.items() if row[0] > 3}


def tail(rows):
    hits = [(identity, row) for identity, row in rows.items() if row[0] == 3]
    check(len(hits) == 1)
    return hits[0]


def layouts(root, count, checkpoint, recorded):
    for order in range(3):
        path = root / f"remove-{count}-{order}.aotxccir"
        compact = root / f"remove-small-{count}-{order}.aotxccir"
        run(fixture, "make", path, count, order, 1, order)
        before = sections(path)[1]
        check(len(optional(before)) == count)
        run(cli, "append", path, checkpoint, 2, 2, 101)
        generation, after = sections(path)
        check(generation == 2 and len(after) == count + 2)
        check(optional(after) == optional(before))
        check(all(row[0] != 3 for row in after.values()))
        run(cli, "verify", path)
        run(cli, "compact", path, compact)
        check(optional(sections(compact)[1]) == optional(before))
    for order in range(2):
        path = root / f"add-{count}-{order}.aotxccir"
        compact = root / f"add-small-{count}-{order}.aotxccir"
        run(fixture, "make", path, count, order, 0, order)
        before = sections(path)[1]
        check(bytes([3]) + bytes(15) in optional(before))
        check(len(optional(before)) == count)
        run(cli, "append", path, checkpoint, 2, 3, 101, recorded)
        generation, after = sections(path)
        identity, row = tail(after)
        check(identity not in before and row[3] == recorded.read_bytes())
        check(generation == 2 and len(after) == count + 3)
        check(optional(after) == optional(before))
        run(cli, "append", path, checkpoint, 3, 4, 102, recorded)
        check(tail(sections(path)[1])[0] == identity)
        run(cli, "verify", path)
        run(cli, "compact", path, compact)
        check(optional(sections(compact)[1]) == optional(before))


def interleaved(root, count, checkpoint):
    for serial in range(count):
        path = root / f"race-{count}-{serial}.aotxccir"
        compact = root / f"race-small-{count}-{serial}.aotxccir"
        run(fixture, "make", path, count, serial % 2, 0, serial)
        original = sections(path)[1]
        ready_read, ready_write = os.pipe()
        go_read, go_write = os.pipe()
        environment = os.environ.copy()
        environment.update(LD_PRELOAD=preload, AOTX_CCIR_PAUSE_PATH=str(path),
                           AOTX_CCIR_PAUSE_READY=str(ready_write),
                           AOTX_CCIR_PAUSE_GO=str(go_read))
        child = subprocess.Popen([cli, "append", str(path), str(checkpoint), "2", "2", "101"],
                                 env=environment, pass_fds=(ready_write, go_read),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.close(ready_write)
        os.close(go_read)
        try:
            check(bool(select.select([ready_read], [], [], 10)[0]))
            check(os.read(ready_read, 1) == b"r")
            run(fixture, "add", path, serial)
            acknowledged = path.read_bytes()
            generation, added = sections(path)
            check(generation == 2 and len(added) == count + 3)
            additions = {identity: row for identity, row in added.items() if row[0] == 9000}
            check(len(additions) == 1)
            check(all(added[identity] == value for identity, value in optional(original).items()))
            os.write(go_write, b"g")
            output, error = child.communicate(timeout=20)
            check(child.returncode == 7 and "file changed; read it again" in error)
            check("complete" not in output and path.read_bytes() == acknowledged)
        finally:
            os.close(ready_read)
            os.close(go_write)
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        run(cli, "append", path, checkpoint, 2, 2, 101)
        generation, retried = sections(path)
        check(generation == 3 and optional(retried) == optional(added))
        run(cli, "compact", path, compact)
        check(optional(sections(compact)[1]) == optional(added))
    print(f"ccir CLI interleaving N={count}: {count} changed-file refusals and retained additions")


with tempfile.TemporaryDirectory(prefix="aotx-ccir-transactions-") as directory:
    root = pathlib.Path(directory)
    checkpoint = root / "checkpoint.bin"
    recorded = root / "tail.bin"
    checkpoint.write_bytes(bytes((index * 37) % 251 for index in range(219)))
    recorded.write_bytes(bytes((index * 71) % 253 for index in range(137)))
    for count in [1, 64]:
        layouts(root, count, checkpoint, recorded)
        interleaved(root, count, checkpoint)

print(f"ccir CLI transactions: {checks} checks, 0 failed")
