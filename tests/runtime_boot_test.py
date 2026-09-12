#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check complete file activation, continued input and recovery without source directories.
# Owns: New output files and the boot processes started here.
# Threading: One disk driver; CUDA processes each input batch.
# Lifetime: Four runtime processes and one copied file per case.

# Inputs: build, source, store, new output, and vector or text batch size. Output: logs and checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, wait
from text_boot_test import setup
from checkpoint_boot_test import spawn
from capacity_boot_test import batch, color
from context_boot_bytes import corpus, request, selected


class RuntimeTest(Test):
    def command(self, argv, label):
        self.commands += 1
        path = self.output / f"{self.commands:03d}-{label}.log"
        argv = list(map(str, argv)); begin = time.monotonic()
        self.record(command=argv, output=str(path))
        with path.open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        self.record(exit=result.returncode, seconds=time.monotonic() - begin, output=str(path))
        self.check(result.returncode == 0, f"{label} command exit")
        return path.read_text()


class RuntimeRun(Run):
    def __init__(self, test, label, ccir, extra=()):
        self.test, self.boot = test, None
        self.journal = test.output / f"{label}-journal"
        self.path = test.output / f"{label}-boot.log"
        self.log = self.path.open("w")
        argv = [test.build / "aotx_runtime_offline", "stdbuf", "-oL", "-eL", test.build / "aotx_boot", "--ccir", ccir,
                "--journal", self.journal, "--ticks", "0"]
        argv.extend(extra)
        self.started = time.time_ns()
        test.record(command=list(map(str, argv)), output=str(self.path))
        self.child = subprocess.Popen(list(map(str, argv)), cwd=test.source, stdin=subprocess.PIPE,
                                      stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.children = {}; test.active.append(self); test.record(pid=self.child.pid)


def durable(run):
    pattern = (r"memory mirror: committed (\d+) durable (\d+) generation (\d+) pending (\d+) error (\d+)"
               r" runtime source (\d+) durable (\d+)")
    target = None
    def received():
        nonlocal target
        prior = len(re.findall(pattern, run.console())); run.send("memory")
        def next_row():
            found = re.findall(pattern, run.console())
            return tuple(map(int, found[-1])) if len(found) > prior else None
        row = wait(next_row, run.child, 60)
        if target is None:
            target = row[5]
        run.test.check(not row[4], "complete runtime mirror has no disk error", counters=row)
        return row if row[0] == row[1] and row[2] and row[6] >= target else None
    row = wait(received, run.child, 300)
    run.test.check(True, "memory and the observed runtime source are durable", counters=row, target=target)
    return row


def sections(test, path):
    def reader():
        stream = path.open("rb")
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            held, current = os.fstat(stream.fileno()), path.stat()
            if (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino):
                return stream
        except BlockingIOError:
            pass
        except BaseException:
            stream.close()
            raise
        stream.close()
        return None
    result = {}
    with wait(reader, seconds=300) as stream:
        text = test.command([test.build / "aotx_ccir", "inspect", path], "inspect-runtime")
        for line in text.splitlines():
            match = re.search(r"type (\d+) schema (\d+) required (\d+).* offset (\d+) bytes (\d+)", line)
            if not match:
                continue
            kind, schema, required, offset, size = map(int, match.groups())
            test.check(required == 1, "runtime dependency is required")
            if kind != 6:
                stream.seek(offset); result[kind] = stream.read(size)
    test.check(set(result) == {1, 2, 4, 5, 7}, "complete runtime state section families")
    return result


def replay_records(f, data, kind):
    result, cursor = [], 128
    for _ in range(f.get(data, 48)):
        size = f.get(data, cursor); cursor += 8
        block = data[cursor:cursor + size]; cursor += size
        for j in range(f.get(block, 40, 4)):
            record = block[64 + j * 256:64 + (j + 1) * 256]
            if record[44] == 1 and record[45] == kind:
                result.append(record[64:64 + f.get(record, 48, 4)])
    if cursor != len(data):
        raise AssertionError("replay frames do not cover the extent")
    return result


def inspect_identity(run, marker):
    before = len(run.console()); run.send("module conductor")
    wait(lambda: marker in run.console()[before:], run.child)
    run.test.check(True, "device catalog shows the changed identity manifest")


def bindings(test, f, live, count, ordinal):
    stride = f.get(live, 12, 4)
    test.check(stride == 17416 and f.get(live, 16, 4) == count, "file has the complete binding batch")
    for i in range(count):
        row = live[128 + i * stride:128 + (i + 1) * stride]
        test.check(f.get(row, 0, 4) == i and f.get(row, 16) == ordinal and
                   row[24:40] == f.identity(10000 + i) and row[56:72] == f.identity(8000 + i),
                   "file preserves each private principal, conversation and ordinal")
        test.check(f.get(row, 72, 4) == ordinal and f.get(row, 76, 4) == 1,
                   "file preserves automatic retention and distinct working focus")


def reply(run, f, count, turn, text):
    for i in range(count):
        run.reply(i, turn, color(i))
        audit = [r for r in run.events(i) if r.get("kind") == "selection" and r.get("turn") == turn]
        expected = selected(f, i, 1, text)
        run.test.check(len(audit) == 1 and audit[0]["text"].endswith(
            " objects " + ",".join(k.hex() + "@1" for k in expected)),
            "new input recalls the exact private requirement and its source-paired state")


def exercise(test, count, text):
    f = setup(test); values = corpus(f, count, text); cut = len(values)
    operation, op = ("text", 6) if text else ("query", 4)
    inputs = test.output / "inputs"
    checkpoint, memory = inputs / "checkpoint", inputs / "memory.aotxccir"
    checkpoint.write_bytes(f.image(values, cut, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", memory, 1], "prepared-memory")
    store = test.output / "selected-models"; store.mkdir()
    entries = [json.loads(line) for line in (test.store / "manifest.jsonl").read_text().splitlines()]
    for entry in entries:
        local = store / Path(entry["path"]).name
        local.symlink_to((test.store / entry["path"]).resolve()); entry["path"] = local.name
    (store / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    runtime = test.output / "state.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", memory, "--models", store,
                  "--modules", test.output / "modules", "--settings", test.output / "settings",
                  "--roles", "language,embedding" if text else "language", "--output", runtime,
                  "--phrases", test.source / "tests/fixtures/quality/refusal-phrases.txt"], "pack-runtime")
    initial = sections(test, runtime)
    affect = bool(f.get(initial[5], 20, 4) & 1)
    test.check(f.get(initial[1], 8, 4) == 3 and f.get(initial[7], 12, 4) == 1,
               "new runtime has the creation profile")
    shutil.rmtree(store); shutil.rmtree(test.output / "modules")
    (test.output / "settings").unlink(); shutil.rmtree(inputs)
    test.check(not store.exists() and not memory.exists(), "selected source files are removed")
    run = RuntimeRun(test, "before", runtime); run.ready(); spawn(run, count)
    test.check("runtime: recovered state is durable; input is ready" in run.path.read_text(),
               "initial state reaches the complete mirror before new input")
    test.check("network: IPv4 and IPv6 sockets are disabled" in run.path.read_text(),
               "initial activation runs with IP sockets disabled")
    if affect:
        run.send("set affect.on 1"); durable(run)
    bind = test.output / "bind"; data = batch(f, b"AOTXBND1", 64, cut, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 160, 4); f.put(data, at + 60, 1, 4)
    bind.write_bytes(data); run.operation("bind", bind, 3, count); durable(run)
    path = test.output / "input"; path.write_bytes(request(f, count, cut, 1, text))
    run.operation(operation, path, op, count); cut += 3 * count
    reply(run, f, count, 1, text)
    durable(run)
    changed = test.output / "changed-identity" / "conductor"
    shutil.copytree(test.source / "modules/roles/conductor", changed)
    marker = "Keeps each member requirement with its source."
    manifest = changed / "module.manifest"
    identity_text = re.sub(r"(?m)^description:.*$", "description: " + marker, manifest.read_text())
    manifest.write_text(identity_text.replace("version: 1", "version: 2"))
    with (changed / "overlay.txt").open("a") as stream:
        stream.write("\nKeep each member requirement with its source.\n")
    run.send(f"import {changed}"); inspect_identity(run, marker)
    run.send("set sample.seed 19"); durable(run)
    run.stop(killed=True)
    before = sections(test, runtime)
    bindings(test, f, before[4], count, 1)
    test.check(f.get(before[7], 12, 4) == 2 and f.get(before[4], 16, 4) == count,
               "published runtime contains the complete live binding batch and recovery log")
    old_affect = replay_records(f, before[7], 31)
    if affect:
        agents = {f.get(body, 0, 4) for body in old_affect if any(body[8:24])}
        test.check(agents == set(range(count)), "every agent has nonzero affect state in the complete file")
    else:
        test.check(not old_affect, "the build without affect has no affect state records")
    copied = test.output / "copy.aotxccir"
    test.command([test.build / "aotx_ccir", "compact", runtime, copied], "copy-runtime")
    runtime.unlink(); shutil.rmtree(run.journal); bind.unlink(); path.unlink(); shutil.rmtree(changed.parent)
    after = RuntimeRun(test, "after", copied); after.ready()
    test.check("runtime: recovered state is durable; input is ready" in after.path.read_text(),
               "copied file restores and reaches its own complete mirror")
    test.check("network: IPv4 and IPv6 sockets are disabled" in after.path.read_text(),
               "copied file recovery runs with IP sockets disabled")
    match = re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)", after.path.read_text())
    test.check(match and int(match[1]) == f.get(before[7], 40) and int(match[2], 16) == f.get(before[7], 32)
               and not int(match[3]) and not int(match[4]), "exact runtime record count and state hash are restored")
    inspect_identity(after, marker)
    durable(after)
    restored = sections(test, copied)
    bindings(test, f, restored[4], count, 1)
    test.check(restored[2] == before[2], "copied runtime restores exact typed memory")
    test.check(replay_records(f, restored[7], 31) == old_affect, "complete affect state records survive file-only recovery")
    test.check(replay_records(f, restored[7], 21) == replay_records(f, before[7], 21),
               "device settings survive file-only recovery byte for byte")
    data = request(f, count, cut, 1, text)
    for i in range(count):
        at = 64 + i * 8256; f.put(data, at + 32, 2)
        data[at + 64:at + 80] = f.identity(100000 + 2 * 64 + i)
        data[at + 112:at + 128] = f.identity(200000 + 2 * 64 + i)
    path.write_bytes(data)
    after.operation(operation, path, op, count)
    reply(after, f, count, 2, text)
    durable(after); after.stop()
    final = sections(test, copied); bindings(test, f, final[4], count, 2)
    test.check(f.get(final[2], 32) == cut + 3 * count, "continued retention advances the exact object sequence")
    test.check(not store.exists() and not runtime.exists(), "continued runtime needs only the copied file")
    previous = after
    for ordinal in (3, 4):
        saved = final
        shutil.rmtree(previous.journal)
        resumed = RuntimeRun(test, f"repeat-{ordinal}", copied); resumed.ready()
        match = re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)",
                          resumed.path.read_text())
        test.check(match and int(match[1]) == f.get(saved[7], 40) and int(match[2], 16) == f.get(saved[7], 32)
                   and not int(match[3]) and not int(match[4]),
                   "a file saved after recovery restores its exact count and hash again")
        inspect_identity(resumed, marker); durable(resumed)
        recovered = sections(test, copied)
        bindings(test, f, recovered[4], count, ordinal - 1)
        test.check(recovered[2] == saved[2], "repeated recovery preserves every typed memory byte")
        for kind in (21, 31):
            test.check(replay_records(f, recovered[7], kind) == replay_records(f, saved[7], kind),
                       "repeated recovery preserves recorded settings and affect")
        data = request(f, count, f.get(saved[2], 32), 1, text)
        for i in range(count):
            at = 64 + i * 8256; f.put(data, at + 32, ordinal)
            data[at + 64:at + 80] = f.identity(100000 + ordinal * 64 + i)
            data[at + 112:at + 128] = f.identity(200000 + ordinal * 64 + i)
        path.write_bytes(data)
        resumed.operation(operation, path, op, count); reply(resumed, f, count, ordinal, text)
        durable(resumed); resumed.stop()
        final = sections(test, copied); bindings(test, f, final[4], count, ordinal)
        test.check(f.get(final[2], 32) == cut + 3 * (ordinal - 1) * count,
                   "fresh input after repeated recovery advances the exact memory sequence")
        previous = resumed


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("1", "64", "text-1", "text-64"):
        print("usage: runtime_boot_test.py BUILD SOURCE STORE OUTPUT 1|64|text-1|text-64", file=sys.stderr); return 2
    build, source, store, output = (Path(v).resolve() for v in sys.argv[1:5])
    test = RuntimeTest(build, source, store, output, snapshot_every=64)
    begin, status = time.monotonic(), 0
    try:
        exercise(test, int(sys.argv[5].split("-")[-1]), sys.argv[5].startswith("text-"))
    except Exception as error:
        status = 1; (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"runtime boot failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    status |= any(not check["passed"] for check in test.checks)
    result = dict(status=status, checks=len(test.checks), seconds=time.monotonic() - begin)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
