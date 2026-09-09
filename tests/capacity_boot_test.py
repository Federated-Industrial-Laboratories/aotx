#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check live recall and retention beyond the compact memory capacity.
# Owns: One new output directory and only the boot processes started here.
# Threading: One disk driver; the device owns memory, recall and language processing.
# Lifetime: One bounded test with a local language model and a cold journal restart.

# Inputs: build, source, model store, new output and optional batch count. Output: files, logs and exact checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, FIELDS, envelope, rows, wait
from retain_boot_test import compiled_limits, transfers

COLORS = ("amber", "blue", "green", "red", "gold", "white", "orange", "violet")


def color(slot, new=False):
    return COLORS[(slot + (7 if new else 0)) % len(COLORS)]


def prompt(slot, first=False):
    if first:
        return (f"Member {slot}: The new check color is {color(slot, True)}. "
                "State the private check color from memory. Reply with one word.")
    return f"Member {slot}: State the new check color in the remembered input. Reply with one word."


def vector(slot, sign=1):
    return struct.pack("<3f", float(sign), sign * slot / 128.0, 0.0)


def batch(f, magic, row_bytes, cut, count):
    data = envelope(f, magic, row_bytes, cut)
    data.extend(bytes((count - 1) * row_bytes)); f.put(data, 8, count, 4)
    return data


def memory(f, text):
    data = bytearray(32) + text.encode()
    data[:8] = b"AOTXMEM1"; f.put(data, 8, 1, 4); f.put(data, 12, len(text.encode()), 4)
    return data


def objects(f, count):
    values = []
    for i in range(1024):
        text = (f"Private archive {i}: " + chr(65 + i % 26) * 1024)[:1024]
        values.append((f.object_row(1, 50000 + i, 20000 + i, i + 1), memory(f, text)))
    for i in range(count):
        for j in range(2):
            source, component, item = (60000 + 6 * i + 3 * j + k for k in range(3))
            text = f"Member {i}: The private check color is {color(i + j)}."
            values.append((f.object_row(1, source, 10000 + i, len(values) + 1), memory(f, text)))
            row = f.object_row(9, component, 10000 + i, len(values) + 1)
            f.put(row, 180, 4, 4); row[96:112] = f.identity(source); f.put(row, 112, 1)
            data = bytearray(128) + vector(i, -1 if j else 1)
            data[:8] = b"AOTXVEC2"
            for offset, value in ((8, 2), (12, 3), (16, 4), (20, 1)):
                f.put(data, offset, value, 4)
            data[24:56], data[56:88], data[88:104] = f.MODEL, f.PROCESSOR, f.identity(source)
            f.put(data, 104, 1); values.append((row, data))
            row = f.object_row(7, item, 10000 + i, len(values) + 1)
            row[96:112] = f.identity(source); f.put(row, 112, 1)
            row[208:224] = f.identity(component); f.put(row, 224, 1)
            values.append((row, memory(f, text)))
    return values


def request(f, cut, ordinal, count):
    data = batch(f, b"AOTXLIV1", 8256, cut, count)
    for i in range(count):
        base = 64 + i * 8256
        f.put(data, base, i, 4); f.put(data, base + 4, int(ordinal > 1), 4)
        data[base + 16:base + 32] = f.identity(8000 + i); f.put(data, base + 32, ordinal)
        q, text = bytearray(8192), prompt(i, ordinal == 1).encode()
        q[:16], q[16:32] = f.identity(3001 + (ordinal - 1) * 64 + i), f.identity(10000 + i)
        q[48:64] = f.identity(4001 + (ordinal - 1) * 64 + i)
        q[64:96], q[96:128], q[160:172] = f.MODEL, f.PROCESSOR, vector(i)
        for offset, value in ((128, 3), (132, 1), (136, 512), (148, len(text))):
            f.put(q, offset, value, 4)
        q[4640:4640 + len(text)] = text; data[base + 64:base + 8256] = q
    return data


def retain_request(f, cut, count):
    data = batch(f, b"AOTXRTN1", 160, cut, count)
    for i in range(count):
        r = bytearray(160); f.put(r, 0, i, 4); f.put(r, 4, 1, 4)
        r[8:24] = f.identity(8000 + i); f.put(r, 24, 1)
        r[32:48], r[48:64], r[64:80] = (f.identity(n + i) for n in (3001, 2000, 1000))
        f.put(r, 104, 1); r[112:128] = f.identity(10000 + i)
        f.put(r, 128, 750000 + i, 4); f.put(r, 132, 1, 4)
        data[64 + i * 160:224 + i * 160] = r
    return data


def prepare(test, count):
    limits = compiled_limits(test)
    spec = importlib.util.spec_from_file_location("capacity_bytes", test.source / "tests/recall_cli_test.py")
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    (test.output / "settings").write_text("sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 32\n"
        "decode.think_limit = 0\ntools.mask = 0\nagent.recall_k = 0\nagent.compact_at = 128\n"
        "derive.list = console,bus,transcript,tokens,pages\n")
    shutil.copytree(test.source / "modules/roles", test.output / "modules")
    entries = rows(test.store / "manifest.jsonl")
    test.check(any(r.get("role") == "language" for r in entries), "local language model exists")
    test.record(model_entries=entries, roles="language", claim="capacity integration only")
    values = objects(f, count); image = f.image(values, len(values), 10)
    payload = f.get(image, 24)
    test.check(len(values) > 1024 and payload > 1048576 and len(values) + 3 * count <= limits["objects"] and
               payload + 1024 * count < limits["payload_bytes"], "store exceeds compact limits and fits compiled capacity")
    test.check(all(values[1024 + 6 * i + j][0][64:80] == f.identity(10000 + i)
                   for i in range(count) for j in range(6)), "distinct private dependencies exceed slot 255")
    inputs = test.output / "inputs"; inputs.mkdir()
    checkpoint, ccir, bind = inputs / "checkpoint", inputs / "memory.aotxccir", inputs / "bind"
    checkpoint.write_bytes(image)
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack")
    data = batch(f, b"AOTXBND1", 64, len(values), count)
    for i in range(count):
        base = 64 + i * 64; f.put(data, base, i, 4)
        data[base + 8:base + 24], data[base + 40:base + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, base + 56, 160, 4)
    bind.write_bytes(data)
    (test.output / "capacity.json").write_text(json.dumps(dict(compiled=limits, batch=count, objects=len(values),
        payload_bytes=payload, image_bytes=len(image), source_slot=1024, vector_slot=1025, memory_slot=1026), indent=2) + "\n")
    return f, inputs, ccir, bind, len(values)


def selected(test, run, f, turn, slot):
    identity = 60002 + 6 * slot if turn == 1 else 2000 + slot
    expected = f"objects {f.identity(identity).hex()}@1"
    audit = [r for r in run.events(slot) if r.get("kind") == "selection" and r.get("turn") == turn]
    test.check(len(audit) == 1 and audit[0]["text"].endswith(expected), "exact private selection in audit")
    test.check(len(audit) == 1 and f"principal {f.identity(10000 + slot).hex()}" in audit[0]["text"] and
               "scope 0 " in audit[0]["text"], "selection retains private principal")


def retained(test, f, records, original, cut, count):
    sources = [r for r in records if r[0] == 8]; results = [r for r in records if r[0] == 9]
    test.check(len(sources) == len(results) == 1 and sources[0][1] == results[0][1] and
               sources[0][2] == original, "exact retention request and matching result")
    data = results[0][2]
    test.check(data[:8] == b"AOTXRCH1" and f.get(data, 8, 4) == count and f.get(data, 12, 4) == 1 and
               f.get(data, 40, 4) == 384 and f.get(data, 44, 4) == 0, "retained batch admitted")
    tail = data[64 + 384 * count:]; stored = dict((r[8:24], (r, p)) for r, p in f.state_rows(tail))
    test.check(tail[:8] == b"AOTXLOG1" and len(tail) == f.get(data, 48) and f.get(tail, 20, 4) == 3 * count and
               f.get(tail, 32) == cut + 1, "canonical mutation follows the full store cut")
    test.check(set(stored) == {f.identity(n + i) for i in range(count) for n in (3001, 1000, 2000)}, "exact new object IDs")
    choices = [r[2] for r in records if r[0] == 5]; queries = [r[2] for r in records if r[0] == 4]
    test.check(len(choices) == len(queries) == 2, "two complete query and choice batches")
    for i in range(count):
        base = 64 + i * 384; row = data[base:base + 384]
        test.check(row[:160] == original[64 + i * 160:224 + i * 160] and f.get(row, 160, 4) == 1,
                   "retained input and focus admitted")
        test.check(row[192:208] == f.identity(2000 + i) and f.get(row, 208) == 1 and
                   not any(row[216:384]), "exact focus after large-store mutation")
        for identity in (3001 + i, 2000 + i):
            row, payload = stored[f.identity(identity)]
            test.check(payload == memory(f, prompt(i, True)) and row[64:80] == f.identity(10000 + i) and
                       f.get(row, 176, 4) == 0, "retained input has exact private text")
        row, encoded = stored[f.identity(1000 + i)]
        test.check(encoded[:8] == b"AOTXVEC2" and encoded[24:88] == f.MODEL + f.PROCESSOR and
                   encoded[88:104] == f.identity(3001 + i) and f.get(encoded, 104) == 1 and
                   encoded[128:] == vector(i) and row[96:112] == f.identity(3001 + i), "exact vector and typed dependency")
        qbase = 128 + i * 8256
        test.check(not any(queries[0][qbase + 140:qbase + 148]), "first query has no memory references")
        for chosen, identity in zip(choices, (60002 + 6 * i, 2000 + i)):
            base = 64 + i * 592
            test.check(f.get(chosen, 8, 4) == count and f.get(chosen, base, 4) == i and
                       f.get(chosen, base + 68, 4) == 1 and chosen[base + 80:base + 96] == f.identity(identity) and
                       f.get(chosen, base + 96) == 1, "recorded choice names exactly one expected memory")


def exercise(test, count):
    f, inputs, ccir, bind, cut = prepare(test, count)
    run = Run(test, "before"); run.ready()
    for start in range(1, max(count, 2), 8):
        size = min(8, max(count, 2) - start)
        run.send(f"spawn worker {size}")
        expected = "spawn: worker on slots " + " ".join(str(i) for i in range(start, start + size))
        wait(lambda: expected in run.console(), run.child)
    if count == 1:
        run.send("agent 1 pages 160"); run.send("task 1 Reply with the word ready."); run.reply(1, 1, "ready")
    run.operation("load", ccir, 1, 0); run.operation("bind", bind, 3, count)
    path = inputs / "query"; path.write_bytes(request(f, cut, 1, count))
    run.operation("query", path, 4, count)
    for i in range(count):
        run.reply(i, 1, color(i)); selected(test, run, f, 1, i)
    data = retain_request(f, cut, count); path = inputs / "retain"; path.write_bytes(data)
    run.operation("retain", path, 8, count)
    path = inputs / "focus"; path.write_bytes(request(f, cut + 3 * count, 2, count))
    run.operation("query", path, 4, count)
    for i in range(count):
        run.reply(i, 2, color(i, True)); selected(test, run, f, 2, i)
    run.summary(); run.stop(killed=True)
    old, manifests, audit = run.summary(), run.turns(), [run.events(i) for i in range(count)]
    original = transfers(test, run, f); retained(test, f, original, data, cut, count)
    paths = list(inputs.iterdir()); shutil.rmtree(inputs)
    test.check(all(not p.exists() for p in paths), "original state and request files removed")
    after = Run(test, "after", True); after.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(report[1]) == int(old["replayed"]) and int(report[2], 16) == int(old["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "exact large-store restore without refused work")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore identity and state hash")
    wait(lambda: len(after.turns()) >= len(manifests) and all(len(after.events(i)) >= len(audit[i]) for i in range(count)), after.child)
    keys = lambda data: sorted(tuple(r[k] for k in FIELDS) for r in data)
    test.check(keys(after.turns()) == keys(manifests) and [after.events(i) for i in range(count)] == audit, "exact restored prompts and private audit")
    test.check(transfers(test, after, f) == original, "exact restored queries, choices, mutation and focus")
    path = test.output / "next-query"; path.write_bytes(request(f, cut + 3 * count, 3, count))
    after.operation("query", path, 4, count)
    for i in range(count):
        after.reply(i, 3, color(i, True)); selected(test, after, f, 3, i)
    test.check(all(not p.exists() for p in paths), "original files remain absent after new recall")
    after.stop(); after.summary(); test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) not in (5, 6) or (len(sys.argv) == 6 and sys.argv[5] not in ("1", "64")):
        print("usage: capacity_boot_test.py BUILD SOURCE STORE OUTPUT [1|64]", file=sys.stderr); return 2
    count = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    test = Test(*(Path(v).resolve() for v in sys.argv[1:5]), snapshot_every=1024); status, start = 0, time.monotonic()
    try:
        exercise(test, count)
    except (Exception, KeyboardInterrupt) as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        if any(not r["passed"] for r in test.checks):
            status = 1
        test.flush_checks()
        result = dict(batch=count, checks=len(test.checks), failed=sum(not r["passed"] for r in test.checks),
                      seconds=time.monotonic() - start, exit=status, claim="capacity integration only")
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
