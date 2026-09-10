#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check actual recall commands with independent files and exact context bytes.
# Inputs: recall program and fixture program. Output: check count. Exit: 0 pass, 1 fail.
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

CHECKS = 0
HEADER, OBJECT, QUERY = 128, 256, 8192
LINEAGE = (9000).to_bytes(16, "little")
MODEL = hashlib.sha256(b"prepared model").digest()
PROCESSOR = hashlib.sha256(b"prepared processor").digest()


def check(value, label):
    global CHECKS
    CHECKS += 1
    if not value:
        raise AssertionError(f"check {CHECKS}: {label}")


def run(command, status=0, timeout=45):
    result = subprocess.run([str(v) for v in command], capture_output=True, timeout=timeout)
    check(result.returncode == status,
          f"exit {result.returncode}, expected {status}: {result.stderr.decode(errors='replace')}")
    return result


def put(data, offset, value, width=8):
    data[offset:offset + width] = value.to_bytes(width, "little")


def get(data, offset, width=8):
    return int.from_bytes(data[offset:offset + width], "little")


def identity(value):
    return value.to_bytes(16, "little")


def object_row(kind, oid, principal, sequence, evidence=0):
    row = bytearray(OBJECT)
    put(row, 0, 1, 2)
    put(row, 2, kind, 2)
    row[8:24], row[24:40] = identity(oid), LINEAGE
    for offset, value in ((40, 1), (48, sequence), (56, sequence), (232, 1)):
        put(row, offset, value)
    row[64:80], row[120:136] = identity(principal), identity(principal)
    put(row, 180, 1 if kind == 9 else 3, 4)
    put(row, 184, evidence, 4)
    put(row, 192, 0xFFFFFFFF, 4)
    return row


def image(objects, sequence, tick, tail=False):
    payload = bytearray()
    rows = bytearray()
    for row, value in objects:
        row = bytearray(row)
        put(row, 160, len(payload))
        put(row, 168, len(value))
        rows.extend(row)
        payload.extend(value)
    data = bytearray(HEADER) + rows + payload
    data[:8] = b"AOTXLOG1" if tail else b"AOTXOBJ1"
    for offset, value, width in ((8, 1, 4), (12, HEADER, 4), (16, OBJECT, 4),
                                 (20, len(objects), 4), (24, len(payload), 8),
                                 (32, sequence, 8), (40, tick, 8), (64, HEADER, 8),
                                 (72, HEADER + len(rows), 8), (80, len(data), 8), (88, 1, 4)):
        put(data, offset, value, width)
    data[48:64] = LINEAGE
    return bytes(data)


def fixtures(n):
    vectors, memories, query_rows, contexts = [], [], [], []
    for i in range(n):
        text = f'Person {i}: "meal" \\ path\n\ttest caf\u00e9 {101 + 7 * i}'.encode()
        vector = struct.pack("<3f", i + 1, 2 * i + 3, 7)
        payload = bytearray(128) + vector
        payload[:8] = b"AOTXVEC1"
        for offset, value in ((8, 1), (12, 3), (16, 4), (20, 1)):
            put(payload, offset, value, 4)
        payload[24:56], payload[56:88] = MODEL, PROCESSOR
        payload[88:120] = hashlib.sha256(text).digest()
        vectors.append((object_row(9, 1000 + i, 10000 + i, i + 1), payload))
        row = object_row(1, 2000 + i, 10000 + i, n + i + 1, i % 3)
        row[208:224] = identity(1000 + i)
        put(row, 224, 1)
        payload = bytearray(32) + text
        payload[:8] = b"AOTXMEM1"
        put(payload, 8, 1, 4)
        put(payload, 12, len(text), 4)
        memories.append((row, payload))
        query = bytearray(QUERY)
        query[:16], query[16:32] = identity(3000 + i), identity(10000 + i)
        query[48:64], query[64:96], query[96:128] = identity(4000 + i), MODEL, PROCESSOR
        current = f'Input {i}: "cook"\n\tnow \\ caf\u00e9'.encode()
        for offset, value in ((128, 3), (132, 4), (136, 4096), (148, len(current))):
            put(query, offset, value, 4)
        query[160:172] = vector
        query[4640:4640 + len(current)] = current
        reason = 3
        if i % 3 in (0, 1):
            put(query, 144, 1, 4)
            query[4448:4464] = identity(2000 + i)
            put(query, 4464, 1)
            reason = 2
        if i % 3 == 0:
            put(query, 140, 1, 4)
            query[4256:4272] = identity(2000 + i)
            put(query, 4272, 1)
            reason = 1
        prefix = (f"[memory id={identity(2000 + i).hex()} version=1 source=3 "
                  f"evidence={i % 3} reason={reason}]\n").encode()
        contexts.append(prefix + text + b"\n[input]\n" + current)
        query_rows.append(query)
    requests = bytearray(64) + b"".join(query_rows)
    requests[:8] = b"AOTXREQ1"
    put(requests, 8, n, 4)
    put(requests, 12, 1, 4)
    requests[16:32] = LINEAGE
    put(requests, 32, 2 * n)
    put(requests, 40, QUERY, 4)
    return (image(vectors, n, 10), image(memories, n + 1, 11, True),
            bytes(requests), contexts, vectors + memories)


def container(path):
    data = path.read_bytes()
    check(hashlib.sha256(data[:4064]).digest() == data[4064:4096], "prologue digest")
    roots = [data[4096:8192], data[8192:12288]]
    roots = [row for row in roots if any(row)]
    check(bool(roots), "published root exists")
    root = max(roots, key=lambda row: get(row, 16))
    check(hashlib.sha256(root[:4064]).digest() == root[4064:], "root digest")
    start, length = get(root, 24), get(root, 32)
    commit = data[start:start + length]
    check(hashlib.sha256(commit).digest() == root[48:80], "commit digest")
    start, count = get(commit, 72), get(commit, 80, 4)
    directory = data[start:start + count * 128]
    check(hashlib.sha256(directory).digest() == commit[96:128], "directory digest")
    sections = {}
    for i in range(count):
        row = directory[i * 128:(i + 1) * 128]
        start, length = get(row, 24), get(row, 32)
        payload = data[start:start + length]
        check(len(payload) == length and hashlib.sha256(payload).digest() == row[56:88], "section digest")
        sections[row[8:24]] = (get(row, 0, 4), get(row, 4, 2), get(row, 6, 2), payload)
    return (get(commit, 48), get(commit, 56), get(commit, 64)), sections


def state_rows(checkpoint):
    count, payload_start = get(checkpoint, 20, 4), get(checkpoint, 72)
    rows = []
    for i in range(count):
        row = checkpoint[128 + i * 256:128 + (i + 1) * 256]
        start, length = payload_start + get(row, 160), get(row, 168)
        rows.append((row, checkpoint[start:start + length]))
    return rows


def case(program, fixture, base, n, lifecycle=False):
    directory = base / f"batch-{n}-{int(lifecycle)}"
    directory.mkdir()
    raw, tail, requests = directory / "checkpoint", directory / "tail", directory / "requests"
    source = directory / "source ' \n;$().aotxccir"
    output = directory / "selected.aotxccir"
    checkpoint, log, queries, contexts, original = fixtures(n)
    if lifecycle:
        checkpoint = bytearray(image(original, 2 * n, 11))
        put(checkpoint, 8, 2, 4); put(checkpoint, 96, 2 * n); put(checkpoint, 124, 80, 4)
    raw.write_bytes(checkpoint)
    tail.write_bytes(log)
    requests.write_bytes(queries)
    run([fixture, raw, "-" if lifecycle else tail, source, n])
    source_bytes = source.read_bytes()
    source_meta, source_sections = container(source)
    check(source_meta == (2 * n if lifecycle else n, 2 * n, 11), "source checkpoint and tail metadata")
    fifo, fifo_output = directory / "requests.fifo", directory / "fifo-output.aotxccir"
    os.mkfifo(fifo)
    failed = run([program, "select", source, fifo, fifo_output], 1, timeout=5)
    check(not failed.stdout and failed.stderr, "FIFO request refuses without context")
    check(not fifo_output.exists() and source.read_bytes() == source_bytes, "FIFO refusal retains source and creates no output")
    fifo.unlink()
    result = run([program, "select", source, requests, output])
    check(not result.stderr, "successful selection has no errors")
    selected = [json.loads(line) for line in result.stdout.splitlines()]
    check(len(selected) == n, "one output per distinct query")
    check(source.read_bytes() == source_bytes, "select keeps source bytes")
    for i, row in enumerate(selected):
        check(set(row) == {"request", "selection", "cut", "count", "searches", "context"}, "JSON fields")
        check(row["request"] == identity(3000 + i).hex(), "request ID")
        check(row["selection"] == identity(4000 + i).hex(), "selection ID")
        check((row["cut"], row["count"], row["searches"]) == (2 * n, 1, 1), "selection counters")
        check(row["context"].encode() == contexts[i], "complete exact context and JSON escaping")
    metadata, sections = container(output)
    check(metadata == (4 * n, 4 * n, 12), "new sequence and tick")
    check(len(sections) == n + 2, "folded tail and retained optional count")
    optional = {key: value for key, value in source_sections.items() if value[0] >= 9000}
    check(len(optional) == n, "optional comparison is populated")
    for key, value in optional.items():
        check(sections.get(key) == value, "optional ID, schema, flags and bytes remain exact")
    states = [value[3] for value in sections.values() if value[0] == 2]
    check(len(states) == 1, "one checkpoint")
    objects = state_rows(states[0])
    check(len(objects) == 4 * n, "two stored objects per request")
    for i, (row, payload) in enumerate(original):
        actual, value = objects[i]
        expected = bytearray(row)
        put(expected, 160, get(actual, 160))
        put(expected, 168, len(payload))
        check(actual == expected and value == payload, "original object and prepared payload remain exact")
    records = {row[8:24]: (row, payload) for row, payload in objects[2 * n:]}
    for i in range(n):
        row, payload = records[identity(3000 + i)]
        check(get(row, 2, 2) == 1 and get(row, 40) == (2 * n + 2 * i + 1 if lifecycle else 1),
              "recorded request event and exact version")
        check(payload == b"AOTXQUE1" + (2 * n).to_bytes(8, "little") +
              queries[64 + i * QUERY:64 + (i + 1) * QUERY], "exact original cut and query bytes")
        row, payload = records[identity(4000 + i)]
        check(get(row, 2, 2) == 10 and row[96:112] == identity(3000 + i) and
              get(row, 40) == (2 * n + 2 * i + 2 if lifecycle else 1) and
              get(row, 112) == (2 * n + 2 * i + 1 if lifecycle else 1), "selection source and exact version binding")
        check(len(payload) == 48 and get(payload, 4, 4) == 1 and
              payload[16:32] == identity(2000 + i) and get(payload, 32) == 1, "exact selected version")
    saved_bytes = output.read_bytes()
    for destination in (output, source, directory / "absent" / "output"):
        failed = run([program, "select", source, requests, destination], 1)
        check(not failed.stdout and failed.stderr, "failed output publishes no context")
        check(source.read_bytes() == source_bytes and output.read_bytes() == saved_bytes, "failed output retains files")
    failures = []
    bad = bytearray(queries)
    put(bad, 8, 65, 4)
    failures.append(bad)
    failures.extend((queries + b"x", queries[:-1]))
    bad = bytearray(queries)
    bad[64 + (n - 1) * QUERY + 156] = 1
    failures.append(bad)
    bad = bytearray(queries)
    put(bad, 32, 2 * n + 1)
    failures.append(bad)
    bad = bytearray(queries)
    bad[64 + 16:64 + 32] = identity(999999)
    failures.append(bad)
    for i, data in enumerate(failures):
        bad_path, destination = directory / f"bad-{i}", directory / f"refused-{i}"
        bad_path.write_bytes(data)
        failed = run([program, "select", source, bad_path, destination], 1)
        check(not failed.stdout and failed.stderr, "invalid batch publishes no context")
        check(not destination.exists() and source.read_bytes() == source_bytes, "invalid batch has no partial file")
    for path in (source, requests, raw, tail):
        path.unlink()
    check(all(not path.exists() for path in (source, requests, raw, tail)), "all original paths removed")
    replay = run([program, "replay", output])
    check(not replay.stderr, "replay has no errors")
    restored = [json.loads(line) for line in replay.stdout.splitlines()]
    check(len(restored) == n, "replay row count")
    for before, after in zip(selected, restored):
        check(after["searches"] == 0, "replay performs zero searches")
        expected = dict(before, searches=0)
        check(after == expected, "replay keeps exact context, IDs, order and original cut")
    check(output.read_bytes() == saved_bytes, "replay does not write the file")
    withdrawn = bytearray(states[0])
    put(withdrawn, 128 + (2 * n - 1) * 256 + 184, 3, 4)
    raw.write_bytes(withdrawn)
    refused = directory / "withdrawn.aotxccir"
    run([fixture, raw, "-", refused, n])
    raw.unlink()
    refused_bytes = refused.read_bytes()
    failure = run([program, "replay", refused], 1)
    check(not failure.stdout and failure.stderr, "withdrawn final selection suppresses all replay output")
    check(refused.read_bytes() == refused_bytes, "refused replay keeps file bytes")
    print(f"recall CLI N={n} complete")


def main():
    if len(sys.argv) != 3:
        return 2
    program, fixture = map(pathlib.Path, sys.argv[1:])
    help_result = run([program, "--help"])
    check(b"select INPUT REQUESTS OUTPUT" in help_result.stdout, "select help")
    check(b"replay INPUT" in help_result.stdout, "replay help")
    run([program, "select"], 2)
    with tempfile.TemporaryDirectory(prefix="aotx-recall-") as temporary:
        for n in (1, 64):
            case(program, fixture, pathlib.Path(temporary), n)
            case(program, fixture, pathlib.Path(temporary), n, True)
    print(f"recall CLI: {CHECKS} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
