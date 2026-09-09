#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check text embedding, live recall and exact recovery through real processes.
# Owns: New output directories, input files and the boot processes started here.
# Threading: One disk driver; the device prepares and consumes query batches.
# Lifetime: One bounded integration test with existing language and embedding models.

# Inputs: build, source, model store and new output directory. Output: logs and checks.
# Exit: 0 pass, 1 failed check or cleanup, 4 bad arguments.
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, INPUT, FIELDS, envelope, rows, wait
PROCESSOR = bytes.fromhex("7d12af1d2cd1e5194def983d1fd8073d1c36c444eea39c2dcf9bbe394e75892d")


class TextRun(Run):
    def __init__(self, test, label, restore=False):
        self.test, self.boot = test, None
        self.journal = test.output / "journal"
        self.path = test.output / f"{label}-boot.log"
        self.log = self.path.open("w")
        argv = ["stdbuf", "-oL", "-eL", test.build / "aotx_boot", "--models", test.store,
                "--roles", "language,embedding", "--journal", self.journal,
                "--modules", test.output / "modules", "--settings", test.output / "settings", "--ticks", "0"]
        if restore:
            argv.append("--restore")
        self.started = time.time_ns()
        test.record(command=list(map(str, argv)), output=str(self.path))
        self.child = subprocess.Popen(list(map(str, argv)), cwd=test.source, stdin=subprocess.PIPE,
                                      stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.children = {}
        test.active.append(self)
        test.record(pid=self.child.pid)

    def reply(self, turn):
        wait(lambda: any(r.get("agent") == 0 and r.get("turn") == turn for r in self.turns()), self.child)
        row = wait(lambda: next((r for r in self.events(0) if r.get("kind") == "reply"
                                and r.get("turn") == turn), None), self.child)
        self.test.check(bool(row.get("text", "").strip()), "text request has a language reply", reply=row)
        self.test.check(not any(r.get("status") == "prompt_refused" for r in self.events(0)), "text context fits the prompt")


def setup(test):
    (test.output / "settings").write_text("sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 32\n"
        "decode.think_limit = 0\ntools.mask = 0\nagent.recall_k = 0\nagent.compact_at = 128\n"
        "derive.list = console,bus,transcript,tokens,pages\n")
    shutil.copytree(test.source / "modules/roles", test.output / "modules")
    entries = rows(test.store / "manifest.jsonl")
    test.record(model_entries=entries, roles="language,embedding", claim="integration only")
    test.check({"language", "embedding"} <= {r.get("role") for r in entries}, "both local model roles are present")
    spec = importlib.util.spec_from_file_location("recall_bytes", test.source / "tests/recall_cli_test.py")
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    (test.output / "inputs").mkdir()
    return f


def query(f, cut, ordinal, required=False):
    value = envelope(f, b"AOTXTXT1", 8256, cut)
    value[80:96] = f.identity(8000); f.put(value, 96, ordinal)
    q = bytearray(8192)
    q[:16], q[16:32], q[48:64] = f.identity(3000 + ordinal), f.identity(10000), f.identity(4000 + ordinal)
    for offset, number in ((132, 1), (136, 512), (140, int(required)), (148, len(INPUT))):
        f.put(q, offset, number, 4)
    if required:
        q[4256:4272] = f.identity(2000); f.put(q, 4272, 1)
    q[4640:4640 + len(INPUT)] = INPUT.encode()
    value[128:] = q
    return value


def pack(test, f, objects):
    source = test.output / "inputs"
    checkpoint, ccir = source / "checkpoint", source / "memory.aotxccir"
    checkpoint.write_bytes(f.image(objects, len(objects), 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack")
    bind = envelope(f, b"AOTXBND1", 64, len(objects))
    bind[72:88], bind[104:120] = f.identity(10000), f.identity(8000); f.put(bind, 120, 160, 4)
    (source / "bind").write_bytes(bind)
    return ccir, source / "bind"


def decisions(test, run, f):
    text = test.command([test.build / "aotx_journal", "records", run.journal, "--boot", run.boot], "records")
    complete, active, identity, total = [], bytearray(), None, 0
    for line in text.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields.get("type") != "33" or fields.get("class") != "1":
            continue
        p = bytes.fromhex(fields["body"])
        test.check(len(p) == int(fields["body_len"]) and 32 < len(p) <= 192, "typed journal part bounds")
        if f.get(p, 4, 4) != 7:
            continue
        size, offset = f.get(p, 24, 4), f.get(p, 28, 4)
        if not offset:
            test.check(not active and 64 <= size <= 562240, "bounded decision starts in order")
            total, identity = size, p[8:24]
        test.check(p[8:24] == identity and size == total and offset == len(active), "exact decision fragment order")
        test.check(len(p) - 32 == min(160, total - offset), "exact decision fragment length")
        active.extend(p[32:])
        if len(active) == total:
            test.check(active[:8] == b"AOTXTCH1" and f.get(active, 40, 4) == 8784 and
                       f.get(active, 44, 4) == 0, "successful text decision header")
            complete.append(bytes(active)); active.clear(); identity = None
    test.check(not active and bool(complete), "complete text decisions exist")
    return complete


def seed(test):
    f = setup(test)
    ccir, bind = pack(test, f, [])
    run = TextRun(test, "seed"); run.ready()
    run.operation("load", ccir, 1, 0); run.operation("bind", bind, 3, 1)
    path = test.output / "inputs/text"
    path.write_bytes(query(f, 0, 1))
    run.operation("text", path, 6, 1); run.reply(1); run.stop()
    found = decisions(test, run, f)
    test.check(len(found) == 1 and f.get(found[0], 8, 4) == 1, "one generated query row")
    prepared = found[0][128:8320]
    width = f.get(prepared, 128, 4)
    test.check(0 < width <= 1024 and any(prepared[64:96]) and any(prepared[96:128]), "generated model, processor and width")
    test.check(f.get(found[0], 8320 + 4, 4) == 0, "empty store produces an empty selection")
    embedding = next(r for r in rows(test.store / "manifest.jsonl") if r["role"] == "embedding")
    test.check(prepared[64:96].hex() == embedding["sha256"], "prepared query identifies loaded embedding weights")
    test.check(prepared[96:128] == PROCESSOR, "prepared query identifies the fixed processor")
    (test.output / "prepared-query.bin").write_bytes(prepared)
    shutil.rmtree(test.output / "inputs")
    return prepared


def corpus(f, prepared):
    width = f.get(prepared, 128, 4)
    vectors, memories = [], []
    for i, color in enumerate(("amber", "teal", "coral")):
        principal = 10000 + (i == 2)
        text = f"The check color is {color}.".encode()
        vector = bytearray(prepared[160:160 + width * 4])
        if i == 1:
            for offset in range(3, len(vector), 4):
                vector[offset] ^= 0x80
        payload = bytearray(128) + vector
        payload[:8] = b"AOTXVEC1"
        for offset, number in ((8, 1), (12, width), (16, 4), (20, 1)):
            f.put(payload, offset, number, 4)
        payload[24:56], payload[56:88] = prepared[64:96], prepared[96:128]
        payload[88:120] = hashlib.sha256(text).digest()
        vectors.append((f.object_row(9, 1000 + i, principal, i + 1), payload))
        row = f.object_row(2, 2000 + i, principal, 4 + i)
        row[208:224] = f.identity(1000 + i); f.put(row, 224, 1)
        payload = bytearray(32) + text; payload[:8] = b"AOTXMEM1"
        f.put(payload, 8, 1, 4); f.put(payload, 12, len(text), 4)
        memories.append((row, payload))
    return vectors + memories


def audit(test, run, f, ordinal):
    entries = run.events(0)
    lines = [r for r in entries if r.get("kind") == "line" and r.get("status") == "accepted" and r.get("turn") == ordinal]
    choices = [r for r in entries if r.get("kind") == "selection" and r.get("turn") == ordinal]
    test.check(len(lines) == 1 and lines[0]["text"] == INPUT, "exact original text in audit")
    marker = f"request {f.identity(3000 + ordinal).hex()}"
    choices = [r for r in choices if marker in r.get("text", "")]
    test.check(len(choices) == 1 and choices[0]["status"] == "selected", "one matching text choice in audit")
    text = choices[0]["text"]
    test.check(f"objects {f.identity(2000).hex()}@1" in text and f.identity(2001).hex() not in text and
               f.identity(2002).hex() not in text, "positive vector selected within private scope")


def exercise(test, prepared):
    f = setup(test)
    ccir, bind = pack(test, f, corpus(f, prepared))
    run = TextRun(test, "before"); run.ready()
    run.send("spawn worker")
    wait(lambda: "spawn: worker on slots 1" in run.console(), run.child)
    run.send("agent 1 pages 160")
    run.send("task 1 Reply with the word ready.")
    Run.reply(run, 1, 1, "ready")
    run.operation("load", ccir, 1, 0); run.operation("bind", bind, 3, 1)
    run.send("task 0 Refused ordinary input before the typed request.")
    for ordinal in (1, 2):
        path = test.output / "inputs" / f"text-{ordinal}"
        path.write_bytes(query(f, 6, ordinal, ordinal == 1))
        run.operation("text", path, 6, 1); run.reply(ordinal); audit(test, run, f, ordinal)
    run.summary(); run.stop(killed=True)
    old, manifests, transcript = run.summary(), run.turns(), run.events(0)
    original = decisions(test, run, f)
    test.check(len(original) == 2 and f.get(original[1], 128 + 140, 4) == 0 and
               f.get(original[1], 8320 + 4, 4) == 1, "discretionary recall has no required pin")
    paths = list((test.output / "inputs").iterdir()); shutil.rmtree(test.output / "inputs")
    test.check(all(not p.exists() for p in paths), "original input files removed", paths=list(map(str, paths)))
    after = TextRun(test, "after", True); after.ready()
    match = wait(lambda: re.search(r"^restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                 r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(match[1]) == int(old["replayed"]) and int(match[2], 16) == int(old["state_hash"], 16), "exact restored records and hash")
    test.check(int(match[3]) == 0 and int(match[6]) == 0, "no restore decode or input refusal")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore parent and hash")
    wait(lambda: len(after.turns()) >= len(manifests) and len(after.events(0)) >= len(transcript), after.child)
    keys = lambda values: sorted(tuple(r[k] for k in FIELDS) for r in values)
    test.check(keys(after.turns()) == keys(manifests) and after.events(0) == transcript, "exact replayed prompts, replies and audit")
    test.check(decisions(test, after, f) == original, "prepared vectors and choices replay byte for byte")
    path = test.output / "next-text"; path.write_bytes(query(f, 6, 3))
    after.operation("text", path, 6, 1); after.reply(3); audit(test, after, f, 3)
    test.check(all(not p.exists() for p in paths), "source files remain absent after recovery")
    after.stop(); after.summary()
    test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) != 5:
        print("usage: text_boot_test.py BUILD SOURCE STORE OUTPUT", file=sys.stderr); return 4
    build, source, store, output = (Path(v).resolve() for v in sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    tests, status, begin = [], 0, time.monotonic()
    try:
        first = Test(build, source, store, output / "seed"); tests.append(first)
        prepared = seed(first)
        second = Test(build, source, store, output / "live"); tests.append(second)
        exercise(second, prepared)
    except Exception as error:
        status = 1; (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"text boot failed: {error}", file=sys.stderr, flush=True)
    finally:
        for test in reversed(tests):
            for run in reversed(test.active):
                try:
                    run.close()
                except Exception as error:
                    status = 1; test.record(cleanup_error=str(error))
        checks = [check for test in tests for check in test.checks]
        (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        result = dict(checks=len(checks), failed=sum(not r["passed"] for r in checks),
                      seconds=time.monotonic() - begin, exit=status, claim="integration only")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
