#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check live memory through a real boot, file feeder and journal restore.
# Owns: One new output directory, its input files and the boot processes it starts.
# Threading: One disk test driver; the boot process runs device batches.
# Lifetime: One bounded test with an existing language model store.

# Inputs: build, source, model store and new output directory. Output: logs and checks.
# Exit: 0 pass, 1 structure failure, 2 response failure, 3 both, 4 bad arguments.
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
INPUT = "State the check color in the memory block. Reply with one word."
FIELDS = ("agent", "turn", "input_hash", "output_hash", "tokens", "finish", "tool", "request")


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def wait(predicate, child=None, seconds=180):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"boot exited before completion: {child.returncode}")
        time.sleep(0.1)
    raise TimeoutError("the required state did not arrive before the time limit")


def process_state(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def children_of(pid):
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        return {int(child): process_state(int(child)) for child in path.read_text().split()}
    except FileNotFoundError:
        return {}


def children_exited(children):
    return all((current := process_state(pid)) is None or current[0] == "Z" or
               old is None or current[1] != old[1] for pid, old in children.items())


class Test:
    def __init__(self, build, source, store, output, snapshot_every=1):
        self.build, self.source, self.store, self.output = build, source, store, output
        if snapshot_every < 1:
            raise ValueError("check snapshot interval must be positive")
        self.snapshot_every = snapshot_every
        self.checks, self.active, self.commands = [], [], 0
        output.mkdir(parents=True, exist_ok=False)

    def record(self, **entry):
        entry["time_ns"] = time.time_ns()
        with (self.output / "commands.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")

    def check(self, value, label, kind="structure", **details):
        self.checks.append(dict(check=label, passed=bool(value), kind=kind, **details))
        if not value or len(self.checks) % self.snapshot_every == 0:
            self.flush_checks()
        print(f"{label}: {'PASS' if value else 'FAIL'}", flush=True)
        if not value and kind == "structure":
            raise AssertionError(label)

    def flush_checks(self):
        (self.output / "checks.json").write_text(json.dumps(self.checks, indent=2) + "\n")

    def command(self, argv, label):
        self.commands += 1
        begin = time.monotonic()
        argv = list(map(str, argv))
        path = self.output / f"{self.commands:03d}-{label}.log"
        self.record(command=argv, output=str(path))
        with path.open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        self.record(exit=result.returncode, seconds=time.monotonic() - begin, output=str(path))
        self.check(result.returncode == 0, f"{label} command exit")
        return path.read_text()


class Run:
    def __init__(self, test, label, restore=False):
        self.test, self.boot = test, None
        self.journal = test.output / "journal"
        self.path = test.output / f"{label}-boot.log"
        self.log = self.path.open("w")
        argv = ["stdbuf", "-oL", "-eL", test.build / "aotx_boot", "--models", test.store,
                "--roles", "language", "--journal", self.journal, "--modules", test.output / "modules",
                "--settings", test.output / "settings", "--ticks", "0"]
        if restore:
            argv.append("--restore")
        self.started = time.time_ns()
        test.record(command=list(map(str, argv)), output=str(self.path))
        self.child = subprocess.Popen(list(map(str, argv)), cwd=test.source, stdin=subprocess.PIPE,
                                      stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.children = {}
        test.active.append(self)
        test.record(pid=self.child.pid)

    def ready(self):
        phase = self.journal / "phase"
        def running():
            match = re.search(r"^boot: id ([0-9a-f]+)", self.path.read_text(), re.M)
            if match and phase.exists() and phase.stat().st_mtime_ns >= self.started:
                self.boot = f"{int(match[1], 16):016x}"
                return phase.read_text().startswith("running ")
            return False
        wait(running, self.child, 360)
        self.children.update(children_of(self.child.pid))

    def send(self, text):
        self.test.record(pid=self.child.pid, stdin=text)
        self.child.stdin.write(text + "\n")
        self.child.stdin.flush()

    def console(self):
        path = self.journal / self.boot / "console.log"
        return path.read_text() if path.exists() else ""

    def events(self, agent):
        return rows(self.journal / self.boot / "transcript" / f"{agent}.jsonl")

    def turns(self):
        return rows(self.journal / "manifest" / f"{self.boot}.jsonl")

    def operation(self, name, path, op, count):
        pattern = r"memory: operation (\d+) status (\d+) rows (\d+)"
        before = len(re.findall(pattern, self.console()))
        self.send(f"memory {name} {path}")
        def completed():
            found = re.findall(pattern, self.console())
            return found[before] if len(found) > before else None
        value = wait(completed, self.child)
        self.test.check(tuple(map(int, value)) == (op, 0, count), f"memory {name} device verdict", verdict=value)

    def reply(self, agent, turn, expected):
        wait(lambda: any(r.get("agent") == agent and r.get("turn") == turn for r in self.turns()), self.child)
        reply = wait(lambda: next((r for r in self.events(agent) if r.get("kind") == "reply"
                                   and r.get("turn") == turn), None), self.child)
        self.test.check(bool(reply.get("text", "").strip()), f"agent {agent} turn {turn} has a reply", reply=reply)
        self.test.check(not any(r.get("status") == "prompt_refused" for r in self.events(agent)), "prompt capacity admitted")
        self.test.check(bool(re.fullmatch(r"\W*" + expected + r"\W*", reply["text"].strip().lower())),
                        f"agent {agent} turn {turn} says {expected}", "behavior", reply=reply["text"])
        return reply

    def summary(self):
        text = self.test.command([self.test.build / "aotx_restore", "--journal", self.journal, "--summary"], "summary")
        line = next(line for line in text.splitlines() if line.startswith("restore boot="))
        result = dict(re.findall(r"(\w+)=([^\s]+)", line))
        self.test.check(int(result["last_tick"]) > 0 and int(result["replayed"]) > 0, "populated durable summary")
        return result

    def stop(self, killed=False):
        self.children.update(children_of(self.child.pid))
        if killed:
            self.test.record(pid=self.child.pid, signal="SIGKILL")
            self.child.kill()
        else:
            self.send("quit")
        status = self.child.wait(timeout=60)
        self.child.stdin.close()
        self.test.record(pid=self.child.pid, exit=status)
        wait(lambda: children_exited(self.children), seconds=60)
        self.test.check(status == (-signal.SIGKILL if killed else 0), "owned boot stop status")
        self.test.check(True, "owned boot children exited", children=sorted(self.children))
        self.log.close()

    def close(self):
        self.children.update(children_of(self.child.pid))
        if self.child.poll() is None:
            self.test.record(pid=self.child.pid, signal="SIGTERM")
            self.child.terminate()
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.test.record(pid=self.child.pid, signal="SIGKILL")
                self.child.kill()
                self.child.wait(timeout=30)
        wait(lambda: children_exited(self.children), seconds=60)
        if not self.child.stdin.closed:
            self.child.stdin.close()
        self.log.close()


def envelope(fixture, magic, row_bytes, sequence):
    value = bytearray(64 + row_bytes)
    value[:8], value[16:32] = magic, fixture.LINEAGE
    for offset, number, width in ((8, 1, 4), (12, 1, 4), (32, sequence, 8), (40, row_bytes, 4)):
        fixture.put(value, offset, number, width)
    return value


def files(test):
    spec = importlib.util.spec_from_file_location("recall_bytes", test.source / "tests/recall_cli_test.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    _, _, queries, _, objects = f.fixtures(2)
    def memory(text):
        value = bytearray(32) + text.encode()
        value[:8] = b"AOTXMEM1"
        f.put(value, 8, 1, 4); f.put(value, 12, len(text), 4)
        return value
    for i, color in enumerate(("amber", "teal")):
        text = f"The check color is {color}."
        f.put(objects[2 + i][0], 2, 2, 2)
        objects[2 + i] = (objects[2 + i][0], memory(text))
        objects[i][1][88:120] = hashlib.sha256(text.encode()).digest()
    source = test.output / "inputs"
    source.mkdir()
    checkpoint, tail, ccir = source / "checkpoint", source / "tail", source / "memory.aotxccir"
    checkpoint.write_bytes(f.image(objects[:2], 2, 10))
    tail.write_bytes(f.image(objects[2:], 3, 11, True))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, tail, ccir, 1], "pack")
    bind = envelope(f, b"AOTXBND1", 64, 4)
    bind[72:88], bind[104:120] = f.identity(10000), f.identity(8000)
    f.put(bind, 120, 160, 4)
    (source / "bind").write_bytes(bind)
    correction = bytearray(objects[2][0])
    f.put(correction, 40, 2); f.put(correction, 56, 5)
    (source / "update").write_bytes(f.image([(correction, memory("The check color is violet."))], 5, 12, True))
    def query(ordinal):
        sequence, version = (4, 1) if ordinal == 1 else (5, 2)
        value = envelope(f, b"AOTXLIV1", 8256, sequence)
        value[80:96] = f.identity(8000); f.put(value, 96, ordinal)
        q = bytearray(queries[64:64 + f.QUERY])
        q[:16], q[48:64] = f.identity(3000 + ordinal), f.identity(4000 + ordinal)
        for at, number in ((132, 1), (136, 512), (140, 1), (144, 0), (148, len(INPUT))):
            f.put(q, at, number, 4)
        f.put(q, 4272, version)
        q[4448:] = bytes(len(q) - 4448)
        q[4640:4640 + len(INPUT)] = INPUT.encode()
        value[128:] = q
        return value
    for ordinal in (1, 2):
        (source / f"query-{ordinal}").write_bytes(query(ordinal))
    test.record(inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    return f, query


def audit(test, run, fixture, ordinal, version):
    events = run.events(0)
    lines = [r for r in events if r.get("kind") == "line" and r.get("status") == "accepted"
             and r.get("turn") == ordinal]
    selected = [r for r in events if r.get("kind") == "selection" and r.get("turn") == ordinal]
    test.check(len(lines) == 1 and lines[0]["text"] == INPUT and lines[0]["status"] == "accepted", "exact typed audit input")
    marker = f"request {fixture.identity(3000 + ordinal).hex()}"
    matches = [r for r in selected if marker in r.get("text", "")]
    test.check(len(matches) == 1 and matches[0]["status"] == "selected", "exact typed audit choice")
    text = matches[0]["text"]
    for marker in (f"ordinal {ordinal} ", f"objects {fixture.identity(2000).hex()}@{version}",
                   f"conversation {fixture.identity(8000).hex()}", f"principal {fixture.identity(10000).hex()}",
                   f"selection {fixture.identity(4000 + ordinal).hex()}", "scope 0 "):
        test.check(marker in text, "typed audit field", field=marker)
    test.check(fixture.identity(2001).hex() not in text, "other principal is absent from selected objects")


def exercise(test):
    (test.output / "settings").write_text("sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 32\n"
        "decode.think_limit = 0\ntools.mask = 0\nagent.recall_k = 0\nagent.compact_at = 128\n"
        "derive.list = console,bus,transcript,tokens,pages\n")
    shutil.copytree(test.source / "modules/roles", test.output / "modules")
    test.record(store=str(test.store), model_entries=rows(test.store / "manifest.jsonl"), roles="language")
    fixture, query = files(test)
    before = Run(test, "before")
    before.ready()
    before.send("spawn worker")
    wait(lambda: "spawn: worker on slots 1" in before.console(), before.child)
    before.send("agent 1 pages 160")
    before.send("task 1 Reply with the word ready.")
    before.reply(1, 1, "ready")
    original = test.output / "inputs"
    before.operation("load", original / "memory.aotxccir", 1, 0)
    before.operation("bind", original / "bind", 3, 1)
    before.send("task 0 Refused ordinary input before the typed request.")
    for ordinal, color in ((1, "amber"), (2, "violet")):
        if ordinal == 2:
            before.operation("apply", original / "update", 2, 0)
        before.operation("query", original / f"query-{ordinal}", 4, 1)
        before.reply(0, ordinal, color)
        audit(test, before, fixture, ordinal, ordinal)
    before.summary()
    before.stop(killed=True)
    old = before.summary()
    old_turns, old_audit = before.turns(), before.events(0)
    test.check({(r["agent"], r["turn"]) for r in old_turns} == {(1, 1), (0, 1), (0, 2)}, "all original turns are present")
    (test.output / "original-manifests.json").write_text(json.dumps(old_turns, indent=2) + "\n")
    paths = list(original.iterdir())
    shutil.rmtree(original)
    test.check(all(not p.exists() for p in paths), "original typed files are absent", paths=list(map(str, paths)))
    after = Run(test, "after", restore=True)
    after.ready()
    match = wait(lambda: re.search(r"^restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                 r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(match[1]) == int(old["replayed"]) and int(match[2], 16) == int(old["state_hash"], 16), "exact restored record count and hash")
    test.check(int(match[3]) == 0 and int(match[6]) == 0, "restore has no rejected records or decode refusals")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore parent and hash")
    wait(lambda: len(after.turns()) >= len(old_turns), after.child)
    key = lambda data: sorted(tuple(row[field] for field in FIELDS) for row in data)
    test.check(key(after.turns()) == key(old_turns), "exact replayed manifest fields and prompt input hashes")
    wait(lambda: len(after.events(0)) >= len(old_audit), after.child)
    test.check(after.events(0) == old_audit, "exact replayed typed audit and replies")
    test.check(all(not p.exists() for p in paths), "restore did not recreate its input dependencies")
    next_query = test.output / "next-query"
    next_query.write_bytes(query(3))
    after.operation("query", next_query, 4, 1)
    after.reply(0, 3, "violet")
    audit(test, after, fixture, 3, 2)
    after.stop()
    after.summary()
    test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) != 5:
        print("usage: live_boot_test.py BUILD SOURCE STORE OUTPUT", file=sys.stderr)
        return 4
    test = Test(*(Path(v).resolve() for v in sys.argv[1:]))
    begin, status = time.monotonic(), 0
    try:
        exercise(test)
    except Exception as error:
        status = 1
        (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"live boot failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1
                test.record(cleanup_error=str(error))
        if any(not row["passed"] and row["kind"] == "behavior" for row in test.checks):
            status |= 2
        result = dict(checks=len(test.checks), failed=sum(not row["passed"] for row in test.checks),
                      seconds=time.monotonic() - begin, exit=status)
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
