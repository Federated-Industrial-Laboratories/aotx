#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check native policy state, memory maintenance, and complete file recovery.
# Owns: One new output directory and the runtime processes started here.
# Threading: One disk driver; CUDA processes the private input batch and policy state.
# Lifetime: A training runtime and two activations of one complete CCIR file.

# Inputs: build, source, model store, output, batch, native image, and architecture.
# Output: exact commands, console logs, and checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Run, wait
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections, replay_records
from checkpoint_boot_test import spawn, durable as memory_durable, file_state
from text_boot_test import setup
from retain_boot_test import compiled_limits
from capacity_boot_test import batch, color, memory
from context_boot_bytes import corpus, request, selected


def policy_status(run, command="status"):
    start = len(run.console()); run.send("policy " + command)
    first = (r"policy: (\w+) mode (\d+) decision (\d+) memory source (\d+) "
             r"state bytes (\d+) status (\d+)")
    second = (r"policy: calls (\d+) last ns (\d+) maximum ns (\d+) "
              r"state hash (\d+) saved generation (\d+)")
    def received():
        text = run.console()[start:]
        a, b = re.search(first, text), re.search(second, text)
        if a and b:
            return dict(state=a[1], mode=int(a[2]), decision=int(a[3]), source=int(a[4]),
                        bytes=int(a[5]), status=int(a[6]), calls=int(b[1]), last_ns=int(b[2]),
                        maximum_ns=int(b[3]), state_hash=int(b[4]), generation=int(b[5]))
        return None
    row = wait(received, run.child, 60)
    run.test.check(row["mode"] == 3 and row["bytes"] == 16 and not row["status"] and
                   row["state"] != "error", "native policy console state is valid", policy=row)
    return row


def settled(run, source, minimum):
    def received():
        row = policy_status(run)
        return row if row["state"] == "quiet" and row["source"] == source and row["decision"] >= minimum else None
    return wait(received, run.child, 300)


def bind(run, f, count, cut, path):
    data = batch(f, b"AOTXBND1", 64, cut, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 160, 4); f.put(data, at + 60, 1, 4)
    path.write_bytes(data); run.operation("bind", path, 3, count)


def answer(run, f, count, cut, ordinal, revision, path):
    data = request(f, count, cut, 1, False)
    for i in range(count):
        at = 64 + i * 8256; f.put(data, at + 32, ordinal)
        data[at + 64:at + 80] = f.identity(50000000 + revision * 64 + i)
        data[at + 112:at + 128] = f.identity(51000000 + revision * 64 + i)
    path.write_bytes(data); run.operation("query", path, 4, count)
    for i in range(count):
        run.reply(i, ordinal, color(i))
        choices = [r for r in run.events(i) if r.get("kind") == "selection" and r.get("turn") == ordinal]
        expected = ",".join(k.hex() + "@1" for k in selected(f, i, 1, False))
        run.test.check(len(choices) == 1 and choices[0]["text"].endswith(" objects " + expected),
                       "actual reply recalls the exact private requirement and its source")
    return cut + 3 * count


def trained_memory(test, count):
    f = setup(test); limits = compiled_limits(test)
    fill = max(256, 12 * count, limits["objects"] // 8)
    pressure = (100 * (24 * count + fill) + 2 * limits["objects"] - 1) // (2 * limits["objects"])
    test.check(fill + 15 * count <= limits["objects"] and 0 < pressure <= 100 and
               15 * count * 100 < pressure * limits["objects"] < (fill + 9 * count) * 100,
               "configured pressure separates the initial store from retained memory")
    values = corpus(f, count, False)
    for row, _ in values:
        f.put(row, 48, f.get(row, 48) + fill); f.put(row, 56, f.get(row, 56) + fill)
    prefix = [(f.object_row(1, 9000000 + i, 9100000 + i, i + 1),
               memory(f, f"Archive entry {i}: " + chr(65 + i % 26) * 128)) for i in range(fill)]
    inputs = test.output / "inputs"; raw = inputs / "initial-checkpoint"
    initial, mirror = inputs / "initial.aotxccir", inputs / "trained-live.aotxccir"
    cut = len(prefix) + len(values); image = f.image(prefix + values, cut, 10)
    test.check(f.get(image, 24) + count * 65536 < limits["payload_bytes"], "prepared memory fits the payload capacity")
    raw.write_bytes(image)
    test.command([test.build / "aotx_recall_cli_fixture", raw, "-", initial, 1], "initial-memory")
    run = Run(test, "training", extra=("--memory-mirror", mirror)); run.ready(); spawn(run, count)
    run.operation("load", initial, 1, 0); memory_durable(run)
    bind(run, f, count, cut, inputs / "training-bind"); memory_durable(run)
    cut = answer(run, f, count, cut, 1, 1, inputs / "training-input")
    memory_durable(run); run.stop()
    stored = file_state(test, mirror, f, count, 1)[2]
    test.check(f.get(stored, 32) == cut and f.get(stored, 20, 4) == fill + 9 * count,
               "real typed input is retained before the creator runtime is packed")
    prepared = bytearray(stored)
    for offset, value, width in ((8, 2, 4), (96, cut, 8), (104, 0, 8), (112, 9 * count, 4),
                                  (116, 1, 4), (120, 1, 4), (124, pressure, 4)):
        f.put(prepared, offset, value, width)
    test.check(prepared[128:] == stored[128:], "prepared lifecycle settings preserve every trained object and payload")
    raw = inputs / "trained-checkpoint"; raw.write_bytes(prepared)
    packed = inputs / "trained.aotxccir"
    test.command([test.build / "aotx_recall_cli_fixture", raw, "-", packed, 1], "trained-memory")
    test.record(batch=count, fill=fill, pressure=pressure, trained_sequence=cut,
                trained_sha256=hashlib.sha256(stored).hexdigest())
    return f, packed, bytes(prepared), cut, fill, pressure, run


def package(test, prepared, image, architecture, pressure):
    sources = test.output / "policy-source"; sources.mkdir()
    local_image = sources / image.name; shutil.copyfile(image, local_image)
    provenance = sources / "provenance.txt"
    provenance.write_text("Native maintenance entry; image SHA256 " + hashlib.sha256(local_image.read_bytes()).hexdigest() + "\n")
    license_file = sources / "LICENSE"; shutil.copyfile(test.source / "LICENSE", license_file)
    policy = sources / "policy.bin"
    test.command([test.build / "aotx_policy_pack", "--output", policy, "--mode", "native",
        "--image", local_image, "--format", "cubin" if image.suffix == ".cubin" else "ptx",
        "--kernel", "aotx_creator_maintenance", "--architecture", architecture,
        "--state-schema", 1, "--state-bytes", 16, "--threads", 64, "--registers", 128,
        "--shared-bytes", 0, "--local-bytes", 0, "--pressure", pressure,
        "--minimum-move", 1, "--backoff", 1, "--provenance", provenance, "--license", license_file], "policy-pack")
    encoded = policy.read_bytes(); digest = hashlib.sha256(encoded).hexdigest()
    inspected = test.command([test.build / "aotx_policy_pack", "--inspect", policy], "policy-inspect")
    test.check(f"policy_digest={digest}\n" in inspected, "policy inspection reports the exact independently computed digest")
    store = test.output / "selected-models"; store.mkdir()
    entries = [json.loads(line) for line in (test.store / "manifest.jsonl").read_text().splitlines() if line.strip()]
    entries = [entry for entry in entries if entry.get("role") == "language"]
    test.check(len(entries) == 1, "one selected language model is packed")
    entry = dict(entries[0]); target = (test.store / entry["path"]).resolve()
    entry["path"] = "language.gguf"; (store / entry["path"]).symlink_to(target)
    (store / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
    runtime = test.output / "runtime.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", prepared, "--models", store,
        "--modules", test.output / "modules", "--settings", test.output / "settings", "--roles", "language",
        "--output", runtime, "--policy", policy,
        "--phrases", test.source / "tests/fixtures/quality/refusal-phrases.txt"], "runtime-pack")
    return runtime, encoded, digest, sources, store


def verify_asset(test, f, data, encoded):
    runtime = data[5]
    test.check(f.get(runtime, 8, 4) == 1 and f.get(runtime, 20, 4) & 16 and
               len(runtime) == 256 + f.get(runtime, 16, 4) * 384,
               "complete runtime has the policy feature and exact asset index")
    found = [runtime[at:at + 384] for at in range(256, len(runtime), 384)
             if runtime[at + 64:at + 320].split(b"\0", 1)[0] == b"policy.bin"]
    test.check(len(found) == 1 and f.get(found[0], 16, 4) == 3 and
               f.get(found[0], 24) == len(encoded) and found[0][32:64] == hashlib.sha256(encoded).digest(),
               "required policy asset has the exact original length and digest")


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def decisions(test, f, data, digest):
    events, active, revision = [], bytearray(), 0
    for part in replay_records(f, data[7], 38):
        total, offset, count = (f.get(part, at, 4) for at in (4, 8, 12))
        if not offset:
            test.check(not active and total == 272, "bounded policy decision starts at the exact state size")
            revision = f.get(part, 16)
        test.check(f.get(part, 0, 4) == 1 and total == 272 and offset == len(active) and
                   f.get(part, 16) == revision and not any(part[24:32]) and
                   len(part) == 32 + count and count == min(160, total - offset),
                   "policy fragments preserve exact revision, bounds, and order")
        active.extend(part[32:])
        if len(active) == total:
            test.check(active[:8] == b"AOTXPD01" and f.get(active, 8, 4) == 1 and
                       f.get(active, 12, 4) == 1 and f.get(active, 16, 4) == 16 and
                       f.get(active, 24) == len(events) + 1 == revision and active[32:64].hex() == digest,
                       "accepted policy event has the exact bundle and state revision")
            action = f.get(active, 192, 4)
            previous = f.get(events[-1], 264) if events else 0
            test.check(action in (0, 1) and f.get(active, 256) == revision and not f.get(active, 196, 4) and
                       f.get(active, 264) == (f.get(active, 64) if action else previous),
                       "native private counters preserve each accepted decision and maintenance source")
            events.append(bytes(active)); active.clear()
    test.check(events and not active, "complete saved policy decisions exist")
    return events


def observed(test, f, row, event):
    test.check(row["decision"] == f.get(event, 24) and row["source"] == f.get(event, 64) and
               row["state_hash"] == fnv(event[256:]), "console state matches every saved private byte")


def reclaimed(run, minimum):
    pattern = r"memory lifecycle: root (\d+) retry floor (\d+) automatic (\d+) removed (\d+) released bytes (\d+)"
    def received():
        start = len(run.console()); run.send("memory")
        row = wait(lambda: re.search(pattern, run.console()[start:]), run.child, 60)
        return row if int(row[3]) == 1 and int(row[4]) >= minimum and int(row[5]) > 0 else None
    row = wait(received, run.child, 300)
    run.test.check(True, "native policy causes actual automatic memory reclamation", lifecycle=row.group(0))


def exercise(test, count, image, architecture):
    f, prepared, trained, cut, fill, pressure, training = trained_memory(test, count)
    runtime, encoded, digest, sources, store = package(test, prepared, image, architecture, pressure)
    initial = sections(test, runtime); verify_asset(test, f, initial, encoded)
    test.check(initial[2] == trained, "complete package contains the exact trained memory and selected lifecycle settings")
    run = RuntimeRun(test, "before", runtime, ("--policy-trust", digest)); run.ready()
    test.check("network: IPv4 and IPv6 sockets are disabled" in run.path.read_text(), "native activation has no IP access")
    reclaimed(run, fill); first = settled(run, cut, 1); durable(run)
    before = sections(test, runtime); saved = decisions(test, f, before, digest); observed(test, f, first, saved[-1])
    test.check(any(f.get(event, 192, 4) == 1 for event in saved), "saved native proposal requests real maintenance")
    test.check(f.get(before[2], 20, 4) + fill == f.get(trained, 20, 4) and f.get(before[2], 96) == cut,
               "maintenance removes the exact eligible archive batch and publishes its root")
    originals = {row[8:24] + row[40:48]: (row, payload) for row, payload in f.state_rows(trained)}
    for row, payload in f.state_rows(before[2]):
        old, original = originals[row[8:24] + row[40:48]]
        test.check(row[:160] == old[:160] and row[168:] == old[168:] and payload == original,
                   "every retained trained object keeps its metadata and payload")
    paused = policy_status(run, "pause"); test.check(paused["state"] == "paused", "actual console pauses policy evaluation")
    spawn(run, count); bind(run, f, count, cut, test.output / "before-bind"); durable(run)
    cut = answer(run, f, count, cut, 1, 2, test.output / "before-input"); durable(run)
    blocked = policy_status(run)
    test.check(blocked["decision"] == paused["decision"] and blocked["calls"] == paused["calls"] and
               blocked["state_hash"] == paused["state_hash"] and cut > blocked["source"],
               "pause blocks new decisions while real foreground work changes memory")
    policy_status(run, "resume"); current = settled(run, cut, paused["decision"] + 1); durable(run)
    test.check(current["calls"] > paused["calls"], "console resume admits the changed observation")
    stopped = policy_status(run, "stop"); test.check(stopped["state"] == "stopped", "actual console stops policy evaluation")
    durable(run); run.stop()
    saved_file = sections(test, runtime); saved = decisions(test, f, saved_file, digest)
    observed(test, f, stopped, saved[-1]); verify_asset(test, f, saved_file, encoded)
    paths = [sources, store, test.output / "modules", test.output / "settings", test.output / "inputs",
             training.journal, run.journal, test.output / "before-bind", test.output / "before-input"]
    for path in paths:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    test.check(all(not path.exists() for path in paths), "original policy, module, input, store links, and journals are absent")
    after = RuntimeRun(test, "after", runtime, ("--policy-trust", digest)); after.ready()
    restored = policy_status(after)
    test.check(restored["calls"] == 0, "file-only recovery does not execute the native policy")
    observed(test, f, restored, saved[-1]); durable(after)
    restored_file = sections(test, runtime)
    test.check(decisions(test, f, restored_file, digest) == saved, "every policy decision and private state byte survives recovery")
    test.check(restored_file[2] == saved_file[2], "file-only recovery preserves exact maintained memory")
    verify_asset(test, f, restored_file, encoded)
    match = re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)", after.path.read_text())
    test.check(match and int(match[1]) == f.get(saved_file[7], 40) and int(match[2], 16) == f.get(saved_file[7], 32)
               and int(match[3]) == int(match[4]) == 0, "file-only restore has the exact record count and state hash")
    policy_status(after, "resume")
    cut = answer(after, f, count, cut, 2, 3, test.output / "after-input")
    final = settled(after, cut, restored["decision"] + 1); durable(after); policy_status(after, "stop"); after.stop()
    final_file = sections(test, runtime); final_events = decisions(test, f, final_file, digest)
    test.check(final_events[:len(saved)] == saved and len(final_events) > len(saved),
               "fresh work extends the exact recovered decision prefix")
    observed(test, f, final, final_events[-1])
    test.check(f.get(final_file[2], 32) == cut and all(not path.exists() for path in paths),
               "fresh private recall and retention need only the complete runtime file")


def main():
    parser = argparse.ArgumentParser(description="Check native policy packaging and complete file recovery.")
    for name in ("build", "source", "store", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("batch", type=int, choices=(1, 64))
    parser.add_argument("image", type=lambda value: Path(value).resolve())
    parser.add_argument("architecture", type=int)
    args = parser.parse_args()
    if args.architecture < 1 or not args.image.is_file() or args.image.suffix not in (".ptx", ".cubin"):
        parser.error("a native PTX or cubin image and a positive architecture are required")
    test = RuntimeTest(args.build, args.source, args.store, args.output, snapshot_every=128)
    begin, status = time.monotonic(), 0
    try:
        exercise(test, args.batch, args.image, args.architecture)
    except (Exception, KeyboardInterrupt) as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"policy runtime failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        test.flush_checks(); status |= any(not row["passed"] for row in test.checks)
        result = dict(batch=args.batch, checks=len(test.checks), failed=sum(not row["passed"] for row in test.checks),
                      seconds=time.monotonic() - begin, exit=int(status))
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
