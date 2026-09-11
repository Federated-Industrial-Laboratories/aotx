#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check native image replies and complete runtime recovery without source files.
# Owns: New input copies, runtime files, journals and only the processes started here.
# Threading: One disk driver; CUDA admits, decodes and encodes all images and conversations.
# Lifetime: Initial activation and two fresh recovery processes.

# Inputs: build, source, model store, image, new output, count 1 or 64.
# Output: source identities, commands and checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import wait, rows
from checkpoint_boot_test import spawn
from capacity_boot_test import batch
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections, replay_records


def prepare(test, count):
    shutil.copytree(test.source / "modules/roles", test.output / "modules")
    (test.output / "settings").write_text("sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 64\n"
        "decode.think_limit = 0\ntools.mask = 0\nagent.recall_k = 0\nderive.list = console,bus,transcript,tokens,pages\n")
    spec = importlib.util.spec_from_file_location("image_bytes", test.source / "tests/recall_cli_test.py")
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    checkpoint = test.output / "checkpoint"
    checkpoint.write_bytes(f.image([], 0, 10))
    memory = test.output / "memory.aotxccir"
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", memory, 1], "empty-memory")
    store = test.output / "store"; store.mkdir()
    models, vision = rows(test.store / "manifest.jsonl"), rows(test.store / "vision.jsonl")
    for entry in models + vision:
        source = (test.store / entry["path"]).resolve()
        target = store / Path(entry["path"]).name
        if not target.exists(): target.symlink_to(source)
        entry["path"] = target.name
    (store / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in models))
    (store / "vision.jsonl").write_text("".join(json.dumps(e) + "\n" for e in vision))
    for name in ("media.profile", "LICENSE-Apache-2.0.txt"):
        if (test.store / name).is_file(): shutil.copy2(test.store / name, store / name)
    runtime = test.output / "runtime.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", memory, "--models", store,
        "--roles", "language,embedding" if count > 1 else "language", "--modules", test.output / "modules", "--settings", test.output / "settings",
        "--phrases", test.source / "tests/fixtures/quality/refusal-phrases.txt", "--output", runtime], "pack-image-runtime")
    initial = sections(test, runtime)
    test.check(bool(f.get(initial[5], 20, 4) & 2), "runtime requires its image component")
    shutil.rmtree(store); shutil.rmtree(test.output / "modules")
    (test.output / "settings").unlink(); checkpoint.unlink(); memory.unlink()
    return f, runtime


def source_files(test, image, count):
    result = []
    colors = ((220, 12, 12, "red"), (12, 24, 220, "blue"), (12, 190, 24, "green"), (240, 220, 12, "yellow"))
    for i in range(count):
        path = test.output / f"input-{i}"
        if count == 1:
            shutil.copyfile(image, path); kind, expected = "jpeg", "bird"
            header = b""
        else:
            width, height = 256 + i % 8, 256 + i // 8
            red, green, blue, expected = colors[i % len(colors)]
            data = bytearray((red, green, blue)) * (width * height)
            top = 24 if i % 2 else height - 56
            for y in range(top, top + 32):
                for x in range(24 + i % 8, 56 + i % 8):
                    at = (y * width + x) * 3; data[at:at + 3] = b"\0\0\0"
            path.write_bytes(data); kind = f"rgb8 {width} {height}"
            header = b"AOTXRGB1" + struct.pack("<IIQ", width, height, len(data))
        digest = hashlib.sha256(header + path.read_bytes()).hexdigest()
        result.append(dict(path=str(path), kind=kind, digest=digest, bytes=path.stat().st_size + len(header), expected=expected))
    test.check(len({r["digest"] for r in result}) == count, "the source batch has distinct immutable identities")
    (test.output / "sources.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def answer(run, slot, turn):
    wait(lambda: any(r.get("agent") == slot and r.get("turn") == turn for r in run.turns()), run.child, 300)
    value = wait(lambda: next((r for r in run.events(slot) if r.get("kind") == "reply" and
        r.get("turn") == turn), None), run.child, 300)
    run.test.check(bool(value.get("text", "").strip()), "native image request has a visible reply", slot=slot, turn=turn, reply=value)
    run.test.check(not any(r.get("status") == "prompt_refused" for r in run.events(slot)), "native image prompt fits its profile", slot=slot)
    return value["text"].lower()


def ask(run, sources, turn, f, cut):
    request = batch(f, b"AOTXTXT1", 8256, cut, len(sources)) if len(sources) > 1 else None
    for i, source in enumerate(sources):
        question = "What is the main background color? Reply with one color word."
        if len(sources) == 1:
            question = ("Describe the animal in this image briefly." if turn == 1 else
                        "Which direction does the animal face in the image?" if turn == 2 else
                        "Describe the visible markings on the animal.")
        line = f"[image:{source['digest']}] {question}"
        if request is None:
            run.send(f"say {line}")
        else:
            at = 64 + i * 8256; f.put(request, at, i, 4)
            request[at + 16:at + 32] = f.identity(8000 + i); f.put(request, at + 32, turn)
            q = at + 64; text = line.encode()
            request[q:q + 16], request[q + 16:q + 32], request[q + 48:q + 64] = (
                f.identity(300000 + turn * 64 + i), f.identity(10000 + i), f.identity(400000 + turn * 64 + i))
            for offset, value in ((132, 1), (136, 512), (148, len(text))): f.put(request, q + offset, value, 4)
            request[q + 4640:q + 4640 + len(text)] = text
    if request is not None:
        path = run.test.output / "request"; path.write_bytes(request)
        run.operation("text", path, 6, len(sources)); path.unlink()
    for i, source in enumerate(sources):
        value = answer(run, i, turn)
        if len(sources) == 1:
            wanted = r"\b(bird|starling)\b" if turn == 1 else r"\bright\b" if turn == 2 else r"\b(spot|speck|white|pattern|mark)"
        else: wanted = rf"\b{source['expected']}\b"
        run.test.check(bool(re.search(wanted, value)), "reply identifies visible image content", kind="behavior", slot=i, turn=turn, reply=value)


def exercise(test, image, count):
    f, runtime = prepare(test, count)
    sources = source_files(test, image, count)
    run = RuntimeRun(test, "initial", runtime); run.ready(); spawn(run, count)
    if count > 1:
        bindings = batch(f, b"AOTXBND1", 64, 0, count)
        for i in range(count):
            at = 64 + i * 64; f.put(bindings, at, i, 4)
            bindings[at + 8:at + 24], bindings[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
            f.put(bindings, at + 56, 160, 4); f.put(bindings, at + 60, 1, 4)
        path = test.output / "bindings"; path.write_bytes(bindings)
        run.operation("bind", path, 3, count); path.unlink()
    for i, source in enumerate(sources):
        before = run.path.read_text().count("image: uploaded")
        run.send(f"image load {i} private {source['kind']} {source['path']}")
        wait(lambda: run.path.read_text().count("image: uploaded") > before, run.child, 180)
        Path(source["path"]).unlink()
    ask(run, sources, 1, f, 0); durable(run); run.stop()
    saved = sections(test, runtime)
    canonical = replay_records(f, saved[7], 35)
    by_id = {}
    contiguous = True
    for p in canonical:
        op, transfer = f.get(p, 4, 4), p[8:24].hex()
        if op == 1:
            by_id[transfer] = dict(digest=p[96:128].hex(), bytes=f.get(p, 24), data=bytearray())
        elif op == 2:
            row = by_id[transfer]
            contiguous &= f.get(p, 32) == len(row["data"])
            row["data"].extend(p[40:])
    test.check(len(by_id) == count, "all source identities occur in the complete file")
    test.check(contiguous, "retained source chunks are contiguous")
    for row in by_id.values():
        test.check(len(row["data"]) == row["bytes"] and hashlib.sha256(row["data"]).hexdigest() == row["digest"],
                   "complete file retains exact image bytes and their identity")
    for turn in (2, 3):
        copied = test.output / f"recovery-{turn}.aotxccir"
        test.command([test.build / "aotx_ccir", "compact", runtime, copied], "copy-image-runtime")
        runtime.unlink(); shutil.rmtree(run.journal)
        run = RuntimeRun(test, f"recovered-{turn}", copied); run.ready()
        match = re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)", run.path.read_text())
        test.check(match and int(match[1]) == f.get(saved[7], 40) and int(match[2], 16) == f.get(saved[7], 32)
                   and not int(match[3]) and not int(match[4]), "image runtime restores its exact count and hash")
        test.check("network: IPv4 and IPv6 sockets are disabled" in run.path.read_text(), "image recovery uses no network")
        test.check(all(not Path(s["path"]).exists() for s in sources), "new image questions have no source files")
        ask(run, sources, turn, f, f.get(saved[2], 32)); durable(run); run.stop()
        saved = sections(test, copied)
        test.check(replay_records(f, saved[7], 35) == canonical, "repeated recovery preserves all source records exactly")
        runtime = copied


def main():
    if len(sys.argv) != 7 or sys.argv[6] not in ("1", "64"):
        print("usage: image_runtime_test.py BUILD SOURCE STORE IMAGE OUTPUT 1|64", file=sys.stderr); return 2
    build, source, store, image, output = (Path(v).resolve() for v in sys.argv[1:6])
    test = RuntimeTest(build, source, store, output, snapshot_every=512)
    begin, status = time.monotonic(), 0
    try: exercise(test, image, int(sys.argv[6]))
    except Exception as error:
        status = 1; (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"image runtime failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try: run.close()
            except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    status |= any(not check["passed"] for check in test.checks)
    result = dict(status=status, checks=len(test.checks), seconds=time.monotonic() - begin)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__": sys.exit(main())
