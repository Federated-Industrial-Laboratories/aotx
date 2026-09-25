#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check source batches, creator decisions and interrupted background results.
# Owns: Bounded disk observations and exact independent journal expectations.
# Threading: One disk driver; CUDA owns all appraisal and decoder state.
# Lifetime: One actual runtime and its ordinary journal recovery.

# Inputs: Runtime, fixture module and source batch. Output: checks and observations.
# Exit: Caller exceptions on malformed bytes or an expired caller deadline.
import hashlib
import re
import shutil
import time

from appraisal_result_fixture import decode_result, decode_evidence
from live_boot_test import wait
from appraisal_runtime_cases import rows, current, source_id, response, PROCESSOR
from policy_runtime_test import fnv


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def left(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The caller deadline expired.")
        return remaining

    def until(self, run, predicate):
        return wait(predicate, run.child, self.left())


def console(run, command, pattern, deadline):
    start = len(run.console()); run.send(command)
    return deadline.until(run, lambda: re.search(pattern, run.console()[start:]))


def appraisal(run, deadline):
    pattern = (r"appraisal: flags (\d+) pending (\d+) active (\d+) status (\d+).*?"
               r"appraisal work: calls (\d+) completed (\d+) refused (\d+) interrupted (\d+)")
    match = console(run, "appraisal status", "(?s)" + pattern, deadline)
    return dict(zip(("flags", "pending", "active", "status", "calls", "completed", "refused", "interrupted"), map(int, match.groups())))


def policy(run, mode, deadline):
    pattern = (r"(?s)policy: (\w+) mode (\d+) decision (\d+) memory source (\d+) state bytes (\d+) status (\d+).*?"
               r"policy: calls (\d+) last ns (\d+) maximum ns (\d+) state hash (\d+) saved generation (\d+).*?"
               r"policy: abi 2 appraisal pending (\d+) work revision (\d+) accepted revision (\d+)")
    match = console(run, "policy status", pattern, deadline)
    result = dict(zip(("mode", "decision", "source", "bytes", "status", "calls", "last_ns", "maximum_ns", "hash", "generation",
                       "pending", "revision", "accepted"), map(int, match.groups()[1:])))
    result["state"] = match[1]
    run.test.check(result["mode"] == (1 if mode == "supplied" else 3) and result["bytes"] == 16 and
                   not result["status"] and result["state"] != "error", "the exact selected ABI 2 creator remains valid", policy=result)
    return result


def pages(run, deadline):
    match = console(run, "memory", r"memory: pages free (\d+) of (\d+)", deadline)
    return tuple(map(int, match.groups()))


def released(run, deadline):
    def complete():
        value = pages(run, deadline)
        return value if value[0] == value[1] else None
    value = deadline.until(run, complete)
    run.test.check(value[1] > 0, "all physical cache pages return to the free pool", pages=value)
    return value


def replies(run, count, ordinal, expected, deadline):
    for slot in range(count):
        deadline.until(run, lambda: any(r.get("agent") == slot and r.get("turn") == ordinal for r in run.turns()))
        found = deadline.until(run, lambda: next((r for r in run.events(slot) if r.get("kind") == "reply" and r.get("turn") == ordinal), None))
        run.test.check(re.fullmatch(r"\W*" + re.escape(expected) + r"\W*", found.get("text", "").strip().lower()) is not None,
                       "each foreground slot produces the requested reply", kind="behavior", slot=slot, ordinal=ordinal, reply=found)
        run.test.check(len([r for r in run.events(slot) if r.get("kind") == "reply" and r.get("turn") == ordinal]) == 1 and
                       not any(r.get("status") == "prompt_refused" for r in run.events(slot)),
                       "foreground input has one admitted reply and no leaked internal turn", slot=slot)


def package(test, args):
    folder = test.output / "creator"; folder.mkdir()
    provenance = folder / "provenance.txt"; license_file = folder / "LICENSE"
    shutil.copyfile(test.source / "LICENSE", license_file)
    provenance.write_text("ABI 2 background appraisal acceptance.\n")
    bundle = folder / "policy.bin"
    command = [test.build / "aotx_policy_pack", "--output", bundle, "--mode", args.mode, "--abi", 2,
               "--provenance", provenance, "--license", license_file]
    if args.mode == "native":
        image = folder / "maintenance.ptx"; shutil.copyfile(args.native_image, image)
        provenance.write_text("Public maintenance entry; image SHA256 " + hashlib.sha256(image.read_bytes()).hexdigest() + "\n")
        command += ["--image", image, "--format", "ptx", "--kernel", "aotx_creator_maintenance", "--architecture", args.architecture,
                    "--state-schema", 1, "--state-bytes", 16, "--threads", 64, "--registers", 128, "--shared-bytes", 0, "--local-bytes", 0]
    test.command(command, "policy-pack")
    raw = bundle.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    inspected = test.command([test.build / "aotx_policy_pack", "--inspect", bundle], "policy-inspect")
    test.check("abi=2\n" in inspected and f"mode={1 if args.mode == 'supplied' else 3}\n" in inspected and
               f"policy_digest={digest}\n" in inspected, "policy packing preserves the explicit ABI, mode and trusted digest")
    return bundle, digest


def records(test, run, f, image_limit):
    text = test.command([test.build / "aotx_journal", "records", run.journal, "--boot", run.boot], "background-records")
    active, result = {}, []
    for line in text.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields.get("class") != "1" or fields.get("type") not in ("33", "38"):
            continue
        p = bytes.fromhex(fields["body"]); kind = int(fields["type"])
        test.check(32 < len(p) <= 192 and len(p) == int(fields["body_len"]), "journal fragment has exact physical bounds")
        if kind == 33:
            op, total, offset = (f.get(p, at, 4) for at in (4, 24, 28))
            if op not in (16, 17):
                continue
            key = kind, p[8:24].hex(), op
            limit = 64 + 64 * (32 if op == 16 else 8320) + (0 if op == 16 else image_limit)
        else:
            total, offset, count = (f.get(p, at, 4) for at in (4, 8, 12))
            key, op, limit = (kind, f.get(p, 16)), 38, 272
            test.check(count == len(p) - 32 and not any(p[24:32]), "policy fragment declares its exact data extent")
        if not offset:
            test.check(key not in active and 64 <= total <= limit, "a bounded complete journal operation begins once")
            active[key] = dict(data=bytearray(), tick=int(fields["tick"]), first=int(fields["seq"]), total=total)
        state = active.get(key)
        test.check(state is not None and f.get(p, 0, 4) == 1 and offset == len(state["data"]) and
                   total == state["total"] and len(p) - 32 == min(160, total - offset), "fragments preserve exact order and operation identity")
        state["data"].extend(p[32:])
        if len(state["data"]) == total:
            result.append(dict(op=op, identity=key[1], data=bytes(state["data"]), tick=state["tick"],
                               end_tick=int(fields["tick"]), first=state["first"], last=int(fields["seq"])))
            del active[key]
    test.check(not active, "all saved policy and appraisal operations are complete")
    return result


def verify(test, f, state, recorded, count, digest, model):
    decisions = [r for r in recorded if r["op"] == 38]
    requests = [r for r in recorded if r["op"] == 16]
    results = [r for r in recorded if r["op"] == 17]
    test.check(len(requests) == len(results) == 3 and len(decisions) == 3, "each nonempty source batch causes one creator decision and appraisal")
    stored = rows(f, state)
    for ordinal, (decision, request, result) in enumerate(zip(decisions, requests, results), 1):
        p, q, r = decision["data"], request["data"], result["data"]
        error = 11 if ordinal == 2 else 0
        test.check(p[:8] == b"AOTXPD01" and f.get(p, 8, 4) == f.get(p, 12, 4) == 1 and f.get(p, 16, 4) == 16 and
                   f.get(p, 20, 4) == 2 and f.get(p, 24) == ordinal and p[32:64].hex() == digest and
                   f.get(p, 164, 4) == 2 and f.get(p, 180, 4) == count and f.get(p, 192, 4) == 2 and not f.get(p, 196, 4),
                   "an exact accepted ABI 2 decision selects each admitted pending source batch")
        test.check(f.get(p, 256) == ordinal and not f.get(p, 264) and f.get(p, 64) == f.get(q, 16),
                   "creator state counts evaluation without recording a maintenance attempt")
        test.check(q[:8] == b"AOTXAPR1" and f.get(q, 8, 4) == 1 and f.get(q, 12, 4) == count and f.get(q, 48, 4) == 1 and
                   request["identity"] == result["identity"] and decision["last"] < request["first"] < result["first"],
                   "recorded background work follows its complete creator decision")
        frame = decode_result(r)
        test.check(len(frame["rows"]) == count and frame["sequence"] == f.get(q, 16) and frame["status"] == error,
                   "complete result records the exact batch status and source cut")
        tail = frame["tail"]
        test.check(f.get(tail, 20, 4) == count * (1 if error else 3), "interrupted work can publish only queue versions")
        seen = set()
        for i, row in enumerate(frame["rows"]):
            entry = q[64 + i * 32:96 + i * 32]
            key = bytes(entry[:16]), f.get(entry, 16)
            test.check(key not in seen and key in stored and (row["queue"], row["version"]) == key and row["model"] == model,
                       "every decoder lease has an exact independent queue, status and selected model")
            seen.add(key)
            queue = stored[key][0]
            test.check(queue[96:112] == source_id(f, i, ordinal) and queue[120:136] == f.identity(10000 + i),
                       "background sources retain their distinct admitted actors")
            if not error:
                response(test, row["reply"])
                if frame["version"] == 2:
                    source = stored[(bytes(queue[96:112]), f.get(queue, 112))][1][32:]
                    quotes = decode_evidence(row["first"], source)
                    test.record(appraisal_evidence=quotes, first_model_sha256=row["first_model"].hex())
    queues = current(f, state, b"AOTXAPQ1")
    test.check(len(queues) == count * 3, "each ordinary foreground source retains its own attributed appraisal queue")
    for ordinal in (1, 2, 3):
        for i in range(count):
            sid = source_id(f, i, ordinal)
            selected = [(row, payload) for row, payload in queues if row[96:112] == sid]
            inferred = [(row, payload) for row, payload in stored.values() if row[96:112] == sid and f.get(row, 2, 2) in (3, 4)]
            test.check(len(selected) == 1 and f.get(selected[0][1], 12, 4) == (3 if ordinal == 2 else 1) and
                       len(inferred) == (0 if ordinal == 2 else 2), "one source gives one complete exposure or an interruption with no interpretation")
            if ordinal != 2:
                relation = next(payload for _, payload in inferred if payload[:8] == b"AOTXREL1")
                test.check(f.get(relation, 12, 4) == 1 and relation[72:104] == PROCESSOR and relation[104:136] == model,
                           "background completion preserves exactly one exposure and exact processor provenance")
    test.record(background_operations=[{k: v for k, v in item.items() if k != "data"} for item in recorded],
                decision_hashes=[fnv(item["data"][256:]) for item in decisions])
    return [item["data"] for item in recorded]
