#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check authenticated shared appraisal, useful recall and complete file recovery.
# Owns: One new output directory, one complete runtime file and its child processes.
# Threading: Concurrent common API clients; CUDA owns model and memory processing.
# Lifetime: Hosted room input, saved appraisal evidence and a file-only restart.

# Inputs: Build, source, store, output and batch. Output: Commands, source receipts and checks.
# Exit: 0 pass, 1 failed checks or cleanup, 2 bad arguments.
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from aiohttp import ClientSession, ClientTimeout

sys.dont_write_bytecode = True
from text_boot_test import setup
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections
from retain_boot_test import compiled_limits, transfers
from appraisal_runtime_test import control, process, status, restore_report, recovered_state
from appraisal_runtime_cases import evidence
from shared_runtime_test import aotx_configuration, aotx_load_config, aotx_write_grants
from shared_runtime_client import aotx_shared_client
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http
from appraisal_shared_cases import report, recall, targets, useful, wave, assessment, result_records, selections, exposed, events


def prepare(test, count):
    external = test.store; local = test.output / "selected-models"; local.mkdir()
    entries = [json.loads(line) for line in (external / "manifest.jsonl").read_text().splitlines() if line.strip()]
    entries = [dict(entry) for entry in entries if entry.get("role") in ("language", "embedding")]
    test.check(len(entries) == 2 and {entry["role"] for entry in entries} == {"language", "embedding"},
               "shared appraisal selects one actual language model and one embedding model")
    for entry in entries:
        path = (external / entry["path"]).resolve(); entry["path"] = entry["role"] + ".gguf"
        (local / entry["path"]).symlink_to(path)
    (local / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    test.store = local; f = setup(test); limits = compiled_limits(test)
    test.check(limits["objects"] >= count * 64 and limits["payload_bytes"] >= count * 32768,
               "configured memory fits the complete shared source and recall batches")
    raw, memory = test.output / "inputs/empty-checkpoint", test.output / "inputs/memory.aotxccir"
    raw.write_bytes(f.image([], 0, 10))
    test.command([test.build / "aotx_recall_cli_fixture", raw, "-", memory, 1], "empty-shared-memory")
    runtime = test.output / "runtime.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", memory, "--models", local,
        "--roles", "language,embedding", "--modules", test.output / "modules", "--settings", test.output / "settings",
        "--phrases", test.source / "tests/fixtures/quality/refusal-phrases.txt", "--shared", "--output", runtime], "pack-shared-runtime")
    initial = sections(test, runtime)
    test.check(bool(f.get(initial[5], 20, 4) & 8) and not f.get(initial[2], 20, 4),
               "shared runtime starts with empty cognitive memory and required shared support")
    originals = [local, test.output / "modules", test.output / "settings", test.output / "inputs"]
    for path in originals:
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
    test.check(all(not path.exists() for path in originals), "owned model paths, modules, settings and initial memory are removed")
    model_digest = bytes.fromhex(next(entry["sha256"] for entry in entries if entry["role"] == "language"))
    test.check(len(model_digest) == 32, "actual shared model identity has the declared digest width")
    test.record(complete_runtime=str(runtime), complete_copies=1, model_sha256=model_digest.hex())
    return f, runtime, model_digest, originals


def configuration(test, count, label, revision, pages):
    cfg, path, grants, keys = aotx_configuration(test, count, label, revision)
    cfg["models"] = {"text": {"role": "language", "published_at": 1}}
    for principal in cfg["principals"]:
        principal["models"] = ["text"]; principal["pages"] = pages
    path.write_text(json.dumps(cfg)); aotx_write_grants(aotx_load_config(path), grants)
    test.record(shared_grant_pages=pages, revision=revision, model_roles=["language"])
    return cfg, path, grants, keys


def records(test, run, f):
    limit = compiled_limits(test)["image_bytes"]
    return transfers(test, run, f, {14: 64 + 64 * 18672 + limit, 17: 64 + 64 * 8320 + limit})


async def conversation(client, space):
    value, _ = await client.done("/spaces/" + space + "/conversations")
    return "con-" + client.lineage + "-" + value["resource"]


async def initial(test, run, f, runtime, cfg, path, keys, args, model_digest):
    gateway = aotx_http_run(test, path, "initial"); url = "http://127.0.0.1:" + str(cfg["port"])
    try:
        async with ClientSession(timeout=ClientTimeout(total=args.work_seconds + 60)) as http:
            clients = [aotx_shared_client(test, http, url, key, "%032x" % (i + 1), args.work_seconds) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            await aotx_http(test, http, "GET", url + "/v1/models", {"Authorization": "Bearer invalid"}, 401)
            caps = await clients[0].request("GET", "/capabilities"); limits = caps["limits"]
            test.check(limits["participants"] >= len(clients) and limits["conversations"] >= 2 * args.batch and
                       limits["members"] >= args.batch - 1 and limits["receipts"] >= len(clients) + 6 * args.batch,
                       "shared runtime capacity fits all distinct callers, conversations and exact receipts", limits=limits)
            for client in clients:
                person = await client.discover()
                if not person["registered"]: await client.done("/participant")
            created, _ = await clients[0].done("/spaces", scope="room")
            space = "spc-" + clients[0].lineage + "-" + created["resource"]
            for client in clients[1:args.batch]:
                await clients[0].done("/spaces/" + space + "/members", participant=client.participant, permissions=["read", "write"])
            outsider = clients[args.batch]
            await outsider.request("GET", "/spaces/" + space, (404,))
            await outsider.request("GET", "/spaces/" + space + "/memory", (404,))
            conversations = await asyncio.gather(*(conversation(client, space) for client in clients[:args.batch]))
            forged = clients[0].command(text=report(0), model="text", actor=outsider.participant)
            sequence = clients[0].sequence
            await clients[0].request("POST", "/conversations/" + conversations[0] + "/inputs", (400,), json=forged)
            test.check(int((await clients[0].discover())["next_sequence"]) == sequence,
                       "a caller cannot supply another source actor or consume a sequence through an invalid field")
            control(run, "on", 3); control(run, f"limits {args.pages} {args.tokens} {args.ticks} {args.batch}")
            control(run, "priority 0 1000000"); control(run, "background off", 3); durable(run)
            inputs = await wave(test, clients, conversations, space, report, 1, args.pages)
            await events(test, clients, [[entry] for entry in inputs]); durable(run)
            before = status(run)
            test.check(not before["calls"] and before["pending"] == args.batch and not before["active"],
                       "shared input queues appraisals while background generation stays off", appraisal=before)
            from appraisal_shared_checkpoint import save_pending
            save_pending(test, args, runtime, space, inputs, model_digest, sections(test, runtime)[2])
            process(run, args.batch, args.work_seconds); durable(run)
            state = sections(test, runtime)[2]
            accepted = assessment(test, f, state, inputs, space, model_digest)
            known = evidence(f, state); counters = status(run)
            actual_records = records(test, run, f)
            result_hashes = result_records(test, f, actual_records, state, accepted, model_digest)
            if args.assessment_only:
                saved = dict(batch=args.batch, space=space, inputs=inputs, accepted=accepted,
                             model_sha256=model_digest.hex(), result_sha256=result_hashes,
                             memory_sha256=hashlib.sha256(state).hexdigest())
                (test.output / "assessment-input.json").write_text(json.dumps(saved, indent=2) + "\n")
                return None
            selections(test, f, actual_records, state, inputs, accepted, False)
            await events(test, clients, [[entry] for entry in inputs])
            await exposed(test, f, clients[min(1, args.batch - 1)], space, state, accepted)
            await outsider.request("GET", "/operations/" + inputs[0]["receipt"]["id"], (404,))
            control(run, "writes off", 2); durable(run)
            reads = await asyncio.gather(*(conversation(client, space) for client in clients[:args.batch]))
            requested = targets(args.batch, 1)
            recalled = await wave(test, clients, reads, space, lambda i: recall(requested[i], 1), 1, args.pages, requested)
            durable(run); saved = sections(test, runtime)
            selections(test, f, records(test, run, f), saved[2], recalled, accepted, True); useful(test, recalled)
            test.check(evidence(f, saved[2]) == known and status(run)["calls"] == counters["calls"] and not status(run)["pending"],
                       "read-only shared recall adds neither model appraisal nor exposure")
            history = [[inputs[i], recalled[i]] for i in range(args.batch)]
            await events(test, clients, history)
            retained = dict(space=space, inputs=inputs, recalled=recalled, conversations=reads, accepted=accepted,
                result_sha256=result_hashes, actors=[client.participant for client in clients], source_count=args.batch)
            (test.output / "retained.json").write_text(json.dumps(retained, indent=2) + "\n")
            return retained, saved, known
    finally:
        gateway.close()


async def recovered(test, run, f, runtime, cfg, path, keys, old_keys, args, retained, before, known, model_digest):
    gateway = aotx_http_run(test, path, "recovered"); url = "http://127.0.0.1:" + str(cfg["port"])
    try:
        async with ClientSession(timeout=ClientTimeout(total=args.work_seconds + 60)) as http:
            clients = [aotx_shared_client(test, http, url, key, "%032x" % (i + 1), args.work_seconds) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            await aotx_http(test, http, "GET", url + "/v1/models", {"Authorization": "Bearer " + old_keys[0]}, 401)
            for client in clients:
                test.check((await client.discover())["registered"], "file recovery restores registered participants with new deployment credentials")
            history = [[retained["inputs"][i], retained["recalled"][i]] for i in range(args.batch)]
            for i, group in enumerate(history):
                for entry in group:
                    old = entry["receipt"]; actual = await clients[i].terminal(old["id"])
                    test.check(all(actual[k] == old[k] for k in ("id", "actor", "sequence", "input_order", "state", "status", "usage", "finish", "exact_output")),
                               "file-only shared recovery preserves exact authenticated model results")
                    retry = await clients[i].request("POST", "/conversations/" + entry["conversation"] + "/inputs", json=entry["body"])
                    test.check(retry["id"] == old["id"] and retry["output"]["base64"] == old["exact_output"],
                               "recovered canonical shared retry cannot repeat model inference")
            await events(test, clients, history)
            await exposed(test, f, clients[0], retained["space"], before[2], retained["accepted"])
            await clients[args.batch].request("GET", "/spaces/" + retained["space"] + "/memory", (404,))
            await clients[args.batch].request("GET", "/operations/" + retained["inputs"][0]["receipt"]["id"], (404,))
            durable(run); restored = sections(test, runtime)
            test.check(restored[2] == before[2] and evidence(f, restored[2]) == known and status(run)["calls"] == 0,
                       "saved shared result reads and retries preserve exact memory without appraisal generation")
            retained_hashes = result_records(test, f, records(test, run, f), restored[2], retained["accepted"], model_digest)
            test.check(retained_hashes == retained["result_sha256"], "file-only recovery preserves every actual shared appraisal result byte")
            requested = targets(args.batch, 2)
            continued = await wave(test, clients, retained["conversations"], retained["space"],
                                   lambda i: recall(requested[i], 2), 2, args.pages, requested)
            durable(run); final = sections(test, runtime)
            selections(test, f, records(test, run, f), final[2], continued, retained["accepted"], True); useful(test, continued)
            for i, entry in enumerate(continued): history[i].append(entry)
            await events(test, clients, history)
            test.check(evidence(f, final[2]) == known and status(run)["calls"] == 0 and not status(run)["pending"],
                       "continued file-only shared recall preserves the original external exposures")
    finally:
        gateway.close()


def exercise(test, args):
    if args.resume_pending:
        from appraisal_shared_pending import exercise as resume
        return resume(test, args)
    if args.resume_assessment or args.resume_recall:
        from appraisal_shared_resume import exercise as resume
        return resume(test, args)
    f, runtime, model_digest, originals = prepare(test, args.batch)
    cfg, path, grants, keys = configuration(test, args.batch, "initial", 1, args.pages)
    run = RuntimeRun(test, "initial", runtime, extra=("--service-grants", grants)); run.ready(args.ready_seconds)
    test.check("network: IPv4 and IPv6 sockets are disabled" in run.path.read_text(),
               "GPU runtime uses the standard local gateway seam with network clients outside it")
    result = asyncio.run(initial(test, run, f, runtime, cfg, path, keys, args, model_digest))
    if args.assessment_only:
        durable(run); run.stop(); return
    retained, saved, known = result
    durable(run); run.stop(killed=True); saved = sections(test, runtime)
    shutil.rmtree(run.journal)
    test.check(not run.journal.exists() and all(not path.exists() for path in originals),
               "complete shared recovery has no prior journal or owned model source paths")
    cfg, path, grants, new_keys = configuration(test, args.batch, "recovered", 2, args.pages)
    after = RuntimeRun(test, "recovered", runtime, extra=("--service-grants", grants)); after.ready(args.ready_seconds)
    restore_report(test, after, f.get(saved[7], 40), f.get(saved[7], 32)); durable(after)
    restored = sections(test, runtime)
    recovered_state(test, f, saved, restored)
    test.check(status(after)["flags"] == 2 and not status(after)["calls"],
               "complete hosted mirror restores read-only appraisal settings without new generation")
    asyncio.run(recovered(test, after, f, runtime, cfg, path, new_keys, keys, args, retained, restored, known, model_digest))
    durable(after); after.stop()
    test.check(all(not path.exists() for path in originals) and not run.journal.exists(),
               "continued shared operation needs only the single complete runtime and fresh deployment grants")


def main():
    parser = argparse.ArgumentParser(description="Check actual shared model appraisal and file-only recovery.")
    for name in ("build", "source", "store", "output"): parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("batch", type=int, choices=(1, 64))
    parser.add_argument("--assessment-only", action="store_true",
                        help="Stop after actual appraisal and recorded-result checks.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume-assessment", type=lambda value: Path(value).resolve(),
                      help="Use a saved successful appraisal case for recall and recovery checks.")
    mode.add_argument("--resume-pending", type=lambda value: Path(value).resolve(),
                      help="Use saved pending inputs and stop after their appraisal checks.")
    mode.add_argument("--resume-recall", type=lambda value: Path(value).resolve(),
                      help="Use a completed recall checkpoint for file recovery checks.")
    parser.add_argument("--pages", type=int, default=160)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--ticks", type=int, default=16384)
    parser.add_argument("--ready-seconds", type=int, default=900)
    parser.add_argument("--work-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.resume_pending and not args.assessment_only:
        parser.error("Pending input recovery requires --assessment-only.")
    if args.assessment_only and (args.resume_assessment or args.resume_recall):
        parser.error("Completed stage recovery cannot use --assessment-only.")
    if min(args.pages, args.tokens, args.ticks, args.ready_seconds, args.work_seconds) < 1:
        parser.error("Resource counts and caller time limits must be positive.")
    if len(str(args.output / "recovered-journal/service.sock").encode()) >= 108:
        parser.error("The output path exceeds the local socket path limit.")
    test = RuntimeTest(args.build, args.source, args.store, args.output, snapshot_every=1024)
    begin, code = time.monotonic(), 0
    try:
        exercise(test, args)
    except (Exception, KeyboardInterrupt) as error:
        code = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"shared appraisal failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: code = 1; test.record(cleanup_error=str(error))
        test.flush_checks(); code |= any(not row["passed"] for row in test.checks)
    result = dict(batch=args.batch, assessment_only=args.assessment_only, resumed_assessment=bool(args.resume_assessment),
                  resumed_pending=bool(args.resume_pending), resumed_recall=bool(args.resume_recall),
                  checks=len(test.checks), failed=sum(not row["passed"] for row in test.checks),
                  seconds=time.monotonic() - begin, exit=int(code))
    (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__": sys.exit(main())
