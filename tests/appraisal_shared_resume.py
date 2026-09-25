#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Continue shared recall and recovery from an exact successful appraisal case.
# Inputs: Existing case metadata and complete runtime. Outputs: New copies, checks and receipts.
# Exit: Caller exceptions; no prior runtime file or case record is changed.
import asyncio
import hashlib
import json
from aiohttp import ClientSession, ClientTimeout
import recall_cli_test as f
from appraisal_runtime_cases import evidence
from appraisal_runtime_test import control, status, restore_report, recovered_state
from appraisal_shared_cases import exposed, events, targets, recall, wave, selections, useful, result_records
from gateway_runtime_test import aotx_http_run, aotx_ready
from runtime_boot_test import RuntimeRun, durable, sections
from shared_runtime_client import aotx_shared_client


async def first(test, run, cfg, path, keys, args, old, before):
    from appraisal_shared_test import conversation, records
    gateway = aotx_http_run(test, path, "continued"); url = "http://127.0.0.1:" + str(cfg["port"])
    try:
        async with ClientSession(timeout=ClientTimeout(total=args.work_seconds + 60)) as http:
            clients = [aotx_shared_client(test, http, url, key, "%032x" % (i + 1), args.work_seconds) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            for client in clients:
                test.check((await client.discover())["registered"], "copied appraisal case retains registered participants")
            for i, entry in enumerate(old["inputs"]):
                actual = await clients[i].terminal(entry["receipt"]["id"])
                test.check(all(actual[k] == entry["receipt"][k] for k in
                    ("id", "actor", "sequence", "input_order", "state", "status", "usage", "finish", "exact_output")),
                    "copied appraisal case retains the exact original response")
            await events(test, clients, [[entry] for entry in old["inputs"]])
            await exposed(test, f, clients[0], old["space"], before[2], old["accepted"])
            hashes = result_records(test, f, records(test, run, f), before[2], old["accepted"], bytes.fromhex(old["model_sha256"]))
            test.check(hashes == old["result_sha256"], "copied appraisal result bytes equal the accepted case")
            control(run, "writes off", 2); durable(run)
            conversations = await asyncio.gather(*(conversation(client, old["space"]) for client in clients[:args.batch]))
            requested = targets(args.batch, 1)
            recalled = await wave(test, clients, conversations, old["space"], lambda i: recall(requested[i], 1),
                                  1, args.pages, requested)
            durable(run); saved = sections(test, run.runtime_file)
            selections(test, f, records(test, run, f), saved[2], recalled, old["accepted"], True); useful(test, recalled)
            test.check(evidence(f, saved[2]) == evidence(f, before[2]) and not status(run)["calls"] and not status(run)["pending"],
                       "continued recall preserves accepted evidence without another appraisal")
            retained = dict(old, conversations=conversations, recalled=recalled,
                            actors=[client.participant for client in clients], source_count=args.batch)
            return retained, saved
    finally:
        gateway.close()


def exercise(test, args):
    from appraisal_shared_test import configuration, recovered
    from appraisal_shared_checkpoint import save_recall, validate_recall
    if args.resume_recall:
        prior = args.resume_recall
        state = sections(test, prior / "runtime.aotxccir")
        validate_recall(test, prior, state[2])
        retained = json.loads((prior / "retained.json").read_text())
        test.check(retained["source_count"] == args.batch, "recall checkpoint has the requested source batch")
        _, _, _, keys = configuration(test, args.batch, "prior", 2, args.pages)
        return complete(test, args, prior / "runtime.aotxccir", retained, keys)
    prior = args.resume_assessment
    result = json.loads((prior / "result.json").read_text())
    old = json.loads((prior / "assessment-input.json").read_text())
    test.check(result["exit"] == 0 and result["failed"] == 0 and result["assessment_only"] and
               result["batch"] == old["batch"] == args.batch and len(old["inputs"]) == len(old["accepted"]) == args.batch,
               "continuation requires a successful complete appraisal batch")
    original = prior / "runtime.aotxccir"; runtime = test.output / "runtime.aotxccir"
    before = sections(test, original)
    test.check(hashlib.sha256(before[2]).hexdigest() == old["memory_sha256"],
               "saved appraisal metadata names the exact complete memory state")
    test.record(resumed_assessment=str(prior), metadata_sha256=hashlib.sha256(
        (prior / "assessment-input.json").read_bytes()).hexdigest())
    test.command([test.build / "aotx_ccir", "compact", original, runtime], "copy-accepted-appraisal")
    test.check(not (test.output / "initial-journal").exists(), "continuation starts without the original journal")
    cfg, path, grants, keys = configuration(test, args.batch, "initial", 2, args.pages)
    run = RuntimeRun(test, "initial", runtime, extra=("--service-grants", grants)); run.runtime_file = runtime
    run.ready(args.ready_seconds); restore_report(test, run, f.get(before[7], 40), f.get(before[7], 32)); durable(run)
    restored = sections(test, runtime); recovered_state(test, f, before, restored)
    test.check(status(run)["calls"] == 0, "copied appraisal recovery performs no fresh appraisal")
    retained, saved = asyncio.run(first(test, run, cfg, path, keys, args, old, restored))
    (test.output / "retained.json").write_text(json.dumps(retained, indent=2) + "\n")
    run.stop()
    if any(not row["passed"] for row in test.checks):
        raise AssertionError("Actual shared recall failed before the next recovery.")
    save_recall(test, saved[2])
    complete(test, args, runtime, retained, keys)


def complete(test, args, runtime, retained, keys):
    from appraisal_shared_test import configuration, recovered
    saved = sections(test, runtime)
    copied = test.output / "recovered.aotxccir"
    test.command([test.build / "aotx_ccir", "compact", runtime, copied], "copy-shared-recall")
    cfg, path, grants, new_keys = configuration(test, args.batch, "recovered", 3, args.pages)
    after = RuntimeRun(test, "recovered", copied, extra=("--service-grants", grants)); after.ready(args.ready_seconds)
    restore_report(test, after, f.get(saved[7], 40), f.get(saved[7], 32)); durable(after)
    restored = sections(test, copied); recovered_state(test, f, saved, restored)
    asyncio.run(recovered(test, after, f, copied, cfg, path, new_keys, keys, args, retained, restored,
                          evidence(f, saved[2]), bytes.fromhex(retained["model_sha256"])))
    durable(after); after.stop()
    test.check(not (test.output / "selected-models").exists(),
               "shared continuation uses complete files without an external model directory")
