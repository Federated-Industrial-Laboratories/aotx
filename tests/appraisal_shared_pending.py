#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Resume model appraisal from saved authenticated inputs without new user generation.
# Inputs: Pending metadata and complete runtime. Outputs: A copied runtime and completed assessment metadata.
# Exit: Caller status; prior files are read only.
import asyncio
import json
from aiohttp import ClientSession, ClientTimeout
import recall_cli_test as f
from appraisal_runtime_test import process, status, restore_report, recovered_state
from appraisal_shared_cases import assessment, result_records, selections, events
from appraisal_shared_checkpoint import validate, sha, save
from gateway_runtime_test import aotx_http_run, aotx_ready
from runtime_boot_test import RuntimeRun, durable, sections
from shared_runtime_client import aotx_shared_client


async def assess(test, run, args, saved, cfg, path, keys):
    from appraisal_shared_test import records
    gateway = aotx_http_run(test, path, "pending")
    url = "http://127.0.0.1:" + str(cfg["port"])
    try:
        async with ClientSession(timeout=ClientTimeout(total=args.work_seconds + 60)) as http:
            clients = [aotx_shared_client(test, http, url, key, "%032x" % (i + 1), args.work_seconds)
                       for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            for client in clients:
                test.check((await client.discover())["registered"], "pending recovery preserves every registered participant")
            before = status(run)
            test.check(before["flags"] == 3 and before["pending"] == args.batch and
                       not any(before[key] for key in ("calls", "completed", "active", "refused", "interrupted")),
                       "pending recovery starts without model appraisal", appraisal=before)
            for i, entry in enumerate(saved["inputs"]):
                old = entry["receipt"]; actual = await clients[i].terminal(old["id"])
                test.check(all(actual[key] == old[key] for key in
                    ("id", "actor", "sequence", "input_order", "state", "status", "usage", "finish", "exact_output")),
                    "pending recovery retains the exact original model reply")
                retry = await clients[i].request("POST", "/conversations/" + entry["conversation"] + "/inputs", json=entry["body"])
                test.check(retry["id"] == old["id"] and retry["output"]["base64"] == old["exact_output"],
                           "saved canonical retry creates no new source or user reply")
            await events(test, clients, [[entry] for entry in saved["inputs"]])
            await clients[args.batch].request("GET", "/spaces/" + saved["space"] + "/memory", (404,))
            durable(run)
            test.check(status(run) == before, "saved input reads and retries do not start appraisal")
            process(run, args.batch, args.work_seconds); durable(run)
            state = sections(test, run.runtime_file)[2]
            model = bytes.fromhex(saved["model_sha256"])
            accepted = assessment(test, f, state, saved["inputs"], saved["space"], model)
            recorded = records(test, run, f)
            hashes = result_records(test, f, recorded, state, accepted, model)
            selections(test, f, recorded, state, saved["inputs"], accepted, False)
            test.check(all(row["passed"] for row in test.checks), "all model assessment checks pass before the next stage")
            save(test.output / "assessment-input.json", dict(batch=args.batch, space=saved["space"], inputs=saved["inputs"],
                 accepted=accepted, model_sha256=model.hex(), result_sha256=hashes, memory_sha256=sha(state)))
    finally:
        gateway.close()


def exercise(test, args):
    from appraisal_shared_test import configuration
    prior = args.resume_pending
    saved = json.loads((prior / "pending-input.json").read_text())
    original = prior / "runtime.aotxccir"; runtime = test.output / "runtime.aotxccir"
    before = sections(test, original); validate(test, saved, before[2], args)
    test.command([test.build / "aotx_ccir", "compact", original, runtime], "copy-pending-inputs")
    cfg, path, grants, keys = configuration(test, args.batch, "initial", 2, args.pages)
    run = RuntimeRun(test, "initial", runtime, extra=("--service-grants", grants)); run.runtime_file = runtime
    run.ready(args.ready_seconds); restore_report(test, run, f.get(before[7], 40), f.get(before[7], 32)); durable(run)
    restored = sections(test, runtime); recovered_state(test, f, before, restored)
    validate(test, saved, restored[2], args)
    asyncio.run(assess(test, run, args, saved, cfg, path, keys))
    durable(run); run.stop()
