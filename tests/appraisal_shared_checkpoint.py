#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check saved shared inputs before model work resumes.
# Inputs: Saved metadata and memory bytes. Outputs: Exact checks or an exception.
# Exit: Caller status; no device or model process starts on import.
import base64
import hashlib
import json
from appraisal_runtime_cases import PROCESSOR, current, rows
from appraisal_shared_cases import report
import recall_cli_test as f


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metadata(args, space, inputs, model_digest, state):
    return dict(schema=1, stage="pending", batch=args.batch, space=space, inputs=inputs,
                model_sha256=model_digest.hex(), processor=PROCESSOR.hex(), memory_sha256=sha(state),
                limits=dict(pages=args.pages, tokens=args.tokens, ticks=args.ticks))


def validate(test, saved, state, args):
    test.check(saved["schema"] == 1 and saved["stage"] == "pending" and saved["batch"] == args.batch and
               len(saved["inputs"]) == args.batch and saved["processor"] == PROCESSOR.hex(),
               "saved pending input metadata has the exact batch and processor")
    test.check(saved["limits"] == dict(pages=args.pages, tokens=args.tokens, ticks=args.ticks) and
               saved["memory_sha256"] == sha(state), "saved pending inputs bind exact memory and work limits")
    model = bytes.fromhex(saved["model_sha256"])
    test.check(len(model) == 32 and any(model), "saved pending inputs name a complete model identity")
    available = rows(f, state); queues = current(f, state, b"AOTXAPQ1")
    test.check(len(queues) == args.batch and not current(f, state, b"AOTXREL1") and not current(f, state, kind=3),
               "pending checkpoint contains no completed appraisal")
    owner = bytes.fromhex(saved["space"].split("-")[-1])
    identities, conversations, sources = set(), set(), set()
    for i, entry in enumerate(saved["inputs"]):
        receipt, body = entry["receipt"], entry["body"]
        actor = "%032x" % (i + 1)
        test.check(entry["actor"] == i and receipt["actor"] == actor and receipt["space"] == saved["space"] and
                   receipt["operation"] == 5 and receipt["input_order"] == "1" and
                   receipt["state"] == "completed" and receipt["status"] == 200 and receipt["saved_terminal"] and
                   receipt["saved_admission"] and receipt["usage"]["output_tokens"] > 0 and
                   bool(base64.b64decode(receipt["exact_output"], validate=True)),
                   "each pending source has a distinct authenticated completed input", actor=i)
        expected_key = sha((actor + ":" + receipt["sequence"]).encode())[:32]
        expected = dict(schema="aotx.shared.mutation.v1", lineage=receipt["lineage"], operation_key=expected_key,
                        sequence=receipt["sequence"], text=report(i), model="text", max_output_tokens=32,
                        pages=args.pages, temperature=0)
        test.check(body == expected and receipt["operation_key"] == expected_key and
                   entry["conversation"].startswith("con-" + receipt["lineage"] + "-"),
                   "pending source keeps the exact canonical input and conversation", actor=i)
        found = [(r, p) for r, p in available.values() if f.get(r, 2, 2) == 1 and
                 p[:8] == b"AOTXMEM1" and p[32:] == report(i).encode()]
        test.check(len(found) == 1, "each pending source has one exact retained event", actor=i)
        event, payload = found[0]; sid = bytes(event[8:24]); version = f.get(event, 40)
        test.check(event[120:136] == bytes.fromhex(actor) and event[64:80] == event[80:96] == owner and
                   f.get(event, 176, 4) == 1 and f.get(event, 180, 4) == 3,
                   "pending source retains its authenticated room and author", actor=i)
        found = [(r, p) for r, p in queues if r[96:112] == sid and f.get(r, 112) == version]
        test.check(len(found) == 1, "each pending source has one exact work row", actor=i)
        queue, payload = found[0]
        test.check(len(payload) == 160 and f.get(payload, 8, 4) == 1 and not f.get(payload, 12, 4) and
                   not f.get(payload, 56, 4) and payload[64:96] == PROCESSOR and not any(payload[96:128]) and
                   queue[120:136] == event[120:136] and queue[64:80] == queue[80:96] == owner and
                   f.get(queue, 176, 4) == 1 and f.get(queue, 180, 4) == 4,
                   "pending work has no prior model result and keeps exact attribution", actor=i)
        identities.add(receipt["id"]); conversations.add(entry["conversation"]); sources.add(sid)
    test.check(len(identities) == len(conversations) == len(sources) == args.batch,
               "pending checkpoint covers every distinct input without duplicate rows")


def save_pending(test, args, runtime, space, inputs, model, state):
    value = metadata(args, space, inputs, model, state)
    validate(test, value, state, args)
    save(test.output / "pending-input.json", value)
    test.record(pending_checkpoint=str(runtime), metadata_sha256=sha((test.output / "pending-input.json").read_bytes()))


def save_recall(test, state):
    test.check(all(row["passed"] for row in test.checks), "all recall checks pass before a recovery checkpoint is saved")
    save(test.output / "recall-stage.json", dict(schema=1, stage="recall", exit=0, failed=0, checks=len(test.checks),
         metadata_sha256=sha((test.output / "retained.json").read_bytes()), memory_sha256=sha(state)))


def validate_recall(test, prior, state):
    saved = json.loads((prior / "recall-stage.json").read_text())
    test.check(saved["schema"] == 1 and saved["stage"] == "recall" and
               type(saved["exit"]) is int and type(saved["failed"]) is int and saved["exit"] == saved["failed"] == 0 and
               type(saved["checks"]) is int and saved["checks"] > 0 and
               saved["metadata_sha256"] == sha((prior / "retained.json").read_bytes()) and
               saved["memory_sha256"] == sha(state), "recovery resumes only an exact completed recall checkpoint")
