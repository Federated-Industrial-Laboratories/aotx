#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check saved input boundaries, attribution and incomplete recovery stages.
# Inputs: None. Outputs: Check counts. Exit: 0 pass, 1 failed assertion.
import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from appraisal_shared_checkpoint import metadata, validate, sha, save, validate_recall
from appraisal_runtime_cases import PROCESSOR
from appraisal_shared_cases import report
import recall_cli_test as f


class Check:
    def __init__(self): self.count = 0
    def check(self, value, label, **details):
        self.count += 1
        if not value: raise AssertionError(label)


def fixture(n):
    args = SimpleNamespace(batch=n, pages=160, tokens=512, ticks=16384)
    lineage, room = "01" * 16, "02" * 16
    space = "spc-" + lineage + "-" + room
    objects, inputs = [], []
    for i in range(n):
        actor = "%032x" % (i + 1); sequence = str(i + 3)
        key = sha((actor + ":" + sequence).encode())[:32]
        receipt = dict(actor=actor, space=space, operation=5, input_order="1", state="completed", status=200,
                       saved_terminal=True, saved_admission=True, usage=dict(output_tokens=2),
                       exact_output=base64.b64encode(b"noted").decode(), sequence=sequence, operation_key=key,
                       lineage=lineage, id="op-" + lineage + "-" + key)
        body = dict(schema="aotx.shared.mutation.v1", lineage=lineage, operation_key=key, sequence=sequence,
                    text=report(i), model="text", max_output_tokens=32, pages=160, temperature=0)
        inputs.append(dict(actor=i, conversation="con-" + lineage + "-" + actor, receipt=receipt, body=body))
        event = f.object_row(1, 1000 + i, 1, i + 1)
        event[64:80] = event[80:96] = bytes.fromhex(room); event[120:136] = bytes.fromhex(actor)
        f.put(event, 176, 1, 4)
        raw = report(i).encode(); text = bytearray(32) + raw; text[:8] = b"AOTXMEM1"
        f.put(text, 8, 1, 4); f.put(text, 12, len(raw), 4)
        queue = copy.deepcopy(event); queue[8:24] = f.identity(2000 + i); f.put(queue, 2, 12, 2)
        queue[96:112] = event[8:24]; f.put(queue, 112, 1); f.put(queue, 180, 4, 4)
        payload = bytearray(160); payload[:8] = b"AOTXAPQ1"; f.put(payload, 8, 1, 4); payload[64:96] = PROCESSOR
        objects.extend(((event, text), (queue, payload)))
    state = f.image(objects, 2 * n, 10)
    return args, metadata(args, space, inputs, b"m" * 32, state), state, objects


def main():
    total, rejected = 0, 0
    for n in (1, 31, 32, 33, 63, 64):
        args, saved, state, objects = fixture(n)
        check = Check(); validate(check, saved, state, args); total += check.count
        mutations = []
        for key, value in (("batch", n + 1), ("processor", "00" * 32), ("stage", "complete"),
                           ("memory_sha256", "00" * 32), ("model_sha256", "00" * 32)):
            bad = copy.deepcopy(saved); bad[key] = value; mutations.append((bad, state))
        bad = copy.deepcopy(saved); bad["inputs"].pop(); mutations.append((bad, state))
        for section, key, value in (("receipt", "actor", "ff" * 16), ("receipt", "saved_terminal", False),
                                    ("receipt", "operation_key", "ff" * 16), ("body", "text", "wrong source"),
                                    ("body", "pages", 159)):
            bad = copy.deepcopy(saved); bad["inputs"][-1][section][key] = value; mutations.append((bad, state))
        if n > 1:
            bad = copy.deepcopy(saved); bad["inputs"][-1] = copy.deepcopy(bad["inputs"][0]); mutations.append((bad, state))
        for index, offset, value in ((-1, 12, 1), (-1, 56, 1), (-1, 64, 0), (-1, 96, 1)):
            changed = copy.deepcopy(objects); f.put(changed[index][1], offset, value, 1)
            damaged = f.image(changed, 2 * n, 10); bad = copy.deepcopy(saved); bad["memory_sha256"] = sha(damaged)
            mutations.append((bad, damaged))
        for offset in (176, 180):
            changed = copy.deepcopy(objects); f.put(changed[-1][0], offset, 0, 4)
            damaged = f.image(changed, 2 * n, 10); bad = copy.deepcopy(saved); bad["memory_sha256"] = sha(damaged)
            mutations.append((bad, damaged))
        for bad, damaged in mutations:
            try: validate(Check(), bad, damaged, args)
            except (AssertionError, ValueError): rejected += 1
            else: raise AssertionError("A damaged pending checkpoint was accepted.")
    with tempfile.TemporaryDirectory() as directory:
        prior = Path(directory); save(prior / "retained.json", {"source_count": 64})
        stage = dict(schema=1, stage="recall", exit=0, failed=0, checks=10,
                     metadata_sha256=sha((prior / "retained.json").read_bytes()), memory_sha256=sha(b"state"))
        save(prior / "recall-stage.json", stage); check = Check(); validate_recall(check, prior, b"state"); total += check.count
        for key, value in (("checks", 0), ("exit", 1), ("failed", 1), ("exit", False), ("failed", False), ("memory_sha256", "00" * 32),
                           ("metadata_sha256", "00" * 32)):
            save(prior / "recall-stage.json", dict(stage, **{key: value}))
            try: validate_recall(Check(), prior, b"state")
            except AssertionError: rejected += 1
            else: raise AssertionError("An incomplete recall checkpoint was accepted.")
    print(json.dumps(dict(checks=total + rejected, rejected=rejected, failed=0, exit=0)))


if __name__ == "__main__": main()
