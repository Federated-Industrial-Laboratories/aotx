#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Define actual source inputs and independent appraisal memory checks.
# Owns: Typed input bytes and source, actor, task, correction and recall expectations.
# Threading: One disk test driver; model and memory processing run on CUDA.
# Lifetime: One maintained runtime test at N=1 or N=64.

# Inputs: Fixture module, memory image and batch. Output: checks. Exit: caller exceptions.
import hashlib
import json
import re

from capacity_boot_test import batch, memory

UNKNOWN = 0xffffffff
SCALE = 1000000
PROCESSOR = bytes.fromhex("68300d48012fccab74b3792122ef616fb6fe886d7b482a2d90e311249785a927")
FIELDS = ("benefit", "harm", "arousal", "consequence", "confidence", "regard_gain", "regard_loss",
          "trust_gain", "trust_loss", "evidence", "task", "commitment", "correction")


def response(test, raw):
    pairs = json.loads(raw, object_pairs_hook=lambda values: values)
    test.check(isinstance(pairs, list) and pairs and pairs[0][0] == "support" and
               type(pairs[0][1]) is int and pairs[0][1] in (0, 1),
               "actual output starts with an integer support decision")
    support = pairs[0][1]
    pairs = pairs[1:]
    test.check(isinstance(pairs, list) and len(pairs) == len(FIELDS) and
               all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs),
               "actual output is one complete named appraisal object")
    test.check(tuple(key for key, _ in pairs) == FIELDS,
               "actual output has exact ordered field names without duplicates or additions")
    value = dict(pairs)
    test.check(all(isinstance(value[key], str) for key in ("evidence", "task", "commitment")) and
               all(type(value[key]) is int for key in FIELDS[:9] + ("correction",)),
               "actual appraisal fields have the declared independent JSON types")
    test.check(all(value[key] == UNKNOWN or 0 <= value[key] <= SCALE for key in FIELDS[:9] if key != "consequence") and
               0 <= value["consequence"] <= 4 and 0 <= value["correction"] <= 16,
               "actual appraisal numbers have the declared independent domains")
    supported = value["consequence"] != 0 or any(value[key] != UNKNOWN for key in FIELDS[:9] if key != "consequence")
    test.check(support == supported, "support decision agrees with the accepted dimensions")
    test.check(bool(value["evidence"]) == supported and
               (supported or not value["task"] and not value["commitment"] and value["correction"] == 0),
               "actual output has supporting evidence exactly when it has a known interpretation")
    return value


def task(i):
    return f"packing crate {i:04d}"


def source(i, case):
    if case == "unknown":
        return (f"Visitor{i:04d} says Jordan{i:04d} damaged a tool while {task(i)}. "
                "This report concerns other people only. I am new here and have provided no account "
                "of my own experiences or contributions. Reply with exactly one word: noted.")
    if case == "mixed":
        return (f"While {task(i)}, I completed my work successfully and helped our group. "
                "I also damaged a tool and caused a costly delay. I felt strong excitement and worry. "
                "Reply with exactly one word: noted.")
    if case == "correction":
        return (f"Correction about {task(i)}: my earlier report that I damaged a tool and caused a costly delay was wrong. "
                "No tool was damaged and no delay occurred. I completed the work successfully and helped our group. "
                "Reply with exactly one word: noted.")
    return f"Did I damage a tool during {task(i)}? Use current memory. Reply with exactly one word: yes or no."


def source_id(f, i, ordinal):
    return f.identity(300000 + ordinal * 64 + i)


def initial(f, count):
    values = []
    for i in range(count):
        row = f.object_row(5, 40000 + i, 10000 + i, i + 1)
        f.put(row, 180, 1, 4); row[120:136] = bytes(16)
        values.append((row, memory(f, task(i))))
    return f.image(values, count, 10)


def bindings(f, count, cut, pages, conversation=8000):
    data = batch(f, b"AOTXBND1", 64, cut, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(conversation + i)
        f.put(data, at + 56, pages, 4); f.put(data, at + 60, 2, 4)
    return data


def request(f, count, cut, ordinal, case, appraise=False, conversation=8000, generation=0):
    data = batch(f, b"AOTXTXT1", 8256, cut, count)
    for i in range(count):
        at = 64 + i * 8256; f.put(data, at, i, 4)
        data[at + 16:at + 32] = f.identity(conversation + i); f.put(data, at + 32, ordinal)
        q = at + 64; raw = source(i, case).encode()
        data[q:q + 16], data[q + 16:q + 32], data[q + 48:q + 64] = (
            source_id(f, i, ordinal + generation * 1000), f.identity(10000 + i), f.identity(400000 + (ordinal + generation * 1000) * 64 + i))
        for offset, value in ((132, 16), (136, 4096), (148, len(raw))):
            f.put(data, q + offset, value, 4)
        data[q + 4640:q + 4640 + len(raw)] = raw
        c = q + 6688; data[c:c + 8] = b"AOTXCTX2"
        for offset, value in ((8, 2), (12, 3 if appraise else 1), (32, 1), (44, 2)):
            f.put(data, c + offset, value, 4)
        if appraise:
            f.put(data, c + 40, SCALE, 4)
        data[c + 16:c + 32], data[c + 48:c + 64] = f.identity(40000 + i), f.identity(10000 + i)
        data[q + 7760:q + 7776] = f.identity(10000 + i)
    return data


def rows(f, state):
    return {(bytes(row[8:24]), f.get(row, 40)): (row, payload) for row, payload in f.state_rows(state)}


def current(f, state, magic=None, kind=None):
    values = rows(f, state)
    latest = {}
    superseded = {(bytes(row[136:152]), f.get(row, 152)) for row, _ in values.values() if any(row[136:152])}
    for key, value in values.items():
        if key[0] not in latest or key[1] > latest[key[0]][0]:
            latest[key[0]] = key[1], value
    result = []
    for identity, (version, (row, payload)) in latest.items():
        if (identity, version) in superseded or f.get(row, 4, 4) & 1 or f.get(row, 184, 4) == 3:
            continue
        if magic is not None and payload[:8] != magic:
            continue
        if kind is not None and f.get(row, 2, 2) != kind:
            continue
        result.append((row, payload))
    return result


def evidence(f, state):
    return {(bytes(row[8:24]), f.get(row, 40)): (bytes(row), bytes(payload)) for row, payload in f.state_rows(state)
            if f.get(row, 2, 2) in (3, 4) or payload[:8] == b"AOTXAPQ1"}


def span(test, f, raw, payload, offset, label):
    start, length = f.get(payload, offset, 4), f.get(payload, offset + 4, 4)
    test.check(start <= len(raw) and length <= len(raw) - start and (length or not start), label + " span bounds")
    quote = raw[start:start + length]
    quote.decode("utf-8")
    if length:
        hits = sum(raw[at:at + length] == quote for at in range(len(raw) - length + 1))
        test.check(hits == 1, label + " is a unique exact source quote")
    return quote


def assess(test, f, state, count, ordinal, case, model_digest, old=None):
    all_rows = rows(f, state)
    queues = current(f, state, b"AOTXAPQ1")
    relations = current(f, state, b"AOTXREL1")
    assessments = current(f, state, kind=3)
    result = []
    for i in range(count):
        sid = source_id(f, i, ordinal)
        events = [(row, payload) for (identity, _), (row, payload) in all_rows.items() if identity == sid]
        test.check(len(events) == 1, "actual input has one immutable source event", slot=i, ordinal=ordinal)
        event, event_payload = events[0]; raw = source(i, case).encode(); version = f.get(event, 40)
        test.check(event_payload[:8] == b"AOTXMEM1" and event_payload[32:] == raw and
                   event[120:136] == f.identity(10000 + i), "external source retains its exact text and admitted actor", slot=i)
        chosen = [[(row, payload) for row, payload in collection if row[96:112] == sid and f.get(row, 112) == version]
                  for collection in (queues, relations, assessments)]
        test.check(len(chosen[0]) == len(chosen[1]) == len(chosen[2]) == 1,
                   "each processed source has one completed queue, relationship and assessment", slot=i)
        queue, relation, assessment = (group[0] for group in chosen)
        qr, qp = queue; rr, rp = relation; ar, ap = assessment
        for row, payload in (queue, relation, assessment):
            test.check(row[64:80] == f.identity(10000 + i) and row[80:96] == bytes(16) and
                       row[120:136] == event[120:136] and f.get(row, 176, 4) == 0 and f.get(row, 180, 4) == 4,
                       "derived memory keeps source actor and private scope", slot=i)
        test.check(len(qp) == 160 and f.get(qp, 8, 4) == f.get(qp, 12, 4) == 1 and not f.get(qp, 56, 4) and
                   qp[64:96] == PROCESSOR and qp[96:128] == model_digest,
                   "completed queue identifies the actual processor and model", slot=i)
        test.check(qp[40:56] == qp[128:144] == f.identity(40000 + i) and f.get(qp, 144) == 1,
                   "queue preserves the exact authored task descriptor", slot=i)
        test.check(len(rp) == 192 and f.get(rp, 8, 4) == f.get(rp, 12, 4) == 1 and
                   rp[72:104] == ap[32:64] == PROCESSOR and rp[104:136] == ap[64:96] == model_digest,
                   "one external source contributes exactly one exposure with actual provenance", slot=i)
        test.check(len(ap) == 128 and f.get(ap, 0, 4) == 2 and
                   rp[136:152] == ap[96:112] == qr[8:24] and f.get(rp, 152) == f.get(ap, 112) == f.get(qr, 40),
                   "relationship and assessment use the same completed queue version", slot=i)
        aq = span(test, f, raw, ap, 120, "assessment")
        rq = span(test, f, raw, rp, 48, "relationship")
        tq = span(test, f, raw, rp, 56, "task")
        commitment = span(test, f, raw, rp, 64, "commitment")
        test.check(not commitment, "a report without a promise produces no commitment", kind="behavior", slot=i)
        test.check(aq == rq, "relationship and assessment retain the same evidence quote", slot=i)
        values = [f.get(ap, at, 4) for at in (4, 8, 12, 20)] + [f.get(rp, at, 4) for at in (16, 20, 24, 28)]
        test.check(all(value == UNKNOWN or 0 <= value <= SCALE for value in values) and f.get(ap, 16, 4) <= 4,
                   "all independently decoded values retain their declared domains", slot=i)
        if any(f.get(rp, at, 4) != UNKNOWN for at in (24, 28)):
            test.check(tq == task(i).encode(), "known task trust names the complete registered task", slot=i)
        if case == "unknown":
            test.check(not aq and all(value == UNKNOWN for value in values) and not f.get(ap, 16, 4),
                       "another person's incident gives a new actor no inferred benefit, harm, regard or trust", kind="behavior", slot=i)
        elif case == "mixed":
            test.check(bool(aq) and 0 < f.get(ap, 4, 4) <= SCALE and 0 < f.get(ap, 8, 4) <= SCALE,
                       "actual model retains both supported benefit and harm", kind="behavior", slot=i, values=values, quote=aq.decode())
            test.check(all(0 < f.get(rp, at, 4) <= SCALE for at in (16, 20, 24, 28)),
                       "actual model preserves both helpful and harmful regard and task trust evidence",
                       kind="behavior", slot=i, values=values)
        elif case == "correction":
            test.check(old is not None and ar[136:152].hex() == old[i]["assessment"] and
                       rr[136:152].hex() == old[i]["relationship"] and f.get(ar, 152) == old[i]["assessment_version"] and
                       f.get(rr, 152) == old[i]["relationship_version"],
                       "actual correction supersedes both prior interpretations for the same actor", kind="behavior", slot=i)
            test.check(f.get(ap, 8, 4) in (0, UNKNOWN), "retracted harm is not asserted again", kind="behavior", slot=i)
        result.append(dict(source=sid.hex(), source_version=version, assessment=ar[8:24].hex(), assessment_version=f.get(ar, 40),
                           relationship=rr[8:24].hex(), relationship_version=f.get(rr, 40),
                           queue=qr[8:24].hex(), queue_version=f.get(qr, 40), quote=aq.decode(), values=values))
    test.record(case=case, ordinal=ordinal, batch=count, actual_appraisals=result,
                memory_sha256=hashlib.sha256(state).hexdigest())
    return result


def selected(test, run, f, state, count, ordinal, corrected, old, enabled=True):
    available = rows(f, state)
    for i in range(count):
        events = [row for row in run.events(i) if row.get("kind") == "selection" and row.get("turn") == ordinal]
        test.check(len(events) == 1, "fresh input has one recorded memory selection", slot=i, ordinal=ordinal)
        picked = {(bytes.fromhex(identity), int(version)) for identity, version in
                  re.findall(r"([0-9a-f]{32})@(\d+)", events[0]["text"])}
        for key in picked:
            test.check(key in available, "selected memory names an exact existing version", slot=i)
            row, _ = available[key]
            test.check(f.get(row, 176, 4) != 0 or row[64:80] == f.identity(10000 + i),
                       "selection cannot include another actor's private memory", slot=i)
        cognitive = {key for key in picked if f.get(available[key][0], 2, 2) in (3, 4)}
        expected = corrected[i]
        wanted = {(bytes.fromhex(expected[k]), expected[k + "_version"]) for k in ("assessment", "relationship")}
        if enabled:
            test.check(wanted <= cognitive and (bytes.fromhex(expected["source"]), expected["source_version"]) in picked,
                       "actual recall consumes corrected cognitive evidence with its source", kind="behavior", slot=i)
        else:
            test.check(not cognitive, "disabled cognitive recall adds no appraisal or relationship selection", slot=i)
        forbidden = {(bytes.fromhex(old[i][k]), old[i][k + "_version"]) for k in ("assessment", "relationship")}
        test.check(not (picked & forbidden), "superseded interpretations are absent from the fresh selection", slot=i)
