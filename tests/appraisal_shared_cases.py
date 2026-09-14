# SPDX-License-Identifier: Apache-2.0
# Purpose: Check authenticated shared source, appraisal and recorded recall bytes.
# Owns: Independent expected source IDs, actor bindings and result checks.
# Threading: One disk driver; CUDA produces all assessed values and selections.
# Lifetime: Actual shared input through complete file recovery.

# Inputs: Admitted runtime bytes and actual receipts. Output: Checks and source records.
# Exit: Caller exceptions; no model or device process starts on import.
import asyncio
import base64
import hashlib
import re
from appraisal_runtime_cases import PROCESSOR, SCALE, UNKNOWN, current, rows, span, response, FIELDS
from appraisal_shared_events import EVENTS
from shared_runtime_cases import aotx_shared_memory_object


def report(i):
    event = EVENTS[i]
    return (f"During {event['topic']}, I {event['benefit']}. I also {event['harm']}. "
            "Both outcomes mattered to our group. I felt excitement and worry. "
            "Reply with exactly one word: noted.")


def recall(i, ordinal):
    if ordinal not in (1, 2):
        raise ValueError("invalid recall wave")
    return (f"During {EVENTS[i]['topic']}, what went well and what went badly for the member who reported it? "
            "Use the shared memory and give one complete sentence with both outcomes.")


def targets(count, ordinal):
    if count not in (1, 64) or ordinal not in (1, 2):
        raise ValueError("invalid recall batch or wave")
    offset = 17 if ordinal == 1 else 29
    return tuple((caller + offset) % count for caller in range(count))


def supported(target, text, finish):
    if type(target) is not int or not 0 <= target < len(EVENTS) or type(finish) is not int or finish != 1:
        return False
    text = text.strip()
    if re.fullmatch(r"[^.!?]+[.!]", text) is None:
        return False
    denied = (r"\b(?:no|never|neither|nor|false|untrue|incorrect|inaccurate|without|cannot|unknown|uncertain)\b"
              r"|\bnot\b(?!\s+only\b)|\bden(?:y|ies|ied)\b|\bfail(?:ed|s)?\s+to\b"
              r"|\b(?:isn|wasn|weren|didn|doesn|don|haven|hasn|hadn|wouldn|couldn|shouldn|won|can)['\u2019]t\b")
    if re.search(denied, text, re.I):
        return False
    event = EVENTS[target]
    outcomes = re.sub(re.escape(event["topic"]), "", text, flags=re.I)
    clauses = re.split(r"[,;:]|\b(?:but|and|however|although|whereas|while|yet)\b", outcomes, flags=re.I)
    return all(any(all(re.search(term, clause, re.I) for term in event[key]) for clause in clauses)
               for key in ("benefit_terms", "harm_terms"))


def useful(test, entries):
    for entry in entries:
        text = base64.b64decode(entry["receipt"]["exact_output"], validate=True).decode()
        test.check(supported(entry["target"], text, entry["receipt"]["finish"]),
                   "actual shared answer completes both outcomes of the requested source", kind="behavior",
                   actor=entry["actor"], target=entry["target"], finish=entry["receipt"]["finish"], text=text)


def source_group(item):
    return {(bytes.fromhex(item[key]), item[key + "_version"])
            for key in ("source", "queue", "assessment", "relationship")}


def target_selected(picked, accepted, target):
    return type(target) is int and 0 <= target < len(accepted) and source_group(accepted[target]) <= picked


def receipt(test, value, participant, space):
    test.check(value["actor"] == participant and value["space"] == space and value["operation"] == 5,
               "actual input receipt names its authenticated actor and shared space", receipt=value)


async def wave(test, clients, conversations, space, text, ordinal, pages, requested=None):
    async def one(i):
        value, body = await clients[i].answer(conversations[i], text(i), pages=pages, temperature=0,
                                             max_output_tokens=32 if requested is None else 96)
        receipt(test, value, clients[i].participant, space)
        test.check(value["input_order"] == str(ordinal), "actual caller input has one conversation order")
        retry = await clients[i].request("POST", "/conversations/" + conversations[i] + "/inputs", json=body)
        test.check(retry["id"] == value["id"] and retry["output"]["base64"] == value["exact_output"],
                   "exact shared retry returns the saved model result without a new input")
        entry = dict(actor=i, conversation=conversations[i], receipt=value, body=body)
        if requested is not None:
            entry["target"] = requested[i]
        return entry
    values = await asyncio.gather(*(one(i) for i in range(len(conversations))))
    test.check(len({v["receipt"]["id"] for v in values}) == len(conversations) and
               len({v["body"]["text"] for v in values}) == len(conversations), "the shared input batch has distinct receipts and sources")
    if requested is not None:
        test.check({entry["target"] for entry in values} == set(range(len(conversations))) and
                   all(entry["target"] != entry["actor"] or len(conversations) == 1 for entry in values),
                   "each shared recall targets one distinct member source in the admitted batch")
    return values


def assessment(test, f, state, entries, space, model_digest):
    available = rows(f, state); selected = []
    queues, relations, assessments = current(f, state, b"AOTXAPQ1"), current(f, state, b"AOTXREL1"), current(f, state, kind=3)
    owner = bytes.fromhex(space.split("-")[-1])
    test.check(len(queues) == len(relations) == len(assessments) == len(entries),
               "each shared external input has one current queue, relationship and assessment")
    for i, entry in enumerate(entries):
        actor, raw = bytes.fromhex(entry["receipt"]["actor"]), entry["body"]["text"].encode()
        matches = [(r, p) for r, p in available.values() if f.get(r, 2, 2) == 1 and p[:8] == b"AOTXMEM1" and p[32:] == raw]
        test.check(len(matches) == 1 and actor != owner, "unique shared input has a source actor distinct from its memory owner", actor=i)
        er, ep = matches[0]; sid, version = bytes(er[8:24]), f.get(er, 40)
        test.check(ep[32:] == report(i).encode() and er[120:136] == actor and er[64:80] == er[80:96] == owner and
                   f.get(er, 176, 4) == 1 and f.get(er, 180, 4) == 3, "source text and authenticated caller are preserved in room memory", actor=i)
        groups = [[(r, p) for r, p in group if r[96:112] == sid and f.get(r, 112) == version]
                  for group in (queues, relations, assessments)]
        test.check(all(len(group) == 1 for group in groups), "one complete derived group refers to the exact shared source", actor=i)
        (qr, qp), (rr, rp), (ar, ap) = [group[0] for group in groups]
        for r, _ in [group[0] for group in groups]:
            test.check(r[64:80] == r[80:96] == owner and r[120:136] == actor and f.get(r, 176, 4) == 1 and
                       f.get(r, 180, 4) == 4 and not any(r[136:160]),
                       "derived evidence keeps the room owner and authenticated subject without a correction", actor=i)
        test.check(len(qp) == 160 and f.get(qp, 8, 4) == f.get(qp, 12, 4) == 1 and not f.get(qp, 56, 4) and
                   qp[64:96] == PROCESSOR and qp[96:128] == model_digest,
                   "completed shared queue names the actual processor and resident model", actor=i)
        test.check(not any(qp[40:56] + qp[128:160]) and not any(rp[32:48]) and
                   f.get(rp, 24, 4) == f.get(rp, 28, 4) == UNKNOWN,
                   "an unregistered task cannot acquire task identity or known trust", actor=i)
        test.check(len(ap) == 128 and f.get(ap, 0, 4) == 2 and len(rp) == 192 and
                   f.get(rp, 8, 4) == f.get(rp, 12, 4) == 1 and rp[72:104] == ap[32:64] == PROCESSOR and
                   rp[104:136] == ap[64:96] == model_digest and
                   rp[136:152] == ap[96:112] == qr[8:24] and f.get(rp, 152) == f.get(ap, 112) == f.get(qr, 40),
                   "shared assessment and one exposure use the same exact completed queue", actor=i)
        aq = span(test, f, raw, ap, 120, "shared assessment")
        test.check(aq == span(test, f, raw, rp, 48, "shared relationship"), "shared evidence quotes are equal", actor=i)
        span(test, f, raw, rp, 56, "shared task"); span(test, f, raw, rp, 64, "shared commitment")
        values = [f.get(ap, at, 4) for at in (4, 8, 12, 20)] + [f.get(rp, at, 4) for at in (16, 20, 24, 28)]
        test.check(all(value == UNKNOWN or 0 <= value <= SCALE for value in values) and f.get(ap, 16, 4) <= 4,
                   "actual shared values retain their independent domains", actor=i)
        test.check(bool(aq) and 0 < f.get(ap, 4, 4) <= SCALE and 0 < f.get(ap, 8, 4) <= SCALE,
                   "actual shared appraisal retains supported benefit and harm", kind="behavior", actor=i)
        test.check(any(f.get(rp, at, 4) != UNKNOWN for at in (16, 20)),
                   "actual shared appraisal supplies usable regard evidence", kind="behavior", actor=i)
        selected.append(dict(actor=actor.hex(), owner=owner.hex(), source=sid.hex(), source_version=version,
            queue=qr[8:24].hex(), queue_version=f.get(qr, 40), assessment=ar[8:24].hex(), assessment_version=f.get(ar, 40),
            relationship=rr[8:24].hex(), relationship_version=f.get(rr, 40), quote=aq.decode(), values=values))
    test.check(len({v["actor"] for v in selected}) == len(entries) and len({v["source"] for v in selected}) == len(entries),
               "all authenticated callers and their actual sources remain distinct")
    test.record(shared_assessments=selected, memory_sha256=hashlib.sha256(state).hexdigest())
    return selected


def result_records(test, f, records, state, accepted, model_digest):
    available = rows(f, state); expected = {v["queue"]: v for v in accepted}; seen = set(); results = []
    for op, _, data in records:
        if op != 17:
            continue
        n, tail = f.get(data, 12, 4), f.get(data, 24)
        test.check(data[:8] == b"AOTXAPS1" and f.get(data, 8, 4) == 1 and 0 < n <= 64 and
                   not f.get(data, 32, 4) and not any(data[36:64]) and len(data) == 64 + n * 4160 + tail,
                   "actual shared appraisal result has complete independent framing")
        encoded = data[64 + n * 4160:]
        test.check(encoded[:8] == b"AOTXLOG1" and f.get(encoded, 32) == f.get(data, 16) + 1 and
                   f.get(encoded, 80) == tail, "shared result contains its exact canonical typed tail")
        results.append(data)
        for j in range(n):
            row = data[64 + j * 4160:64 + (j + 1) * 4160]; identity = row[:16].hex(); length = f.get(row, 56, 4)
            test.check(identity in expected and identity not in seen and (row[:16], f.get(row, 16)) in available and
                       row[24:56] == model_digest and not f.get(row, 60, 4) and 0 < length <= 4096 and not any(row[64 + length:]),
                       "actual shared decoder reply has one exact admitted queue and model")
            seen.add(identity); chosen = expected[identity]; value = response(test, row[64:64 + length])
            ap = available[(bytes.fromhex(chosen["assessment"]), chosen["assessment_version"])][1]
            rp = available[(bytes.fromhex(chosen["relationship"]), chosen["relationship_version"])][1]
            raw = available[(bytes.fromhex(chosen["source"]), chosen["source_version"])][1][32:]
            test.check(value["correction"] == 0,
                       "shared decoder reply has exact typed fields and no foreign correction target")
            numbers = [f.get(ap, at, 4) for at in (4, 8, 12, 16, 20)] + [f.get(rp, at, 4) for at in (16, 20, 24, 28)]
            test.check([value[key] for key in FIELDS[:9]] == numbers, "persisted shared values equal actual model output")
            for field, payload, offset in (("evidence", ap, 120), ("task", rp, 56), ("commitment", rp, 64)):
                start, size = f.get(payload, offset, 4), f.get(payload, offset + 4, 4)
                test.check(value[field].encode() == raw[start:start + size], "actual shared result quote equals its persisted source span")
    test.check(seen == set(expected), "all shared sources have exactly one recorded actual appraisal result")
    return [hashlib.sha256(data).hexdigest() for data in results]


def selections(test, f, records, state, entries, accepted, enabled):
    decisions = {}
    for op, _, data in records:
        if op != 14:
            continue
        count, stride = f.get(data, 8, 4), f.get(data, 40, 4)
        test.check(data[:8] == b"AOTXICH1" and 0 < count <= 64 and stride == 13920 and not f.get(data, 44, 4) and
                   len(data) == 64 + count * stride + f.get(data, 48), "shared intake decision has exact framing")
        for j in range(count):
            row = data[64 + j * stride:64 + (j + 1) * stride]; key = row[16:32].hex(), f.get(row, 32)
            test.check(key not in decisions, "one recorded choice exists for each shared conversation input")
            decisions[key] = row
    available = rows(f, state); consumed = []
    for entry in entries:
        key = entry["conversation"].split("-")[-1], int(entry["receipt"]["input_order"])
        test.check(key in decisions, "actual shared receipt has its recorded memory decision")
        row = decisions[key]; q = row[64:8256]; choice = row[13392:13920]
        owner = bytes.fromhex(entry["receipt"]["space"].split("-")[-1]); actor = bytes.fromhex(entry["receipt"]["actor"])
        test.check(q[16:32] == q[32:48] == owner and q[4640:4640 + f.get(q, 148, 4)] == entry["body"]["text"].encode() and
                   f.get(q, 152, 4) == 1, "recorded shared query retains the exact room and input bytes")
        test.check(q[6688:6696] == b"AOTXCTX1" and f.get(q, 6700, 4) == 2 and not f.get(q, 6724, 4) and
                   f.get(q, 6728, 4) == SCALE and not any(q[6704:6724]), "shared query records configured recall without an invented task")
        source_row = available[(bytes(q[:16]), 1)][0]
        test.check(source_row[120:136] == actor and source_row[64:80] == owner,
                   "saved shared query source is bound to the authenticated caller")
        count = f.get(choice, 4, 4)
        test.check(f.get(choice, 0, 4) == 1 and count <= 16 and not any(choice[8:16]) and not any(choice[16 + count * 32:]),
                   "shared final selection has exact bounded entries")
        picked = {(bytes(choice[16 + k * 32:32 + k * 32]), f.get(choice, 32 + k * 32)) for k in range(count)}
        test.check(len(picked) == count and picked <= available.keys(), "shared selection names exact existing versions")
        for identity in picked:
            selected, _ = available[identity]
            test.check(f.get(selected, 176, 4) == 2 or (f.get(selected, 176, 4) == 1 and selected[80:96] == owner),
                       "selected shared memory remains in the admitted room or instance")
        complete = []
        for item in accepted:
            group = source_group(item)
            if group <= picked:
                complete.append(item["source"])
            test.check(not (picked & (group - {(bytes.fromhex(item["source"]), item["source_version"])})) or group <= picked,
                       "shared recall cannot expose a partial appraisal source group")
        if enabled:
            test.check(target_selected(picked, accepted, entry["target"]),
                       "actual shared recall consumes the requested source and its complete evidence group",
                       kind="behavior", actor=entry["actor"], target=entry["target"], selected_sources=complete)
        else:
            test.check(not any(f.get(available[k][0], 2, 2) in (3, 4) for k in picked), "source admission precedes inferred shared appraisal")
        consumed.append(dict(conversation=key[0], ordinal=key[1], actor=actor.hex(), selected_sources=complete,
            target=entry.get("target"), requested_source=accepted[entry["target"]]["source"] if enabled else None))
    test.record(shared_selections=consumed)


async def exposed(test, f, client, space, state, accepted):
    items = await client.inventory(space); table = {item["id"]: item for item in items}; available = rows(f, state)
    for item in accepted:
        for kind in ("source", "queue", "assessment", "relationship"):
            identity = item[kind]; test.check(identity in table, "shared memory API exposes accepted evidence")
            metadata, payload = await aotx_shared_memory_object(test, client, space, identity)
            test.check(metadata == table[identity] and metadata["owner"] == metadata["room"] == item["owner"] and
                       metadata["actor"] == item["actor"] and metadata["scope"] == "room" and
                       metadata["version"] == str(item[kind + "_version"]), "shared memory API preserves exact actor and room metadata")
            test.check(payload == available[(bytes.fromhex(identity), item[kind + "_version"])][1],
                       "shared memory API returns the exact admitted payload")
    return items


async def events(test, clients, entries):
    for i, group in enumerate(entries):
        by_conversation = {}
        for item in group:
            by_conversation.setdefault(item["conversation"], []).append(item)
        for conversation, expected in by_conversation.items():
            value = await clients[i].request("GET", "/conversations/" + conversation + "/events")
            test.check(not value["gap"] and [r["id"] for r in value["items"]] == [e["receipt"]["id"] for e in expected] and
                       [r["input_order"] for r in value["items"]] == [str(j + 1) for j in range(len(expected))] and
                       all(r["actor"] == clients[i].participant for r in value["items"]),
                       "appraisal, retries and recovery add no shared user input turns")
