#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check exact shared recall targets, outcome checks and selected evidence groups.
# Owns: Distinct source facts, reference answers and independent selection byte fixtures.
# Threading: One CPU test process with in-process clients at batch sizes one and 64.
# Lifetime: Report, recall and recorded selection checks; no runtime process starts.

# Inputs: None. Output: Check and failure counts. Exit: 0 pass, 1 failed check.
import asyncio
import base64
import sys

sys.dont_write_bytecode = True
import recall_cli_test as f
from appraisal_shared_events import EVENTS
from appraisal_shared_cases import report, recall, targets, supported, useful, wave, source_group, target_selected, selections

checks = failures = 0


def check(value, label):
    global checks, failures
    checks += 1
    if not value:
        failures += 1
        print("failed: " + label, file=sys.stderr)


class Trace:
    def __init__(self):
        self.rows = []

    def check(self, value, label, **fields):
        self.rows.append(dict(passed=bool(value), label=label, **fields))

    def record(self, **fields):
        pass


def recorded(trace, refused=0):
    bad = [row for row in trace.rows if not row["passed"]]
    check(len(bad) == refused, "exact count of refused checks")
    for row in trace.rows:
        check(row["passed"] or (refused > 0 and row.get("kind") == "behavior"), row["label"])


def reference(i):
    event = EVENTS[i]
    return f"The member {event['benefit']}, but also {event['harm']}."


def answers(count):
    check(len(EVENTS) == 64 and len({item["topic"] for item in EVENTS}) == 64, "all event topics are distinct")
    for i, event in enumerate(EVENTS[:count]):
        check(set(event) == {"topic", "benefit", "harm", "benefit_terms", "harm_terms"}, "event fields are exact")
        check(all(isinstance(event[key], tuple) and event[key] for key in ("benefit_terms", "harm_terms")),
              "both outcomes have explicit term groups")
        text = report(i)
        check(text == f"During {event['topic']}, I {event['benefit']}. I also {event['harm']}. "
              "Both outcomes mattered to our group. I felt excitement and worry. Reply with exactly one word: noted.",
              "reports retain exact event facts and the historical answer instruction")
        check("sharedreport" not in text and text.count("Reply with exactly one word: noted.") == 1,
              "each report carries one historical instruction without a numeric label")
        for ordinal in (1, 2):
            query = recall(i, ordinal)
            check(event["topic"] in query and event["benefit"] not in query and event["harm"] not in query,
                  "recall asks for a topic without supplying its outcomes")
        good = reference(i)
        check(supported(i, good, 1), "independent complete reference answer")
        check(supported(i, f"During {event['topic']}, " + good[0].lower() + good[1:], 1),
              "a topic prefix leaves both stated outcomes intact")
        check(supported(i, f"The member not only {event['benefit']} but also {event['harm']}.", 1),
              "positive paired emphasis does not negate an outcome")
        for finish in (0, 2, 3, 4, True, "1"):
            check(not supported(i, good, finish), "only a normal finish qualifies the answer")
        for bad in (f"The member {event['benefit']}.", f"The member {event['harm']}.",
                    "The work succeeded but caused damage.", "Noted.", "I helped the group and damaged a tool.",
                    "It is false that " + good, "The member never " + good[11:],
                    "The member did not " + good[11:], "There was no harm: " + good,
                    "Neither outcome happened: " + good, "The member didn't " + good[11:],
                    good[:-1], good[:-1] + "?", good + " This was useful."):
            check(not supported(i, bad, 1), "incomplete, generic or negated outcome answer")
        for other in range(len(EVENTS)):
            if other == i:
                continue
            check(not supported(i, reference(other), 1), "another source answer cannot qualify the requested source")
            check(not supported(i, f"During {event['topic']}, " + reference(other), 1),
                  "a requested topic prefix cannot qualify another source answer")
            check(not supported(i, f"The member {event['benefit']} but also {EVENTS[other]['harm']}.", 1),
                  "another source harm cannot qualify the requested harm")
            check(not supported(i, f"The member {EVENTS[other]['benefit']} but also {event['harm']}.", 1),
                  "another source benefit cannot qualify the requested benefit")
            check(not supported(i, f"During {event['topic']}, the member {EVENTS[other]['benefit']} but also {event['harm']}.", 1),
                  "a requested topic prefix cannot replace its supported benefit")


def accepted_rows(count):
    accepted, objects = [], []
    for i in range(count + 1):
        item = {}
        for part, key in enumerate(("source", "queue", "assessment", "relationship")):
            identity, version = 1000 + i * 4 + part, 1 + i * 4 + part
            row = f.object_row((1, 9, 3, 4)[part], identity, 100 + i, 1)
            f.put(row, 40, version); row[64:80] = row[80:96] = f.identity(9000); f.put(row, 176, 1, 4)
            objects.append((row, b"source bytes"))
            item[key], item[key + "_version"] = f.identity(identity).hex(), version
        accepted.append(item)
    return accepted, objects


def selection_fixture(count, ordinal, omit):
    accepted, objects = accepted_rows(count)
    requested, entries = targets(count, ordinal), []
    data = bytearray(64 + count * 13920); data[:8] = b"AOTXICH1"
    f.put(data, 8, count, 4); f.put(data, 40, 13920, 4)
    for i, target in enumerate(requested):
        text, conversation = recall(target, ordinal), f.identity(20000 + i)
        actor, owner = f.identity(100 + i), f.identity(9000)
        entry = dict(actor=i, target=target, conversation="con-fixture-" + conversation.hex(), body=dict(text=text),
                     receipt=dict(input_order=str(ordinal), space="spc-fixture-" + owner.hex(), actor=actor.hex()))
        entries.append(entry)
        row = memoryview(data)[64 + i * 13920:64 + (i + 1) * 13920]
        row[16:32] = conversation; f.put(row, 32, ordinal)
        query = row[64:8256]; query[:16] = f.identity(30000 + i); query[16:32] = query[32:48] = owner
        query[4640:4640 + len(text)] = text.encode(); f.put(query, 148, len(text), 4); f.put(query, 152, 1, 4)
        query[6688:6696] = b"AOTXCTX1"; f.put(query, 6696, 1, 4); f.put(query, 6700, 2, 4); f.put(query, 6728, 1000000, 4)
        source = f.object_row(1, 30000 + i, 100 + i, 1)
        source[64:80] = source[80:96] = owner; f.put(source, 176, 1, 4); objects.append((source, text.encode()))
        picked = source_group(accepted[-1]) | (set() if omit else source_group(accepted[target]))
        choice = row[13392:13920]; f.put(choice, 0, 1, 4); f.put(choice, 4, len(picked), 4)
        for k, (identity, version) in enumerate(sorted(picked)):
            choice[16 + k * 32:32 + k * 32] = identity; f.put(choice, 32 + k * 32, version)
    return [(14, bytes(16), data)], f.image(objects, 1, 1), entries, accepted


def selected(count):
    accepted, _ = accepted_rows(count)
    foreign = source_group(accepted[-1])
    for target in range(count):
        group = source_group(accepted[target])
        check(target_selected(group | foreign, accepted, target), "complete requested group remains selected with another group")
        check(not target_selected(foreign, accepted, target), "another complete group cannot replace the requested group")
        for member in group:
            check(not target_selected((group - {member}) | foreign, accepted, target), "every requested group member is required")
            changed = (group - {member}) | {(member[0], member[1] + 1)} | foreign
            check(not target_selected(changed, accepted, target), "every requested group version is exact")
    for ordinal in (1, 2):
        expected = tuple((i + (17 if ordinal == 1 else 29)) % count for i in range(count))
        check(targets(count, ordinal) == expected and set(expected) == set(range(count)), "exact distinct target rotation")
        check(all(i != target or count == 1 for i, target in enumerate(expected)), "recall crosses members at batch 64")
        for omit in (False, True):
            records, state, entries, admitted = selection_fixture(count, ordinal, omit)
            trace = Trace(); selections(trace, f, records, state, entries, admitted, True)
            recorded(trace, count if omit else 0)
    check((targets(count, 1) != targets(count, 2)) == (count > 1), "recovery uses a different target rotation")


class Client:
    def __init__(self, actor, space, ordinal):
        self.participant, self.space, self.ordinal = f.identity(100 + actor).hex(), space, ordinal
        self.calls = []

    async def answer(self, conversation, text, **fields):
        self.calls.append((conversation, text, fields))
        value = dict(id=conversation, actor=self.participant, space=self.space, operation=5,
                     input_order=str(self.ordinal), exact_output=base64.b64encode(b"Noted.").decode(), finish=2)
        return value, dict(text=text)

    async def request(self, method, path, **fields):
        return dict(id=path.split("/")[-2], output=dict(base64=base64.b64encode(b"Noted.").decode()))


def waves(count):
    space = "spc-fixture-" + f.identity(9000).hex()
    conversations = ["con-fixture-" + f.identity(20000 + i).hex() for i in range(count)]
    for ordinal in (0, 1, 2):
        clients = [Client(i, space, ordinal or 1) for i in range(count)]
        requested = targets(count, ordinal) if ordinal else None
        text = (lambda i: recall(requested[i], ordinal)) if ordinal else report
        trace = Trace()
        entries = asyncio.run(wave(trace, clients, conversations, space, text, ordinal or 1, 160, requested))
        recorded(trace)
        for i, client in enumerate(clients):
            check(len(client.calls) == 1 and client.calls[0][2]["max_output_tokens"] == (96 if ordinal else 32),
                  "recall and source input use their declared separate output allowances")
            check(entries[i].get("target") == (requested[i] if ordinal else None), "actual request metadata retains its target")
        if ordinal:
            for entry in entries:
                entry["receipt"]["finish"] = 1
                entry["receipt"]["exact_output"] = base64.b64encode(reference(entry["target"]).encode()).decode()
            trace = Trace(); useful(trace, entries); recorded(trace)
            entries[0]["receipt"]["finish"] = 2
            trace = Trace(); useful(trace, entries); recorded(trace, 1)
            for entry in entries:
                entry["receipt"]["finish"] = 1
                entry["receipt"]["exact_output"] = base64.b64encode(b"Noted.").decode()
            trace = Trace(); useful(trace, entries); recorded(trace, count)


def main():
    for count in (1, 64):
        before = checks
        answers(count); selected(count); waves(count)
        print(f"shared appraisal events N={count}: {checks - before} checks")
    print(f"shared appraisal events: {checks} checks, {failures} failures")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
