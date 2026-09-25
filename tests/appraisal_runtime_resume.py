# SPDX-License-Identifier: Apache-2.0
# Purpose: Check interrupted appraisal records before a resumed correction completes.
# Owns: Independent queue, source and empty interpretation checks.
# Threading: One disk test driver; CUDA restores and resumes actual work.
# Lifetime: A recorded interruption through ordinary journal recovery.

# Inputs: Memory and recorded result bytes. Output: Checks. Exit: Caller exceptions.
from appraisal_result_fixture import decode_result
from appraisal_runtime_cases import rows, source_id


def interruptions(test, f, state, recorded, count):
    available = rows(f, state)
    expected = {source_id(f, i, 3) for i in range(count)}
    for data in recorded:
        frame = decode_result(data)
        test.check(len(frame["rows"]) == count and frame["status"] == 11,
                   "recovered interruption has the exact full batch and denied status")
        tail = frame["tail"]
        test.check(tail[:8] == b"AOTXLOG1" and f.get(tail, 20, 4) == count and
                   f.get(tail, 32) == frame["sequence"] + 1 and f.get(tail, 80) == len(tail),
                   "interruption publishes only one queue mutation per source")
        changed = rows(f, tail); seen = set()
        for row in frame["rows"]:
            key = row["queue"], row["version"]
            test.check(key in available and not any(row["model"]) and not any(row["first_model"]) and
                       not row["reply"] and not row["first"] and not row["phase"] and not row["second_call"],
                       "recovery interruption contains no generated model output")
            prior, _ = available[key]; source = bytes(prior[96:112])
            test.check(source in expected and source not in seen, "interrupted sources cover each admitted correction once")
            seen.add(source)
            target = (key[0], key[1] + 1)
            test.check(target in changed and target in available, "interruption has its exact next queue version")
            entry, payload = changed[target]
            stored, stored_payload = available[target]
            test.check(entry[:160] == stored[:160] and entry[168:] == stored[168:] and payload == stored_payload and
                       payload[:8] == b"AOTXAPQ1" and
                       f.get(payload, 12, 4) == 3 and f.get(payload, 56, 4) == 11 and
                       entry[96:112] == source, "interrupted queue is preserved without an assessment or relationship")
        test.check(seen == expected, "the interruption covers the complete correction batch")
