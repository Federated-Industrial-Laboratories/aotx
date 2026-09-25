#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check independent readers for both saved appraisal result versions.
# Owns: Distinct output pairs, complete frames and damaged input controls.
# Threading: One test process checks batches of one and 64 rows.
# Lifetime: In-memory fixtures without model assets or runtime processes.

# Inputs: None. Output: Check tally. Exit: 0 pass, 1 failure.
import hashlib
import json
import struct

from appraisal_result_fixture import decode_result, decode_evidence

checks = 0


def check(value):
    global checks
    checks += 1
    assert value, checks


def put(data, at, value, size=4):
    data[at:at + size] = value.to_bytes(size, "little")


def frame(version, count, phase=2, second=1, status=0):
    stride = 4160 if version == 1 else 8320
    data = bytearray(64 + count * stride + 8)
    data[:8] = b"AOTXAPS1"
    struct.pack_into("<IIQQI", data, 8, version, count, 91, 8, status)
    for index in range(count):
        at = 64 + index * stride
        put(data, at, index + 1, 16); put(data, at + 16, index + 7, 8)
        model = hashlib.sha256(str(index).encode()).digest()
        data[at + 24:at + 56] = model
        reply = ('{"score":%d}' % index).encode() if version == 1 or phase == 2 and second else b""
        put(data, at + 56, len(reply)); put(data, at + 60, status)
        data[at + 64:at + 64 + len(reply)] = reply
        if version == 2:
            first = ('["event %d"]' % index).encode() if phase else b""
            put(data, at + 4160, len(first)); put(data, at + 4164, second); put(data, at + 4168, phase)
            if phase:
                data[at + 4192:at + 4224] = model
            data[at + 4224:at + 4224 + len(first)] = first
    data[-8:] = b"AOTXLOG1"
    return data


def refuses(data):
    try:
        decode_result(data)
    except ValueError:
        check(True)
    else:
        check(False)


def main():
    for count in (1, 64):
        for version in (1, 2):
            raw = frame(version, count); result = decode_result(raw)
            check(result["version"] == version and result["sequence"] == 91 and result["tail"] == b"AOTXLOG1")
            check(len(result["rows"]) == count)
            for index, row in enumerate(result["rows"]):
                check(row["queue"] == (index + 1).to_bytes(16, "little") and row["version"] == index + 7)
                check(row["model"] == hashlib.sha256(str(index).encode()).digest())
                check(row["reply"] == ('{"score":%d}' % index).encode())
                check(row["first"] == (('["event %d"]' % index).encode() if version == 2 else b""))
                if version == 2:
                    source = ("Source event %d. Not another actor's event." % index).encode()
                    check(decode_evidence(row["first"], source) == ["event %d" % index])
            last = 64 + (count - 1) * (4160 if version == 1 else 8320)
            for at, value, size in ((8, 3, 4), (12, 65, 4), (24, 0, 8), (32, 13, 4),
                                    (36, 1, 4), (last, 0, 16), (last + 16, 0, 8),
                                    (last + 56, 4097, 4), (last + 60, 11, 4), (last + 4159, 1, 1)):
                bad = bytearray(raw); put(bad, at, value, size); refuses(bad)
            refuses(raw[:-1]); refuses(raw + b"x")
            if count == 64:
                bad = bytearray(raw); bad[last:last + 16] = raw[64:80]; refuses(bad)
        raw = frame(2, count); last = 64 + (count - 1) * 8320
        for at, value in ((4160, 4097), (4164, 2), (4168, 3), (4172, 1), (4160, 0), (4164, 0), (4168, 1)):
            bad = bytearray(raw); put(bad, last + at, value); refuses(bad)
        bad = bytearray(raw); bad[last + 4192] ^= 1; refuses(bad)
        bad = bytearray(raw); bad[last + 8319] = 1; refuses(bad)
        for phase, second in ((0, 0), (1, 0), (2, 0), (2, 1)):
            result = decode_result(frame(2, count, phase, second, 11))
            for row in result["rows"]:
                check(row["phase"] == phase and row["second_call"] == second and row["status"] == 11)
                check(bool(row["first"]) == bool(phase))
        raw = frame(2, count, 1, 0, 11)
        put(raw, last + 56, 1); raw[last + 64] = 65; refuses(raw)
        for index in range(count):
            source = ("Report %d: I did not damage the item. Ren\u00e9 helped. Repeated. Repeated." % index).encode()
            quotes = ["I did not damage the item.", "Ren\u00e9 helped."]
            check(decode_evidence(json.dumps(quotes).encode(), source) == quotes)
            check(decode_evidence(b"[]", source) == [])
            for bad in (["invented"], ["Repeated."], [""], [quotes[0], quotes[0]],
                        [[quotes[0], 2]], [2], {}, quotes * 5):
                try:
                    decode_evidence(json.dumps(bad).encode(), source)
                except ValueError:
                    check(True)
                else:
                    check(False)
            for bad in (b"", b"[", b"[\"\\ud800\"]", b"[\"\xff\"]", b" " * 4097):
                try:
                    decode_evidence(bad, source)
                except ValueError:
                    check(True)
                else:
                    check(False)
    print("appraisal result fixture: %d checks, 0 failures" % checks)


if __name__ == "__main__":
    main()
