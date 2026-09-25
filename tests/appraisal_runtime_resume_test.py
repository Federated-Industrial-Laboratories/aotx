#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check full interrupted batches and defective source, version and output records.
# Inputs: None. Outputs: Check counts. Exit: 0 pass, 1 failed assertion.
import copy
import sys
sys.dont_write_bytecode = True
import recall_cli_test as f
from appraisal_runtime_cases import source_id
from appraisal_runtime_resume import interruptions


class Check:
    def __init__(self):
        self.count = 0

    def check(self, value, label, **details):
        self.count += 1
        if not value:
            raise AssertionError(label)


def fixture(count, version):
    before, changed = [], []
    stride = 4160 if version == 1 else 8320
    result = bytearray(64 + count * stride)
    result[:8] = b"AOTXAPS1"
    for offset, value, width in ((8, version, 4), (12, count, 4), (16, 1000, 8), (32, 11, 4)):
        f.put(result, offset, value, width)
    for i in range(count):
        row = f.object_row(12, 5000 + i, 10000 + i, i + 1)
        row[96:112] = source_id(f, i, 3)
        payload = bytearray(160); payload[:8] = b"AOTXAPQ1"
        f.put(payload, 8, 1, 4); f.put(payload, 12, 0, 4)
        before.append((row, payload))
        next_row, next_payload = copy.deepcopy((row, payload))
        f.put(next_row, 40, 2); f.put(next_payload, 12, 3, 4); f.put(next_payload, 56, 11, 4)
        changed.append((next_row, next_payload))
        at = 64 + i * stride; result[at:at + 16] = row[8:24]
        f.put(result, at + 16, 1); f.put(result, at + 60, 11, 4)
    tail = f.image(changed, 1001, 10, True); f.put(result, 24, len(tail))
    return f.image(before + changed, 1000 + count, 10), result + tail


def main():
    total = rejected = 0
    for count, version in ((1, 1), (64, 1), (1, 2), (64, 2)):
        state, raw = fixture(count, version); check = Check()
        interruptions(check, f, state, [raw], count); total += check.count
        mutations = []
        for offset, value, width in ((12, count + 1, 4), (32, 0, 4), (64 + 16, 2, 8),
                                     (64 + 24, 1, 1), (64 + 56, 1, 4), (64 + 64, 1, 1)):
            bad = bytearray(raw); f.put(bad, offset, value, width); mutations.append((state, bad))
        bad = bytearray(raw); bad[64:80] = f.identity(9999); mutations.append((state, bad))
        if count == 64:
            stride = 4160 if version == 1 else 8320
            bad = bytearray(raw); bad[64 + stride:64 + stride + 24] = raw[64:88]; mutations.append((state, bad))
        if version == 2:
            for offset in (4160, 4164, 4168, 4192, 4224):
                bad = bytearray(raw); f.put(bad, 64 + offset, 1, 1); mutations.append((state, bad))
        for image, bad in mutations:
            try:
                interruptions(Check(), f, image, [bad], count)
            except (AssertionError, ValueError):
                rejected += 1
            else:
                raise AssertionError("A defective interruption was accepted.")
    print(f"Interrupted batches: {total} checks, {rejected} defective records refused, 0 failures")


if __name__ == "__main__":
    main()
